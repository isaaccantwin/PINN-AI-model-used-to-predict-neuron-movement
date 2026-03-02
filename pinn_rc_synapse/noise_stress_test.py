"""
雜訊壓力測試 (Noise Stress Test)
=================================
在 ODE 生成的訓練數據中加入高斯雜訊：
  • V1, V2  : std = 1.0 mV
  • g12, g21 : std = 0.5 nS

目標：
  1. 驗證 PINN 物理殘差 (RC ODE Loss) 是否能過濾雜訊
  2. 比較「髒數據點 (Dots)」與「AI 平滑曲線 (Line)」
  3. 確認 2000 步 Rollout 在有雜訊輸入的情況下依然穩定振盪

輸出：
  noise_stress_result.png  ← 比較圖 + 分析報告
"""

import sys, os
os.environ["PYTHONUTF8"] = "1"
sys.stdout.reconfigure(encoding="utf-8")

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from pathlib import Path

from dual_neuron_model import (
    DualNeuronParams,
    DualNeuronPINN,
    DualNeuronPINNLoss,
    DualNeuronTrainer,
)

# ─────────────────────────────────────────────
#  隨機種子 & 裝置
# ─────────────────────────────────────────────
torch.manual_seed(0)
np.random.seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"裝置: {DEVICE}")

# ─────────────────────────────────────────────
#  物理參數（與 dual_neuron_train.py 完全相同）
# ─────────────────────────────────────────────
DT      = 0.1
T_END   = 200.0
N_STEPS = int(T_END / DT)   # 2000

params = DualNeuronParams(
    tau_m=20.0, E_rest=-70.0, C_m=100.0,
    E_exc=0.0,  E_inh=-80.0,
    tau_s=5.0,  g_max=20.0, V_thresh=-55.0, beta=0.5,
    I_ext1=100.0, I_ext2=10.0,
    dt=DT,
    V_min=-85.0, V_max=-30.0,
    g_min=0.0, g_max_norm=25.0,
)

# 雜訊規格
V_NOISE_STD = 1.0   # mV
G_NOISE_STD = 0.5   # nS

# ─────────────────────────────────────────────
#  生成 ODE 乾淨參考軌跡
# ─────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def syn_drive_np(V):
    return sigmoid(params.beta * (V - params.V_thresh))

def dual_neuron_ode(t, y):
    V1, g12, V2, g21 = y
    dV1  = (-(V1-params.E_rest)/params.tau_m + params.I_ext1/params.C_m
            - g21/params.C_m*(V1-params.E_inh))
    dg12 = -g12/params.tau_s + (params.g_max/params.tau_s)*syn_drive_np(V1)
    dV2  = (-(V2-params.E_rest)/params.tau_m + params.I_ext2/params.C_m
            + g12/params.C_m*(params.E_exc-V2))
    dg21 = -g21/params.tau_s + (params.g_max/params.tau_s)*syn_drive_np(V2)
    return [dV1, dg12, dV2, dg21]

V1_0 = -70.0; g12_0 = 0.0; V2_0 = -70.0; g21_0 = 0.0
t_eval = np.linspace(0.0, T_END, N_STEPS + 1)

print("[*] 正在生成 ODE 乾淨參考軌跡 ...")
sol = solve_ivp(
    dual_neuron_ode, [0.0, T_END],
    [V1_0, g12_0, V2_0, g21_0],
    t_eval=t_eval, method="RK45", rtol=1e-9, atol=1e-11,
)
V1_clean  = sol.y[0]; g12_clean = sol.y[1]
V2_clean  = sol.y[2]; g21_clean = sol.y[3]
print(f"  V1 動態範圍: {V1_clean.max()-V1_clean.min():.2f} mV | "
      f"V2 動態範圍: {V2_clean.max()-V2_clean.min():.2f} mV")

# ─────────────────────────────────────────────
#  注入高斯雜訊 → 「髒數據」
# ─────────────────────────────────────────────
print(f"\n[*] 注入高斯雜訊: V1/V2 std={V_NOISE_STD} mV, g12/g21 std={G_NOISE_STD} nS")
rng = np.random.default_rng(42)
V1_noisy  = V1_clean  + rng.normal(0, V_NOISE_STD, size=V1_clean.shape)
V2_noisy  = V2_clean  + rng.normal(0, V_NOISE_STD, size=V2_clean.shape)
g12_noisy = np.clip(g12_clean + rng.normal(0, G_NOISE_STD, size=g12_clean.shape), 0, None)
g21_noisy = np.clip(g21_clean + rng.normal(0, G_NOISE_STD, size=g21_clean.shape), 0, None)

# SNR 計算
def snr_db(clean, noisy):
    noise = noisy - clean
    sig_power = np.mean(clean**2)
    noise_power = np.mean(noise**2)
    return 10 * np.log10(sig_power / noise_power) if noise_power > 0 else float("inf")

snr_V1  = snr_db(V1_clean,  V1_noisy)
snr_V2  = snr_db(V2_clean,  V2_noisy)
snr_g12 = snr_db(g12_clean, g12_noisy)
snr_g21 = snr_db(g21_clean, g21_noisy)
print(f"  SNR — V1: {snr_V1:.1f} dB | V2: {snr_V2:.1f} dB | "
      f"g12: {snr_g12:.1f} dB | g21: {snr_g21:.1f} dB")

# ─────────────────────────────────────────────
#  建構訓練對（current → next），使用「髒數據」
# ─────────────────────────────────────────────
V1_c_np  = V1_noisy[:-1];  g12_c_np = g12_noisy[:-1]
V2_c_np  = V2_noisy[:-1];  g21_c_np = g21_noisy[:-1]
# 目標 next 也是髒的（讓模型學習從一個雜訊狀態預測下一個雜訊狀態）
V1_n_np  = V1_noisy[1:];   g12_n_np = g12_noisy[1:]
V2_n_np  = V2_noisy[1:];   g21_n_np = g21_noisy[1:]
print(f"  訓練樣本對數: {len(V1_c_np)} (髒數據)")

# ─────────────────────────────────────────────
#  建立模型 & 訓練
# ─────────────────────────────────────────────
model = DualNeuronPINN(
    params=params,
    hidden_layers=4,
    hidden_dim=128,
    activation="swish",
    delta_V_scale=0.5,   # 稍微放寬 NN 修正範圍，讓它能學習平滑化
    delta_g_scale=0.2,
)

loss_fn = DualNeuronPINNLoss(
    params=params,
    lambda_data=1.0,
    lambda_V=1.0,     # 強化物理殘差權重以抵抗雜訊
    lambda_syn=1.0,   # 強化突觸因果約束
)

trainer = DualNeuronTrainer(
    model=model, loss_fn=loss_fn, lr=1e-3, device=DEVICE,
)

print("\n[>>] 開始訓練（髒數據 + 強化物理約束）...")
history = trainer.train(
    epochs=10000,
    V1_c_np=V1_c_np, g12_c_np=g12_c_np, V2_c_np=V2_c_np, g21_c_np=g21_c_np,
    V1_n_np=V1_n_np, g12_n_np=g12_n_np, V2_n_np=V2_n_np, g21_n_np=g21_n_np,
    batch_size=512,
    log_every=1000,
)

# ─────────────────────────────────────────────
#  2000 步 Rollout（從乾淨初始條件開始）
# ─────────────────────────────────────────────
print(f"\n[>>] 執行 {N_STEPS} 步自迴歸 Rollout（乾淨初始條件）...")

# 含 NN 物理過濾的 rollout
V1_nn, g12_nn, V2_nn, g21_nn = trainer.rollout(
    V1_0=V1_0, g12_0=g12_0, V2_0=V2_0, g21_0=g21_0,
    n_steps=N_STEPS,
)
# 純 RK4 作為 baseline
V1_rk4, g12_rk4, V2_rk4, g21_rk4 = trainer.rollout_rk4(
    V1_0=V1_0, g12_0=g12_0, V2_0=V2_0, g21_0=g21_0,
    n_steps=N_STEPS,
)
print(f"  [完成] Rollout 結束，共 {N_STEPS} 步")

# 穩定性檢驗
L = len(V1_nn)
t_plot  = t_eval[:L]
V1_range_nn = V1_nn.max() - V1_nn.min()
V2_range_nn = V2_nn.max() - V2_nn.min()
print(f"  PINN Rollout 動態範圍: V1={V1_range_nn:.2f} mV | V2={V2_range_nn:.2f} mV")
stable = (V1_range_nn > 1.0) and (V2_range_nn > 1.0)
print(f"  振盪穩定性: {'[穩定]' if stable else '[不穩定 / 崩潰]'}")

# ─────────────────────────────────────────────
#  計算去雜訊效能
# ─────────────────────────────────────────────
# PINN 平滑預測 vs 乾淨參考
V1_denoised_err  = np.abs(V1_nn  - V1_clean[:L])
V2_denoised_err  = np.abs(V2_nn  - V2_clean[:L])
g12_denoised_err = np.abs(g12_nn - g12_clean[:L])
g21_denoised_err = np.abs(g21_nn - g21_clean[:L])

# 髒數據 vs 乾淨參考
V1_noise_err  = np.abs(V1_noisy[:L]  - V1_clean[:L])
V2_noise_err  = np.abs(V2_noisy[:L]  - V2_clean[:L])
g12_noise_err = np.abs(g12_noisy[:L] - g12_clean[:L])
g21_noise_err = np.abs(g21_noisy[:L] - g21_clean[:L])

print("\n" + "=" * 70)
print("  去雜訊效能比較 (PINN 預測 vs 髒數據)")
print("=" * 70)
print(f"  V1  | 髒數據誤差: {V1_noise_err.mean():.4f} mV | PINN誤差: {V1_denoised_err.mean():.4f} mV | "
      f"改善: {100*(1-V1_denoised_err.mean()/V1_noise_err.mean()):.1f}%")
print(f"  V2  | 髒數據誤差: {V2_noise_err.mean():.4f} mV | PINN誤差: {V2_denoised_err.mean():.4f} mV | "
      f"改善: {100*(1-V2_denoised_err.mean()/V2_noise_err.mean()):.1f}%")
print(f"  g12 | 髒數據誤差: {g12_noise_err.mean():.4f} nS | PINN誤差: {g12_denoised_err.mean():.4f} nS | "
      f"改善: {100*(1-g12_denoised_err.mean()/g12_noise_err.mean()):.1f}%")
print(f"  g21 | 髒數據誤差: {g21_noise_err.mean():.4f} nS | PINN誤差: {g21_denoised_err.mean():.4f} nS | "
      f"改善: {100*(1-g21_denoised_err.mean()/g21_noise_err.mean()):.1f}%")
print("=" * 70)

# 計算降噪比
V1_improvement  = V1_noise_err.mean()  / max(V1_denoised_err.mean(), 1e-8)
V2_improvement  = V2_noise_err.mean()  / max(V2_denoised_err.mean(), 1e-8)
g12_improvement = g12_noise_err.mean() / max(g12_denoised_err.mean(), 1e-8)
g21_improvement = g21_noise_err.mean() / max(g21_denoised_err.mean(), 1e-8)

# ─────────────────────────────────────────────
#  視覺化：髒數據 (Dots) vs AI 平滑曲線 (Line)
# ─────────────────────────────────────────────
# 為了可視化清晰，僅繪製前 100ms（1000 步）
PLOT_END = 100.0
PLOT_STEPS = int(PLOT_END / DT)

fig = plt.figure(figsize=(18, 20))
fig.patch.set_facecolor("#0E1117")
_suptitle = (
    "雜訊壓力測試 — PINN 物理過濾器\n"
    "Noise Stress Test: Dirty Data (Dots) vs AI Physics-Filtered Prediction (Lines)\n"
    f"V noise: σ={V_NOISE_STD} mV | g noise: σ={G_NOISE_STD} nS | 訓練: 10000 epoch"
)
fig.suptitle(_suptitle, fontsize=12, fontweight="bold", color="white", y=0.98)

gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)
t_short = t_eval[:PLOT_STEPS + 1]

# 共用樣式
DIRTY_STYLE  = dict(alpha=0.25, s=4, zorder=1)
CLEAN_STYLE  = dict(color="#00BFFF", lw=1.0, ls="--", alpha=0.7, zorder=3, label="ODE Clean")
PINN_STYLE   = dict(lw=2.2, zorder=4)
RK4_STYLE    = dict(lw=1.4, ls=":", alpha=0.6, zorder=2)

def ax_style(ax, title, xlabel, ylabel):
    ax.set_facecolor("#1A1D23")
    ax.set_title(title, fontsize=10, color="white", pad=6)
    ax.set_xlabel(xlabel, fontsize=9, color="#AAAAAA")
    ax.set_ylabel(ylabel, fontsize=9, color="#AAAAAA")
    ax.tick_params(colors="#888888", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")
    ax.grid(alpha=0.12, color="#555555")

# [A] V1 — 前 100ms
ax_a = fig.add_subplot(gs[0, :])
ax_a.scatter(t_short, V1_noisy[:PLOT_STEPS+1],
             c="#FF6B6B", label=f"Dirty V1 (σ={V_NOISE_STD}mV)", **DIRTY_STYLE)
ax_a.plot(t_short, V1_clean[:PLOT_STEPS+1], **CLEAN_STYLE)
ax_a.plot(t_short, V1_nn[:PLOT_STEPS+1], color="#FF3333",
          label="PINN Filtered V1 (N1)", **PINN_STYLE)
ax_a.plot(t_short, V1_rk4[:PLOT_STEPS+1], color="#FF9999",
          label="Pure RK4", **RK4_STYLE)
ax_style(ax_a, "[A] N1 膜電位 V1 — 髒數據 vs AI 平滑曲線 (前 100ms)",
         "Time (ms)", "V1 (mV)")
ax_a.legend(fontsize=8, loc="upper right",
            facecolor="#111111", labelcolor="white", framealpha=0.8)

# [B] V2 — 前 100ms
ax_b = fig.add_subplot(gs[1, :])
ax_b.scatter(t_short, V2_noisy[:PLOT_STEPS+1],
             c="#69B3FF", label=f"Dirty V2 (σ={V_NOISE_STD}mV)", **DIRTY_STYLE)
ax_b.plot(t_short, V2_clean[:PLOT_STEPS+1], **CLEAN_STYLE)
ax_b.plot(t_short, V2_nn[:PLOT_STEPS+1], color="#1E90FF",
          label="PINN Filtered V2 (N2)", **PINN_STYLE)
ax_b.plot(t_short, V2_rk4[:PLOT_STEPS+1], color="#89CFF0",
          label="Pure RK4", **RK4_STYLE)
ax_style(ax_b, "[B] N2 膜電位 V2 — 髒數據 vs AI 平滑曲線 (前 100ms)",
         "Time (ms)", "V2 (mV)")
ax_b.legend(fontsize=8, loc="upper right",
            facecolor="#111111", labelcolor="white", framealpha=0.8)

# [C] g12 & g21 — 前 100ms
ax_c = fig.add_subplot(gs[2, 0])
ax_c.scatter(t_short, g12_noisy[:PLOT_STEPS+1],
             c="#FFD700", label=f"Dirty g12 (σ={G_NOISE_STD}nS)", **DIRTY_STYLE)
ax_c.plot(t_short, g12_clean[:PLOT_STEPS+1], color="#00BFFF",
          lw=1.0, ls="--", alpha=0.7, zorder=3)
ax_c.plot(t_short, g12_nn[:PLOT_STEPS+1], color="#FFD700",
          label="PINN g12", **PINN_STYLE)

ax_c.scatter(t_short, g21_noisy[:PLOT_STEPS+1],
             c="#DA70D6", label=f"Dirty g21 (σ={G_NOISE_STD}nS)", **DIRTY_STYLE)
ax_c.plot(t_short, g21_clean[:PLOT_STEPS+1], color="#00CED1",
          lw=1.0, ls="--", alpha=0.7, zorder=3)
ax_c.plot(t_short, g21_nn[:PLOT_STEPS+1], color="#DA70D6",
          label="PINN g21", **PINN_STYLE)
ax_style(ax_c, "[C] 突觸電導 (前 100ms)", "Time (ms)", "Conductance (nS)")
ax_c.legend(fontsize=7, facecolor="#111111", labelcolor="white", framealpha=0.8)

# [D] 2000 步全程 V1 Rollout 穩定性
ax_d = fig.add_subplot(gs[2, 1])
ax_d.plot(t_plot, V1_clean[:L], color="#00BFFF", lw=1.0, ls="--",
          alpha=0.6, label="ODE Clean (2000 steps)")
ax_d.plot(t_plot, V1_nn,  color="#FF3333", lw=1.5, alpha=0.9,
          label="PINN V1 (2000 steps)")
ax_style(ax_d, f"[D] 2000步全程 V1 Rollout 穩定性\n動態範圍: {V1_range_nn:.2f} mV — "
         f"{'✓ 穩定' if stable else '✗ 崩潰'}",
         "Time (ms)", "V1 (mV)")
ax_d.legend(fontsize=7, facecolor="#111111", labelcolor="white", framealpha=0.8)

# [E] 去雜訊誤差比較
ax_e = fig.add_subplot(gs[3, 0])
labels = ["V1", "V2", "g12", "g21"]
dirty_vals  = [V1_noise_err.mean(),  V2_noise_err.mean(),
               g12_noise_err.mean(), g21_noise_err.mean()]
pinn_vals   = [V1_denoised_err.mean(), V2_denoised_err.mean(),
               g12_denoised_err.mean(), g21_denoised_err.mean()]
x = np.arange(len(labels))
w = 0.35
bars1 = ax_e.bar(x - w/2, dirty_vals, w, color="#FF6B6B", alpha=0.8, label="Noisy Data Error")
bars2 = ax_e.bar(x + w/2, pinn_vals,  w, color="#4ECDC4", alpha=0.8, label="PINN Prediction Error")
ax_e.set_xticks(x)
ax_e.set_xticklabels(labels)
ax_style(ax_e, "[E] PINN 去雜訊效能", "Variable", "MAE (mV or nS)")
ax_e.legend(fontsize=8, facecolor="#111111", labelcolor="white", framealpha=0.8)
for bar in bars1: ax_e.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                             f"{bar.get_height():.3f}", ha="center", va="bottom",
                             fontsize=7, color="#FF6B6B")
for bar in bars2: ax_e.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                             f"{bar.get_height():.3f}", ha="center", va="bottom",
                             fontsize=7, color="#4ECDC4")

# [F] 訓練損失曲線
ax_f = fig.add_subplot(gs[3, 1])
epochs_list = list(range(1, len(history)+1))
ax_f.semilogy(epochs_list, [d["total"]   for d in history], color="#FFFFFF",
              lw=1.5, label="Total Loss")
ax_f.semilogy(epochs_list, [d["L_R_V1"]  for d in history], color="#FF6B6B",
              lw=1.0, alpha=0.8, label="R_V1 (RC ODE N1)")
ax_f.semilogy(epochs_list, [d["L_R_V2"]  for d in history], color="#6B9FFF",
              lw=1.0, alpha=0.8, label="R_V2 (RC ODE N2)")
ax_f.semilogy(epochs_list, [d["L_R_g12"] for d in history], color="#FFD700",
              lw=1.0, alpha=0.8, label="R_g12 (Causal)")
ax_f.semilogy(epochs_list, [d["L_R_g21"] for d in history], color="#DA70D6",
              lw=1.0, alpha=0.8, label="R_g21 (Causal)")
ax_f.semilogy(epochs_list, [d["L_data"]  for d in history], color="#4ECDC4",
              lw=1.0, alpha=0.8, label="Data Loss (noisy)")
ax_style(ax_f, "[F] 訓練損失曲線（物理殘差 vs 資料損失）",
         "Epoch", "Loss (log scale)")
ax_f.legend(fontsize=7, facecolor="#111111", labelcolor="white", framealpha=0.8)

out_path = Path("noise_stress_result.png")
plt.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\n[完成] 結果圖已儲存至: {out_path.resolve()}")

# ─────────────────────────────────────────────
#  分析報告
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PINN 雜訊壓力測試 — 分析報告")
print("=" * 70)

print(f"""
【實驗設定】
  訓練雜訊: V1/V2 ← N(0, {V_NOISE_STD}²) mV,  g12/g21 ← N(0, {G_NOISE_STD}²) nS
  物理損失權重: λ_V=1.0, λ_syn=1.0, λ_data=1.0
  訓練輪次: 10,000 epochs

【雜訊特性 (SNR)】
  V1: {snr_V1:.1f} dB | V2: {snr_V2:.1f} dB | g12: {snr_g12:.1f} dB | g21: {snr_g21:.1f} dB

【PINN 物理過濾效能】
  V1  去雜訊改善: {100*(1-V1_denoised_err.mean()/V1_noise_err.mean()):.1f}%
      (誤差 {V1_noise_err.mean():.4f} mV → {V1_denoised_err.mean():.4f} mV)
  V2  去雜訊改善: {100*(1-V2_denoised_err.mean()/V2_noise_err.mean()):.1f}%
      (誤差 {V2_noise_err.mean():.4f} mV → {V2_denoised_err.mean():.4f} mV)
  g12 去雜訊改善: {100*(1-g12_denoised_err.mean()/g12_noise_err.mean()):.1f}%
      (誤差 {g12_noise_err.mean():.4f} nS → {g12_denoised_err.mean():.4f} nS)
  g21 去雜訊改善: {100*(1-g21_denoised_err.mean()/g21_noise_err.mean()):.1f}%
      (誤差 {g21_noise_err.mean():.4f} nS → {g21_denoised_err.mean():.4f} nS)

【2000 步 Rollout 穩定性】
  V1 動態範圍: {V1_range_nn:.2f} mV (參考: {V1_clean.max()-V1_clean.min():.2f} mV)
  V2 動態範圍: {V2_range_nn:.2f} mV (參考: {V2_clean.max()-V2_clean.min():.2f} mV)
  結論: {'[✓ 穩定振盪，未崩潰]' if stable else '[✗ 振盪崩潰]'}

【PINN 物理約束是否真的幫助過濾雜訊？】
""")

# 判斷過濾效果
avg_V_improve = (
    100*(1-V1_denoised_err.mean()/V1_noise_err.mean()) +
    100*(1-V2_denoised_err.mean()/V2_noise_err.mean())
) / 2

if avg_V_improve > 50 and stable:
    conclusion = ("✅ 是的！PINN 的物理約束（RC ODE 殘差 + 突觸因果約束）"
                  "顯著地幫助模型過濾了隨機雜訊。\n"
                  f"   平均電位去雜訊改善率: {avg_V_improve:.1f}%\n"
                  "   物理理由：RC ODE 是一個低通濾波器，"
                  "高斯雜訊無法滿足微分方程約束，\n"
                  "   因此模型被強制只學習符合物理規律的底層訊號。\n"
                  "   Rollout 依然保持穩定振盪，證明過濾成功。")
elif avg_V_improve > 10:
    conclusion = (f"⚠️  PINN 物理約束有一定程度的去雜訊效果 ({avg_V_improve:.1f}%)，\n"
                  "   但改善幅度不顯著。這可能是因為雜訊強度接近訊號強度，\n"
                  "   或物理損失權重需要進一步調整。")
else:
    conclusion = ("❌ 去雜訊效果有限。可能原因：\n"
                  "   1. 雜訊標準差 (1.0mV) 相對訊號幅度過大\n"
                  "   2. 需要增大 λ_V, λ_syn 以強化物理約束\n"
                  "   3. 訓練輪次不足")

print(f"  {conclusion}")
print("\n" + "=" * 70)
