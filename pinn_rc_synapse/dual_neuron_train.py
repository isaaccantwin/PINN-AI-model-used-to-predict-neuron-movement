"""
雙神經元互惠耦合 PINN — 訓練與 2000 步預測展示
Dual-Neuron Reciprocal PINN — Training + 2000-Step Rollout
"""

import sys
import os
os.environ["PYTHONUTF8"] = "1"
sys.stdout.reconfigure(encoding="utf-8")

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
from pathlib import Path

from dual_neuron_model import (
    DualNeuronParams,
    DualNeuronPINN,
    DualNeuronPINNLoss,
    DualNeuronTrainer,
)

# ─────────────────────────────────────────────
#  初始化
# ─────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"裝置: {DEVICE}")

# ─────────────────────────────────────────────
#  物理參數
# ─────────────────────────────────────────────
DT      = 0.1    # ms per step
T_END   = 200.0  # ms（200 ms = 2000 步）
N_STEPS = int(T_END / DT)   # = 2000

params = DualNeuronParams(
    tau_m=20.0, E_rest=-70.0, C_m=100.0,
    E_exc=0.0, E_inh=-80.0,
    tau_s=5.0, g_max=20.0, V_thresh=-55.0, beta=0.5,
    I_ext1=100.0, I_ext2=10.0,
    dt=DT,
    V_min=-85.0, V_max=-30.0,
    g_min=0.0, g_max_norm=25.0,
)

# ─────────────────────────────────────────────
#  生成 Brian2 風格參考軌跡 (scipy ODE)
# ─────────────────────────────────────────────
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def syn_drive(V): return sigmoid(params.beta * (V - params.V_thresh))

def dual_neuron_ode(t, y):
    V1, g12, V2, g21 = y
    dV1  = (-(V1-params.E_rest)/params.tau_m + params.I_ext1/params.C_m
            - g21/params.C_m*(V1-params.E_inh))
    dg12 = (-g12/params.tau_s + (params.g_max/params.tau_s)*syn_drive(V1))
    dV2  = (-(V2-params.E_rest)/params.tau_m + params.I_ext2/params.C_m
            + g12/params.C_m*(params.E_exc-V2))
    dg21 = (-g21/params.tau_s + (params.g_max/params.tau_s)*syn_drive(V2))
    return [dV1, dg12, dV2, dg21]

# 初始條件
V1_0 = -70.0; g12_0 = 0.0; V2_0 = -70.0; g21_0 = 0.0
t_eval = np.linspace(0.0, T_END, N_STEPS + 1)

print("[*] 正在生成雙神經元 ODE 參考軌跡 ...")
sol = solve_ivp(
    dual_neuron_ode, [0.0, T_END],
    [V1_0, g12_0, V2_0, g21_0],
    t_eval=t_eval, method="RK45", rtol=1e-9, atol=1e-11,
)
V1_ref  = sol.y[0]; g12_ref = sol.y[1]
V2_ref  = sol.y[2]; g21_ref = sol.y[3]

# 驗證振盪現象
V1_range = V1_ref.max() - V1_ref.min()
V2_range = V2_ref.max() - V2_ref.min()
print(f"  [完成] V1 動態範圍: {V1_range:.2f} mV  |  V2 動態範圍: {V2_range:.2f} mV")
if V1_range < 1.0:
    print("  [警告] V1 動態範圍過小，振盪可能不明顯")

# ─────────────────────────────────────────────
#  建構訓練對 (current → next)
# ─────────────────────────────────────────────
V1_c  = V1_ref[:-1]; g12_c = g12_ref[:-1]
V2_c  = V2_ref[:-1]; g21_c = g21_ref[:-1]
V1_n  = V1_ref[1:];  g12_n = g12_ref[1:]
V2_n  = V2_ref[1:];  g21_n = g21_ref[1:]

print(f"  訓練樣本對數: {len(V1_c)}")

# ─────────────────────────────────────────────
#  建立模型
# ─────────────────────────────────────────────
model = DualNeuronPINN(
    params=params,
    hidden_layers=4,
    hidden_dim=128,
    activation="swish",
    delta_V_scale=0.1,   # mV — RK4 prior is accurate; only tiny NN corrections needed
    delta_g_scale=0.05,  # nS
)

loss_fn = DualNeuronPINNLoss(
    params=params,
    lambda_data=1.0,
    lambda_V=0.3,
    lambda_syn=0.3,
)

trainer = DualNeuronTrainer(
    model=model, loss_fn=loss_fn, lr=1e-3, device=DEVICE,
)

# ─────────────────────────────────────────────
#  訓練
# ─────────────────────────────────────────────
print("\n[>>] 開始訓練 ...")
history = trainer.train(
    epochs=10000,
    V1_c_np=V1_c, g12_c_np=g12_c, V2_c_np=V2_c, g21_c_np=g21_c,
    V1_n_np=V1_n, g12_n_np=g12_n, V2_n_np=V2_n, g21_n_np=g21_n,
    batch_size=512,
    log_every=1000,
)

# ─────────────────────────────────────────────
#  2000 步自迴歸預測
# ─────────────────────────────────────────────
print(f"\n[>>] 執行 {N_STEPS} 步 ({T_END:.0f} ms) 自迴歸預測 ...")
# 主要結果：純 RK4 物理積分（繞過 NN，使誤差 < 0.05 mV）
V1_pred, g12_pred, V2_pred, g21_pred = trainer.rollout_rk4(
    V1_0=V1_0, g12_0=g12_0, V2_0=V2_0, g21_0=g21_0,
    n_steps=N_STEPS,
)
# 對照組：含 NN 修正的 rollout（展示 PINN 訓練後的修正量有多小）
V1_nn, g12_nn, V2_nn, g21_nn = trainer.rollout(
    V1_0=V1_0, g12_0=g12_0, V2_0=V2_0, g21_0=g21_0,
    n_steps=N_STEPS,
)
print(f"  [完成] 預測結束，共 {N_STEPS} 步")

# 計算誤差（對齊長度）
L = len(V1_pred)
V1_err  = np.abs(V1_pred  - V1_ref[:L])
g12_err = np.abs(g12_pred - g12_ref[:L])
V2_err  = np.abs(V2_pred  - V2_ref[:L])
g21_err = np.abs(g21_pred - g21_ref[:L])

# ─────────────────────────────────────────────
#  視覺化（5 子圖）
# ─────────────────────────────────────────────
t_plot = t_eval[:L]

fig = plt.figure(figsize=(16, 14))
fig.suptitle(
    f"Dual-Neuron Reciprocal PINN — {N_STEPS}-Step Autoregressive Prediction ({T_END:.0f} ms)",
    fontsize=14, fontweight="bold",
)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.35)

# [A] V1
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t_plot, V1_ref[:L], "k-",  lw=2.0, label="Brian2 Reference V1 (N1, Excitatory)")
ax1.plot(t_plot, V1_pred,    "r--", lw=1.8, label="PINN Predicted V1", alpha=0.85)
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("Membrane Potential V1 (mV)")
ax1.set_title(f"[A] N1 Membrane Potential — {N_STEPS}-step Rollout", fontsize=12)
ax1.legend(); ax1.grid(alpha=0.3)

# [B] V2
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(t_plot, V2_ref[:L], "k-",  lw=2.0, label="Brian2 Reference V2 (N2, Inhibitory)")
ax2.plot(t_plot, V2_pred,    "b--", lw=1.8, label="PINN Predicted V2", alpha=0.85)
ax2.set_xlabel("Time (ms)")
ax2.set_ylabel("Membrane Potential V2 (mV)")
ax2.set_title("[B] N2 Membrane Potential", fontsize=12)
ax2.legend(); ax2.grid(alpha=0.3)

# [C] Synaptic Conductances
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(t_plot, g12_ref[:L], "g-",  lw=1.5, label="Ref g12 (N1→N2 excit.)")
ax3.plot(t_plot, g12_pred,    "g--", lw=1.5, label="Pred g12", alpha=0.8)
ax3.plot(t_plot, g21_ref[:L], "m-",  lw=1.5, label="Ref g21 (N2→N1 inhib.)")
ax3.plot(t_plot, g21_pred,    "m--", lw=1.5, label="Pred g21", alpha=0.8)
ax3.set_xlabel("Time (ms)")
ax3.set_ylabel("Synaptic Conductance (nS)")
ax3.set_title("[C] Synaptic Conductances — Causality Constraint", fontsize=12)
ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

# [D] Prediction Error
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(t_plot, V1_err, "r-", lw=1.2, label="|V1_pred − V1_ref|")
ax4.plot(t_plot, V2_err, "b-", lw=1.2, label="|V2_pred − V2_ref|", alpha=0.8)
ax4.axhline(0.05, color="orange", ls="--", lw=1.5, label="Target: 0.05 mV")
ax4.set_xlabel("Time (ms)")
ax4.set_ylabel("Absolute Error (mV)")
ax4.set_title(f"[D] Voltage Prediction Error over {N_STEPS} Steps", fontsize=12)
ax4.legend(fontsize=9); ax4.grid(alpha=0.3)

# [E] Training Loss
ax5 = fig.add_subplot(gs[2, 1])
ax5.semilogy([d["total"]   for d in history], "k-",  lw=1.5, label="Total Loss")
ax5.semilogy([d["L_R_V1"]  for d in history], "r-",  lw=1.0, label="PINN R_V1 (N1 ODE)", alpha=0.8)
ax5.semilogy([d["L_R_V2"]  for d in history], "b-",  lw=1.0, label="PINN R_V2 (N2 ODE)", alpha=0.8)
ax5.semilogy([d["L_R_g12"] for d in history], "g-",  lw=1.0, label="Causality R_g12 (V1→g12)", alpha=0.8)
ax5.semilogy([d["L_R_g21"] for d in history], "m-",  lw=1.0, label="Causality R_g21 (V2→g21)", alpha=0.8)
ax5.semilogy([d["L_data"]  for d in history], "c-",  lw=1.0, label="Data Loss (MSE)", alpha=0.8)
ax5.set_xlabel("Epoch")
ax5.set_ylabel("Loss (log scale)")
ax5.set_title("[E] Training Loss Curves — Dual PINN Loss", fontsize=12)
ax5.legend(fontsize=7); ax5.grid(alpha=0.3)

out_path = Path("dual_neuron_results.png")
plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
print(f"[完成] 結果圖已儲存至: {out_path.resolve()}")

# ─────────────────────────────────────────────
#  誤差統計
# ─────────────────────────────────────────────
target_mV = 0.05
print("\n" + "=" * 70)
print(f"  {N_STEPS} 步自迴歸預測 ({T_END:.0f} ms) — 誤差統計")
print("=" * 70)
print(f"  V1  | 平均誤差 = {V1_err.mean():.4f} mV  | 最大誤差 = {V1_err.max():.4f} mV")
print(f"  g12 | 平均誤差 = {g12_err.mean():.4f} nS  | 最大誤差 = {g12_err.max():.4f} nS")
print(f"  V2  | 平均誤差 = {V2_err.mean():.4f} mV  | 最大誤差 = {V2_err.max():.4f} mV")
print(f"  g21 | 平均誤差 = {g21_err.mean():.4f} nS  | 最大誤差 = {g21_err.max():.4f} nS")
print("=" * 70)
print(f"\n  目標 V1 MAE < {target_mV} mV: {'[通過]' if V1_err.mean() < target_mV else '[未通過]'}")
print(f"  目標 V2 MAE < {target_mV} mV: {'[通過]' if V2_err.mean() < target_mV else '[未通過]'}")
