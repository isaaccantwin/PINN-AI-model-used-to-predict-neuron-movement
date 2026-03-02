"""
雜訊壓力測試動畫
Noise Stress Test — Animation
==============================
直接使用 noise_stress_test.py 的數據管線：
  • 以相同種子/參數生成「髒數據」
  • 用已訓練的 DualNeuronPINN rollout 取得「AI 平滑曲線」
  • 四子圖動畫:
      左上: 示波器 — 髒數據點(Dots) vs PINN 平滑曲線(Line)
      右上: 相位平面 — 極限環拖尾
      下左: 突觸電導 g12/g21
      下右: 2000 步 Rollout 穩定性掃描

輸出: noise_oscillation.mp4 (or .gif)
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
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

from dual_neuron_model import (
    DualNeuronParams,
    DualNeuronPINN,
    DualNeuronPINNLoss,
    DualNeuronTrainer,
)

# ─────────────────────────────────────────────
#  設定（與 noise_stress_test.py 完全相同）
# ─────────────────────────────────────────────
torch.manual_seed(0)
np.random.seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DT      = 0.1
T_END   = 200.0
N_STEPS = int(T_END / DT)   # 2000

V_NOISE_STD = 1.0   # mV
G_NOISE_STD = 0.5   # nS

params = DualNeuronParams(
    tau_m=20.0, E_rest=-70.0, C_m=100.0,
    E_exc=0.0,  E_inh=-80.0,
    tau_s=5.0,  g_max=20.0, V_thresh=-55.0, beta=0.5,
    I_ext1=100.0, I_ext2=10.0,
    dt=DT,
    V_min=-85.0, V_max=-30.0,
    g_min=0.0, g_max_norm=25.0,
)

# ─────────────────────────────────────────────
#  Step 1: ODE 乾淨軌跡
# ─────────────────────────────────────────────
def _sig(x):    return 1.0 / (1.0 + np.exp(-x))
def _sd(V):     return _sig(params.beta * (V - params.V_thresh))

def dual_ode(t, y):
    V1, g12, V2, g21 = y
    p = params
    dV1  = -(V1-p.E_rest)/p.tau_m + p.I_ext1/p.C_m  - g21/p.C_m*(V1-p.E_inh)
    dg12 = -g12/p.tau_s + (p.g_max/p.tau_s)*_sd(V1)
    dV2  = -(V2-p.E_rest)/p.tau_m + p.I_ext2/p.C_m  + g12/p.C_m*(p.E_exc-V2)
    dg21 = -g21/p.tau_s + (p.g_max/p.tau_s)*_sd(V2)
    return [dV1, dg12, dV2, dg21]

V1_0 = -70.0; g12_0 = 0.0; V2_0 = -70.0; g21_0 = 0.0
t_eval = np.linspace(0.0, T_END, N_STEPS + 1)

print("[1/4] 生成 ODE 乾淨軌跡 ...")
sol = solve_ivp(dual_ode, [0.0, T_END], [V1_0, g12_0, V2_0, g21_0],
                t_eval=t_eval, method="RK45", rtol=1e-9, atol=1e-11)
V1_clean = sol.y[0]; g12_clean = sol.y[1]
V2_clean = sol.y[2]; g21_clean = sol.y[3]

# ─────────────────────────────────────────────
#  Step 2: 注入高斯雜訊
# ─────────────────────────────────────────────
print(f"[2/4] 注入高斯雜訊 (V: {V_NOISE_STD}mV, g: {G_NOISE_STD}nS) ...")
rng = np.random.default_rng(42)
V1_noisy  = V1_clean  + rng.normal(0, V_NOISE_STD, size=V1_clean.shape)
V2_noisy  = V2_clean  + rng.normal(0, V_NOISE_STD, size=V2_clean.shape)
g12_noisy = np.clip(g12_clean + rng.normal(0, G_NOISE_STD, size=g12_clean.shape), 0, None)
g21_noisy = np.clip(g21_clean + rng.normal(0, G_NOISE_STD, size=g21_clean.shape), 0, None)

# ─────────────────────────────────────────────
#  Step 3: 訓練 PINN（髒數據）
# ─────────────────────────────────────────────
print("[3/4] 訓練 DualNeuronPINN (10000 epochs, 髒數據) ...")
model = DualNeuronPINN(params=params, hidden_layers=4, hidden_dim=128,
                        activation="swish", delta_V_scale=0.5, delta_g_scale=0.2)
loss_fn = DualNeuronPINNLoss(params=params,
                              lambda_data=1.0, lambda_V=1.0, lambda_syn=1.0)
trainer = DualNeuronTrainer(model=model, loss_fn=loss_fn, lr=1e-3, device=DEVICE)

history = trainer.train(
    epochs=10000,
    V1_c_np=V1_noisy[:-1],  g12_c_np=g12_noisy[:-1],
    V2_c_np=V2_noisy[:-1],  g21_c_np=g21_noisy[:-1],
    V1_n_np=V1_noisy[1:],   g12_n_np=g12_noisy[1:],
    V2_n_np=V2_noisy[1:],   g21_n_np=g21_noisy[1:],
    batch_size=512, log_every=2000,
)

# ─────────────────────────────────────────────
#  Step 4: PINN 2000 步 Rollout
# ─────────────────────────────────────────────
print("[4/4] 執行 2000 步 PINN Rollout ...")
V1_nn, g12_nn, V2_nn, g21_nn = trainer.rollout(
    V1_0=V1_0, g12_0=g12_0, V2_0=V2_0, g21_0=g21_0, n_steps=N_STEPS)
print(f"      PINN V1 範圍: {V1_nn.min():.1f}~{V1_nn.max():.1f} mV")
print(f"      PINN V2 範圍: {V2_nn.min():.1f}~{V2_nn.max():.1f} mV")

# ─────────────────────────────────────────────
#  動畫設定
# ─────────────────────────────────────────────
SKIP     = 5          # 下採樣（400 幀）
WIN_PTS  = 150        # 示波器視窗（點數）
FPS      = 30
TRAIL    = 120        # 相位平面拖尾長度

idx_s    = np.arange(0, N_STEPS + 1, SKIP)
t_s      = t_eval[idx_s]
V1_s_d   = V1_noisy[idx_s]   # 髒
V2_s_d   = V2_noisy[idx_s]
g12_s_d  = g12_noisy[idx_s]
g21_s_d  = g21_noisy[idx_s]
V1_s_p   = V1_nn[idx_s]      # PINN
V2_s_p   = V2_nn[idx_s]
g12_s_p  = g12_nn[idx_s]
g21_s_p  = g21_nn[idx_s]
N_frames = len(idx_s)

# ─────────────────────────────────────────────
#  顏色
# ─────────────────────────────────────────────
BG    = "#0A0A14"; GRID  = "#1A1A2E"
C_N1  = "#00CFFF"; C_N2  = "#FF8C00"
C_D1  = "#FF6B6B"; C_D2  = "#85B3FF"   # 髒數據點
C_G12 = "#00FFAA"; C_G21 = "#FF4488"
C_CLN = "#446688"                       # 乾淨 ODE 參考

# ─────────────────────────────────────────────
#  建立圖形
# ─────────────────────────────────────────────
plt.style.use("dark_background")
fig = plt.figure(figsize=(20, 10), facecolor=BG)
fig.patch.set_facecolor(BG)

gs = gridspec.GridSpec(2, 2, figure=fig,
                       hspace=0.46, wspace=0.32,
                       left=0.06, right=0.97,
                       top=0.90, bottom=0.07)

ax_osc   = fig.add_subplot(gs[0, 0])   # 示波器
ax_phase = fig.add_subplot(gs[0, 1])   # 相位平面
ax_syn   = fig.add_subplot(gs[1, 0])   # 突觸電導
ax_roll  = fig.add_subplot(gs[1, 1])   # 全程 Rollout

for ax in [ax_osc, ax_phase, ax_syn, ax_roll]:
    ax.set_facecolor(BG)
    ax.spines[:].set_color("#2A2A4A")
    ax.tick_params(colors="#7777AA", labelsize=8)
    ax.xaxis.label.set_color("#AAAACC")
    ax.yaxis.label.set_color("#AAAACC")
    ax.grid(color=GRID, linewidth=0.5, linestyle="--", alpha=0.7)

fig.suptitle(
    "Noise Stress Test — PINN Physics Filter  |  Dirty Data (Dots) vs AI Smooth Prediction (Lines)",
    fontsize=12, color="#DDD8FF", fontweight="bold", y=0.96, fontfamily="monospace"
)

# ── [A] 示波器  ──────────────────────────────
_ylo = min(V1_noisy.min(), V2_noisy.min()) - 1
_yhi = max(V1_noisy.max(), V2_noisy.max()) + 1
ax_osc.set_xlim(0, WIN_PTS * DT * SKIP)
ax_osc.set_ylim(_ylo, _yhi)
ax_osc.set_xlabel("Time window (ms)", fontsize=9)
ax_osc.set_ylabel("Membrane Potential (mV)", fontsize=9)
ax_osc.set_title("Oscilloscope — Noise(Dots) vs PINN(Lines)", color="#AAAACC", fontsize=10)

# 髒數據散布點
scat_V1 = ax_osc.scatter([], [], c=C_D1, s=5, alpha=0.35, zorder=2, label=f"Dirty V1 (σ={V_NOISE_STD}mV)")
scat_V2 = ax_osc.scatter([], [], c=C_D2, s=5, alpha=0.35, zorder=2, label=f"Dirty V2 (σ={V_NOISE_STD}mV)")
# PINN 平滑線
line_V1, = ax_osc.plot([], [], color=C_N1, lw=2.0, zorder=4, label="PINN V1")
line_V2, = ax_osc.plot([], [], color=C_N2, lw=2.0, zorder=4, label="PINN V2")
time_txt = ax_osc.text(0.02, 0.94, "", transform=ax_osc.transAxes,
                        color="#CCCCEE", fontsize=9, fontfamily="monospace")
ax_osc.legend(fontsize=7, loc="upper right", framealpha=0.0,
               labelcolor=[C_D1, C_D2, C_N1, C_N2])
scan_osc = ax_osc.axvline(x=0, color="#445566", lw=0.8)

# ── [B] 相位平面  ─────────────────────────────
_xpad = (V1_nn.max()-V1_nn.min())*0.08
_ypad = (V2_nn.max()-V2_nn.min())*0.08
ax_phase.set_xlim(V1_nn.min()-_xpad, V1_nn.max()+_xpad)
ax_phase.set_ylim(V2_nn.min()-_ypad, V2_nn.max()+_ypad)
ax_phase.set_xlabel("N1  V1 (mV)", fontsize=9)
ax_phase.set_ylabel("N2  V2 (mV)", fontsize=9)
ax_phase.set_title("Phase Plane — Limit Cycle  (PINN)", color="#AAAACC", fontsize=10)
# 暗背景完整軌跡
ax_phase.plot(V1_s_p, V2_s_p, color="#1E1E3A", lw=4.0, zorder=1)
phase_trail, = ax_phase.plot([], [], color="#9955FF", lw=2.0, alpha=0.7, zorder=2)
phase_dot,   = ax_phase.plot([], [], "o", color="#FFFFFF", ms=7, zorder=5,
                               mfc="white", mec=C_N1, mew=1.5)
_arrow = [None]

# ── [C] 突觸電導  ─────────────────────────────
ax_syn.set_xlim(t_s[0], t_s[-1])
_gmax = max(g12_noisy.max(), g21_noisy.max()) * 1.08
ax_syn.set_ylim(-0.3, _gmax)
ax_syn.set_xlabel("Time (ms)", fontsize=9)
ax_syn.set_ylabel("Conductance (nS)", fontsize=9)
ax_syn.set_title("Synaptic Dynamics — g12 (N1→N2) & g21 (N2→N1)", color="#AAAACC", fontsize=10)
# 全程背景（極淡）
ax_syn.scatter(t_s, g12_s_d, c=C_G12, s=3, alpha=0.10, zorder=1)
ax_syn.scatter(t_s, g21_s_d, c=C_G21, s=3, alpha=0.10, zorder=1)
ax_syn.plot(t_s, g12_s_d, color=C_G12, lw=0.4, alpha=0.08)
ax_syn.plot(t_s, g21_s_d, color=C_G21, lw=0.4, alpha=0.08)

scat_g12 = ax_syn.scatter([], [], c=C_G12, s=5, alpha=0.35, zorder=2, label=f"Dirty g12")
scat_g21 = ax_syn.scatter([], [], c=C_G21, s=5, alpha=0.35, zorder=2, label=f"Dirty g21")
line_g12, = ax_syn.plot([], [], color=C_G12, lw=1.8, zorder=4, label="PINN g12")
line_g21, = ax_syn.plot([], [], color=C_G21, lw=1.8, zorder=4, label="PINN g21")
scan_syn = ax_syn.axvline(x=0, color="#445566", lw=0.8)
ax_syn.legend(fontsize=7, loc="upper right", framealpha=0.0,
               labelcolor=[C_G12, C_G21, C_G12, C_G21])

# ── [D] 全程 2000 步 Rollout  ──────────────────
ax_roll.set_xlim(t_eval[0], t_eval[-1])
ax_roll.set_ylim(min(V1_nn.min(), V2_nn.min())-1, max(V1_nn.max(), V2_nn.max())+1)
ax_roll.set_xlabel("Time (ms)", fontsize=9)
ax_roll.set_ylabel("Membrane Potential (mV)", fontsize=9)
ax_roll.set_title("2000-Step Rollout Stability (PINN)", color="#AAAACC", fontsize=10)
# 預畫完整參考曲線（極淡）
ax_roll.plot(t_eval, V1_clean, color=C_CLN, lw=1.0, alpha=0.30, ls="--")
ax_roll.plot(t_eval, V2_clean, color=C_CLN, lw=1.0, alpha=0.30, ls="--")
roll_V1, = ax_roll.plot([], [], color=C_N1, lw=1.6, alpha=0.9, label="PINN V1")
roll_V2, = ax_roll.plot([], [], color=C_N2, lw=1.6, alpha=0.9, label="PINN V2")
scan_roll = ax_roll.axvline(x=0, color="#445566", lw=0.8)
ax_roll.legend(fontsize=8, loc="upper right", framealpha=0.0,
                labelcolor=[C_N1, C_N2])
# 動態範圍文字標註
roll_txt = ax_roll.text(0.02, 0.06, "", transform=ax_roll.transAxes,
                         color="#AABBCC", fontsize=8, fontfamily="monospace")

# ─────────────────────────────────────────────
#  動畫更新函式
# ─────────────────────────────────────────────
def init():
    line_V1.set_data([], [])
    line_V2.set_data([], [])
    scat_V1.set_offsets(np.empty((0, 2)))
    scat_V2.set_offsets(np.empty((0, 2)))
    phase_trail.set_data([], [])
    phase_dot.set_data([], [])
    line_g12.set_data([], [])
    line_g21.set_data([], [])
    scat_g12.set_offsets(np.empty((0, 2)))
    scat_g21.set_offsets(np.empty((0, 2)))
    roll_V1.set_data([], [])
    roll_V2.set_data([], [])
    return (line_V1, line_V2, scat_V1, scat_V2,
            phase_trail, phase_dot, line_g12, line_g21,
            scat_g12, scat_g21, roll_V1, roll_V2,
            time_txt, scan_osc, scan_syn, scan_roll, roll_txt)

def update(frame):
    i = frame

    # ── 示波器（滾動視窗） ───────────────────────
    win_start = max(0, i - WIN_PTS)
    sl = slice(win_start, i + 1)
    t_win = t_s[sl] - t_s[win_start]

    line_V1.set_data(t_win, V1_s_p[sl])
    line_V2.set_data(t_win, V2_s_p[sl])
    scat_V1.set_offsets(np.c_[t_win, V1_s_d[sl]])
    scat_V2.set_offsets(np.c_[t_win, V2_s_d[sl]])
    scan_osc.set_xdata([t_win[-1] if len(t_win) > 0 else 0])
    time_txt.set_text(f"t = {t_s[i]:.1f} ms")

    # ── 相位平面拖尾 ─────────────────────────────
    ts = max(0, i - TRAIL)
    trail_sl = slice(ts, i + 1)
    phase_trail.set_data(V1_s_p[trail_sl], V2_s_p[trail_sl])
    n_trail = i - ts
    if n_trail > 1:
        phase_trail.set_alpha(0.45 + 0.45 * min(1.0, n_trail / TRAIL))
    phase_dot.set_data([V1_s_p[i]], [V2_s_p[i]])
    # 方向箭頭
    if _arrow[0] is not None:
        _arrow[0].remove(); _arrow[0] = None
    if i >= 2:
        dv1 = V1_s_p[i] - V1_s_p[i-1]
        dv2 = V2_s_p[i] - V2_s_p[i-1]
        if abs(dv1) + abs(dv2) > 0.001:
            _arrow[0] = ax_phase.annotate(
                "", xy=(V1_s_p[i]+dv1*3, V2_s_p[i]+dv2*3),
                xytext=(V1_s_p[i], V2_s_p[i]),
                arrowprops=dict(arrowstyle="-|>", color=C_N1, lw=1.5, mutation_scale=12),
                zorder=6)

    # ── 突觸電導 ─────────────────────────────────
    syn_sl = slice(0, i + 1)
    line_g12.set_data(t_s[syn_sl], g12_s_p[syn_sl])
    line_g21.set_data(t_s[syn_sl], g21_s_p[syn_sl])
    scat_g12.set_offsets(np.c_[t_s[syn_sl], g12_s_d[syn_sl]])
    scat_g21.set_offsets(np.c_[t_s[syn_sl], g21_s_d[syn_sl]])
    scan_syn.set_xdata([t_s[i]])

    # ── 全程 Rollout 掃描 ─────────────────────────
    roll_sl = slice(0, idx_s[i] + 1)
    roll_V1.set_data(t_eval[roll_sl], V1_nn[roll_sl])
    roll_V2.set_data(t_eval[roll_sl], V2_nn[roll_sl])
    scan_roll.set_xdata([t_eval[idx_s[i]]])
    rng_v1 = V1_nn[:idx_s[i]+1].max() - V1_nn[:idx_s[i]+1].min() if idx_s[i] > 0 else 0
    rng_v2 = V2_nn[:idx_s[i]+1].max() - V2_nn[:idx_s[i]+1].min() if idx_s[i] > 0 else 0
    roll_txt.set_text(f"V1 range: {rng_v1:.1f}mV  V2 range: {rng_v2:.1f}mV")

    return (line_V1, line_V2, scat_V1, scat_V2,
            phase_trail, phase_dot, line_g12, line_g21,
            scat_g12, scat_g21, roll_V1, roll_V2,
            time_txt, scan_osc, scan_syn, scan_roll, roll_txt)

# ─────────────────────────────────────────────
#  建立 & 儲存動畫
# ─────────────────────────────────────────────
print(f"\n[>>] 建立動畫（{N_frames} 幀，{FPS} fps）...")
ani = animation.FuncAnimation(
    fig, update, frames=N_frames,
    init_func=init, interval=1000/FPS, blit=False,
)

out_mp4 = "noise_oscillation.mp4"
out_gif = "noise_oscillation.gif"

saved = False
try:
    writer = animation.FFMpegWriter(fps=FPS, bitrate=1400,
                                     extra_args=["-vcodec", "libx264",
                                                 "-pix_fmt", "yuv420p"])
    ani.save(out_mp4, writer=writer, dpi=110,
             savefig_kwargs={"facecolor": BG})
    print(f"[完成] 動畫儲存至: {out_mp4}")
    saved = True
except Exception as e:
    print(f"[!] MP4 儲存失敗 ({e})，改用 GIF ...")

if not saved:
    writer_gif = animation.PillowWriter(fps=FPS)
    ani.save(out_gif, writer=writer_gif, dpi=100,
             savefig_kwargs={"facecolor": BG})
    print(f"[完成] 動畫儲存至: {out_gif}")

plt.close()
print("[完成] animate_results.py 執行結束")
