"""
逐步 PINN 預測器 — 訓練與 1000 步連續預測展示
================================================================
1. 使用 scipy 生成「Brian2 風格」RC 電路 ODE 數據
2. 建構 (V_current, g_current, I_input) → (V_next, g_next) 訓練對
3. 以 PINN Loss 訓練 StepPredictorPINN
4. 執行 1000 步自迴歸預測，並與參考 ODE 軌跡比較
"""

import sys
import os
os.environ["PYTHONUTF8"] = "1"          # force UTF-8 for subprocesses too
sys.stdout.reconfigure(encoding="utf-8") # force UTF-8 output in Windows terminal

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless rendering (no display needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
from pathlib import Path

from step_predictor_model import (
    StepPhysicsParams,
    StepPredictorPINN,
    StepPredictorLoss,
    StepPredictorTrainer,
)

# ─────────────────────────────────────────────
#  Seeds & Device
# ─────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ─────────────────────────────────────────────
#  Physical Parameters
# ─────────────────────────────────────────────
DT    = 0.1     # ms per step  (matches Brian2-style dt)
T_END = 100.0   # ms total simulation time
N_STEPS = int(T_END / DT)          # = 1000

params = StepPhysicsParams(
    tau_m=20.0,
    E_rest=-70.0,
    E_syn=0.0,
    C_m=100.0,
    tau_s=5.0,
    dt=DT,
    # I/O normalisation ranges (set from known physics)
    V_min=-80.0,  V_max=-50.0,
    g_min=0.0,    g_max=15.0,
    I_min=-0.5,   I_max=0.5,       # nA range for external current
)

# ─────────────────────────────────────────────
#  Generate Reference ODE Data (Brian2-style)
# ─────────────────────────────────────────────
V0_TRUE = -70.0   # mV
G0_TRUE = 10.0    # nS

# External current profile: Gaussian bump + small sinusoidal ripple (nA)
t_full = np.arange(0, T_END + DT, DT)          # (N_STEPS+1,)
I_ext  = (
    0.3 * np.exp(-((t_full - 30.0) ** 2) / (2 * 5.0 ** 2))   # Gaussian pulse at t=30ms
    + 0.1 * np.sin(2 * np.pi * t_full / 20.0)                 # 50 Hz ripple
)

def neuron_ode_with_I(t, y):
    """Full ODE with time-varying injected current (interpolated)."""
    V, g = y
    I = float(np.interp(t, t_full, I_ext))
    dV_dt = (
        -(V - params.E_rest) / params.tau_m
        + (g / params.C_m) * (params.E_syn - V)
        + I / params.C_m
    )
    dg_dt = -g / params.tau_s
    return [dV_dt, dg_dt]

print("[*] 正在生成 Brian2 風格的參考 ODE 軌跡 ...")
sol = solve_ivp(
    neuron_ode_with_I,
    [0.0, T_END],
    [V0_TRUE, G0_TRUE],
    t_eval=t_full,
    method="RK45",
    rtol=1e-9,
    atol=1e-11,
)
V_ref = sol.y[0]   # (N_STEPS+1,)
g_ref = sol.y[1]   # (N_STEPS+1,)
print(f"  [完成] 參考軌跡已生成（共 {len(V_ref)} 個點）")

# ─────────────────────────────────────────────
#  Build Training Pairs
#  current step → next step
# ─────────────────────────────────────────────
V_c = V_ref[:-1]        # (N_STEPS,) current V
g_c = g_ref[:-1]        # (N_STEPS,) current g
I_s = I_ext[:-1]        # (N_STEPS,) current I_input
V_n = V_ref[1:]         # (N_STEPS,) next V  (target)
g_n = g_ref[1:]         # (N_STEPS,) next g  (target)

print(f"  訓練樣本對數：{len(V_c)}")

# ─────────────────────────────────────────────
#  Build Model
# ─────────────────────────────────────────────
model = StepPredictorPINN(
    params=params,
    hidden_layers=4,
    hidden_dim=64,
    activation="swish",
    delta_V_scale=2.0,   # mV — correction range up to ±2 mV per step
    delta_g_scale=0.5,   # nS — correction range up to ±0.5 nS per step
)

loss_fn = StepPredictorLoss(
    params=params,
    lambda_data=1.0,
    lambda_V=0.3,
    lambda_g=0.3,
)

trainer = StepPredictorTrainer(
    model=model,
    loss_fn=loss_fn,
    lr=1e-3,
    device=DEVICE,
)

# ─────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────
print("\n[>>] 開始訓練 ...")
history = trainer.train(
    epochs=8000,
    V_c_data=V_c,
    g_c_data=g_c,
    I_data=I_s,
    V_n_data=V_n,
    g_n_data=g_n,
    batch_size=256,
    log_every=500,
)

# ─────────────────────────────────────────────
#  1000-Step Auto-regressive Rollout
# ─────────────────────────────────────────────
print(f"\n[>>] 執行 {N_STEPS} 步自迴歸預測 ...")
V_pred, g_pred = trainer.rollout(
    V0=V0_TRUE,
    g0=G0_TRUE,
    I_seq=I_ext[:-1],   # 每步餵入參考電流
)
print(f"  [完成] 自迴歸預測結束（共預測 {N_STEPS} 步）")

# Align lengths for comparison
t_compare  = t_full[:len(V_pred)]
V_ref_cmp  = V_ref [:len(V_pred)]
g_ref_cmp  = g_ref [:len(V_pred)]

V_err = np.abs(V_pred - V_ref_cmp)
g_err = np.abs(g_pred - g_ref_cmp)

# ─────────────────────────────────────────────
#  Visualization (4-panel figure)
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(16, 11))
fig.suptitle(
    f"Step-by-Step PINN Predictor — {N_STEPS}-Step Auto-regressive Rollout (RTX 4070)",
    fontsize=15, fontweight="bold",
)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

# ── [A] Membrane Potential V(t) ─────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t_compare, V_ref_cmp, "k-",  linewidth=2.0, label="Brian2 Reference V")
ax1.plot(t_compare, V_pred,    "r--", linewidth=1.8, label="PINN Step Predictor V", alpha=0.85)
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("Membrane Potential V (mV)")
ax1.set_title(f"[A] Membrane Potential V(t) — {N_STEPS}-step Autoregressive Rollout", fontsize=12)
ax1.legend()
ax1.grid(alpha=0.3)

# ── [B] Conductance g(t) ───────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(t_compare, g_ref_cmp, "k-",  linewidth=2.0, label="Brian2 Reference g")
ax2.plot(t_compare, g_pred,    "b--", linewidth=1.8, label="PINN Step Predictor g", alpha=0.85)
ax2.set_xlabel("Time (ms)")
ax2.set_ylabel("Synaptic Conductance g (nS)")
ax2.set_title("[B] Synaptic Conductance g(t) — First-Order Decay", fontsize=12)
ax2.legend()
ax2.grid(alpha=0.3)

# ── [C] Prediction Errors ──────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(t_compare, V_err, "r-", linewidth=1.2, label="|V_pred − V_ref| (mV)")
ax3.plot(t_compare, g_err, "b-", linewidth=1.2, label="|g_pred − g_ref| (nS)", alpha=0.8)
ax3.set_xlabel("Time (ms)")
ax3.set_ylabel("Absolute Error")
ax3.set_title(f"[C] Prediction Error over {N_STEPS} Steps", fontsize=12)
ax3.legend()
ax3.grid(alpha=0.3)

# ── [D] Training Loss Curves ───────────────
ax4 = fig.add_subplot(gs[2, :])
ax4.semilogy([d["total"]       for d in history], color="black", lw=1.5, label="Total Loss")
ax4.semilogy([d["L_physics_V"] for d in history], color="red",   lw=1.2, label="PINN RC ODE Residual (V)", alpha=0.8)
ax4.semilogy([d["L_physics_g"] for d in history], color="green", lw=1.2, label="PINN g Decay Residual", alpha=0.8)
ax4.semilogy([d["L_data"]      for d in history], color="blue",  lw=1.2, label="Data Loss (MSE)", alpha=0.8)
ax4.set_xlabel("Epoch")
ax4.set_ylabel("Loss (log scale)")
ax4.set_title("[D] Training Loss Curves", fontsize=12)
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)

out_path = Path("step_predictor_results.png")
plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
print(f"[完成] 結果圖已儲存至：{out_path.resolve()}")

# ─────────────────────────────────────────────
#  Final Statistics
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  {N_STEPS} 步自迴歸預測 — 誤差統計")
print("=" * 60)
print(f"  V  |  平均絕對誤差 = {V_err.mean():.4f} mV  |  最大誤差 = {V_err.max():.4f} mV")
print(f"  g  |  平均絕對誤差 = {g_err.mean():.4f} nS  |  最大誤差 = {g_err.max():.4f} nS")
print("=" * 60)
print(f"\n  V 平均誤差 < 2 mV：{'[通過]' if V_err.mean() < 2.0 else '[未通過]'}")
print(f"  g 平均誤差 < 1 nS：{'[通過]' if g_err.mean() < 1.0 else '[未通過]'}")
