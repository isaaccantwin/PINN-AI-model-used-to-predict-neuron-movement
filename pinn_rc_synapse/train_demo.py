"""
PINNs Training Demo Script
==========================================
Train and validate an RC Circuit + Synaptic Conductance PINNs model
using synthetic data.
"""


import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering for Windows terminal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from pinn_model import (
    NeuronPINN,
    PhysicsParams,
    PINNLoss,
    PINNTrainer,
)

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# Physics parameters
params = PhysicsParams(
    tau_m=20.0,    # membrane time constant 20 ms
    E_rest=-70.0,  # resting potential -70 mV
    E_syn=0.0,     # excitatory synaptic reversal potential 0 mV
    C_m=100.0,     # membrane capacitance 100 pF
    tau_s=5.0,     # synaptic conductance decay time constant 5 ms
)

# Time range: 0 ~ 100 ms
T_START, T_END = 0.0, 100.0
V0_TRUE = -70.0   # initial membrane potential (mV)
G0_TRUE = 10.0    # initial synaptic conductance (nS)

# ─────────────────────────────────────────────
#  Generate analytical solution (used as "observed data")
# ─────────────────────────────────────────────
# Synaptic conductance analytical solution: g(t) = g0 * exp(-t / tau_s)
# Membrane potential: no simple analytical solution; solved numerically via scipy

from scipy.integrate import solve_ivp

def neuron_ode(t, y):
    """Full ODE system for scipy solver."""
    V, g = y
    dV_dt = -(V - params.E_rest) / params.tau_m + (g / params.C_m) * (params.E_syn - V)
    dg_dt = -g / params.tau_s
    return [dV_dt, dg_dt]

print("[*] Using scipy to generate analytical data...")
t_eval = np.linspace(T_START, T_END, 500)
sol = solve_ivp(
    neuron_ode,
    [T_START, T_END],
    [V0_TRUE, G0_TRUE],
    t_eval=t_eval,
    method="RK45",
    rtol=1e-8,
    atol=1e-10,
)
V_true = sol.y[0]   # shape (500,)
g_true = sol.y[1]   # shape (500,)

# ─────────────────────────────────────────────
#  Prepare PyTorch tensors
# ─────────────────────────────────────────────

# Collocation points (dense sampling for physics residual training)
N_colloc = 1000
t_colloc_np = np.random.uniform(T_START, T_END, (N_colloc, 1)).astype(np.float32)
t_colloc = torch.tensor(t_colloc_np, requires_grad=True, device=DEVICE)

# Simulated observation data (sparse, with noise)
N_obs = 50
idx_obs = np.random.choice(len(t_eval), N_obs, replace=False)
t_data_np = t_eval[idx_obs].astype(np.float32).reshape(-1, 1)
V_data_np = (V_true[idx_obs] + np.random.randn(N_obs) * 0.5).astype(np.float32).reshape(-1, 1)
g_data_np = (g_true[idx_obs] + np.random.randn(N_obs) * 0.1).astype(np.float32).reshape(-1, 1)

t_data = torch.tensor(t_data_np, device=DEVICE)
V_data = torch.tensor(V_data_np, device=DEVICE)
g_data = torch.tensor(g_data_np, device=DEVICE)

# Initial condition
t0 = torch.tensor([[T_START]], dtype=torch.float32, device=DEVICE)

# ─────────────────────────────────────────────
#  Build model, loss function, and trainer
# ─────────────────────────────────────────────
model = NeuronPINN(
    hidden_layers=5,
    hidden_dim=64,
    activation="swish",
    T_scale=100.0,   # normalise t to [0,1]
    V_offset=-70.0,  # network output centred around E_rest
    V_scale=30.0,    # expected peak depolarisation ~30 mV above rest
    g_scale=10.0,    # expected peak conductance ~10 nS
)

loss_fn = PINNLoss(
    params=params,
    lambda_data=1.0,
    lambda_V=1.0,    # RC residual — matched to data loss weight
    lambda_g=1.0,    # conductance decay residual — matched to data loss weight
    lambda_ic=10.0,  # IC anchoring — normalised outputs make this effective without dominating
)

trainer = PINNTrainer(model=model, loss_fn=loss_fn, lr=1e-3, device=DEVICE)

# ─────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────
print("\n[>>] Starting PINNs model training...")
history = trainer.train(
    epochs=20000,
    t_colloc=t_colloc,
    t_data=t_data,
    V_data=V_data,
    g_data=g_data,
    t0=t0,
    V0=V0_TRUE,
    g0=G0_TRUE,
    log_every=500,
)

# ─────────────────────────────────────────────
#  Evaluation and Visualization
# ─────────────────────────────────────────────
print("\n[>>] Generating predictions...")

t_test_np = np.linspace(T_START, T_END, 500).astype(np.float32).reshape(-1, 1)
t_test = torch.tensor(t_test_np, device=DEVICE)
V_pred, g_pred = trainer.predict(t_test)

# ── Build figure ────────────────────────────────
fig = plt.figure(figsize=(15, 10))
fig.suptitle("PINNs RC Circuit + Synaptic Conductance Model", fontsize=16, fontweight="bold")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# [1] Membrane Potential V(t)
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(t_eval, V_true, "k-", linewidth=2, label="Analytical Solution (True V)")
ax1.plot(t_test_np.squeeze(), V_pred.squeeze(), "r--", linewidth=2, label="PINNs Predicted V")
ax1.scatter(t_data_np.squeeze(), V_data_np.squeeze(), s=20, c="blue", alpha=0.5, label="Observed Data (noisy)")
ax1.set_xlabel("Time t (ms)")
ax1.set_ylabel("Membrane Potential V (mV)")
ax1.set_title("Membrane Potential V(t) — RC Circuit ODE")
ax1.legend()
ax1.grid(alpha=0.3)

# [2] Synaptic Conductance g(t)
ax2 = fig.add_subplot(gs[1, :2])
ax2.plot(t_eval, g_true, "k-", linewidth=2, label="Analytical Solution (True g)")
ax2.plot(t_test_np.squeeze(), g_pred.squeeze(), "r--", linewidth=2, label="PINNs Predicted g")
ax2.scatter(t_data_np.squeeze(), g_data_np.squeeze(), s=20, c="green", alpha=0.5, label="Observed Data (noisy)")
ax2.set_xlabel("Time t (ms)")
ax2.set_ylabel("Synaptic Conductance g (nS)")
ax2.set_title("Synaptic Conductance g(t) — First-Order Decay ODE")
ax2.legend()
ax2.grid(alpha=0.3)

# [3] Training Loss Curves
ax3 = fig.add_subplot(gs[0, 2])
epochs = range(1, len(history) + 1)
ax3.semilogy([d.get("total", 0) for d in history], label="Total Loss", color="black")
ax3.semilogy([d.get("L_physics_V", 0) for d in history], label="RC ODE Residual", color="red", alpha=0.7)
ax3.semilogy([d.get("L_physics_g", 0) for d in history], label="g Decay Residual", color="green", alpha=0.7)
ax3.semilogy([d.get("L_data", 0) for d in history], label="Data Loss", color="blue", alpha=0.7)
ax3.semilogy([d.get("L_ic", 0) for d in history], label="IC Loss", color="orange", alpha=0.7)
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Loss (log scale)")
ax3.set_title("Training Loss Curves")
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# [4] Prediction Error Analysis
ax4 = fig.add_subplot(gs[1, 2])
V_err = np.abs(V_pred.squeeze() - np.interp(t_test_np.squeeze(), t_eval, V_true))
g_err = np.abs(g_pred.squeeze() - np.interp(t_test_np.squeeze(), t_eval, g_true))
ax4.plot(t_test_np.squeeze(), V_err, "r-", label="|V_pred - V_true|")
ax4.plot(t_test_np.squeeze(), g_err, "g-", label="|g_pred - g_true|")
ax4.set_xlabel("Time t (ms)")
ax4.set_ylabel("Absolute Error")
ax4.set_title("Prediction Error Analysis")
ax4.legend()
ax4.grid(alpha=0.3)

plt.savefig("pinn_results.png", dpi=150, bbox_inches="tight")
# plt.show() -- Commented out; use Agg backend for headless systems
print("[OK] Results saved as pinn_results.png")

# ── Final Error Statistics ──────────────────────────
print("\n[*] Final Error Statistics:")
print(f"  Membrane Potential V: MAE = {V_err.mean():.4f} mV, Max = {V_err.max():.4f} mV")
print(f"  Synaptic Conductance g: MAE = {g_err.mean():.4f} nS, Max = {g_err.max():.4f} nS")
