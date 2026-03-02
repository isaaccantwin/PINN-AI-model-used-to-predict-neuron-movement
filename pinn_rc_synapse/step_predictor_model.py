"""
Step-by-Step PINN Predictor
============================
An autoregressive Physics-Informed Neural Network that predicts one discrete
time-step at a time.

Architecture
------------
  Input  : [V_current, g_current, I_input]  (3 features)
  Output : [V_next,    g_next   ]            (2 targets  )

The PINN Loss ensures each predicted step respects the RC-circuit ODE:

  dV/dt ≈ (V_next - V_current) / dt  (Euler finite-difference approximation)
  dg/dt ≈ (g_next - g_current) / dt

Physics constraints embedded in loss:

  R_V = (V_next - V_current)/dt  +  (V_current - E_rest)/tau_m
                                  -  (g_current / C_m)*(E_syn - V_current)
        → 0

  R_g = (g_next - g_current)/dt  +  g_current / tau_s
        → 0

The input current I_input drives V through:
  dV/dt = -(V - E_rest)/tau_m + (g/C_m)*(E_syn - V) + I_input/C_m

Usage
-----
  model = StepPredictorPINN(...)
  V_next, g_next = model(V_now, g_now, I_now)  # shapes: (B,1) each
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional


# ─────────────────────────────────────────────
#  Physics Parameters
# ─────────────────────────────────────────────

@dataclass
class StepPhysicsParams:
    """
    RC-circuit + synaptic conductance physical parameters.

    Units
    -----
    tau_m  : ms (membrane time constant)
    E_rest : mV (resting potential)
    E_syn  : mV (synaptic reversal potential; 0 mV = excitatory)
    C_m    : pF (membrane capacitance)
    tau_s  : ms (synaptic conductance decay time constant)
    dt     : ms (simulation time step)
    """
    tau_m:  float = 20.0
    E_rest: float = -70.0
    E_syn:  float = 0.0
    C_m:    float = 100.0
    tau_s:  float = 5.0
    dt:     float = 0.1   # time step size (ms)

    # Normalisation ranges — used to scale inputs/outputs to [-1, 1]
    V_min:  float = -80.0
    V_max:  float = -50.0
    g_min:  float = 0.0
    g_max:  float = 15.0
    I_min:  float = -1.0
    I_max:  float = 1.0   # nA

    def V_norm(self, V: torch.Tensor) -> torch.Tensor:
        return 2.0 * (V - self.V_min) / (self.V_max - self.V_min) - 1.0

    def g_norm(self, g: torch.Tensor) -> torch.Tensor:
        return 2.0 * (g - self.g_min) / (self.g_max - self.g_min) - 1.0

    def I_norm(self, I: torch.Tensor) -> torch.Tensor:
        return 2.0 * (I - self.I_min) / (self.I_max - self.I_min) - 1.0

    def V_denorm(self, V_n: torch.Tensor) -> torch.Tensor:
        return (V_n + 1.0) / 2.0 * (self.V_max - self.V_min) + self.V_min

    def g_denorm(self, g_n: torch.Tensor) -> torch.Tensor:
        return (g_n + 1.0) / 2.0 * (self.g_max - self.g_min) + self.g_min


# ─────────────────────────────────────────────
#  Neural Network
# ─────────────────────────────────────────────

class StepPredictorPINN(nn.Module):
    """
    Autoregressive one-step predictor — Physics-first + NN correction.

    Architecture
    ------------
    Step 1 — Euler physics prior:
        V_euler = V_c + dt * [-(V_c-E_rest)/tau_m + (g_c/C_m)*(E_syn-V_c) + I/C_m]
        g_euler = g_c + dt * [-g_c/tau_s]

    Step 2 — MLP correction (small residual on top of Euler):
        Input:  [V_norm, g_norm, I_norm]
        Output: [ΔV,     Δg    ]

    Step 3 — Final prediction:
        V_next = V_euler + ΔV
        g_next = max(g_euler + Δg, 0)

    This formulation is far more stable over long rollouts because the network
    only needs to learn small physics-consistent corrections rather than the
    entire dynamics from scratch.
    """

    def __init__(
        self,
        params: StepPhysicsParams,
        hidden_layers: int = 4,
        hidden_dim: int = 64,
        activation: str = "swish",
        delta_V_scale: float = 0.5,   # mV — max correction amplitude for V
        delta_g_scale: float = 0.2,   # nS — max correction amplitude for g
    ):
        super().__init__()
        self.params = params
        self.delta_V_scale = delta_V_scale
        self.delta_g_scale = delta_g_scale

        act_map = {
            "tanh":  nn.Tanh(),
            "swish": nn.SiLU(),
            "gelu":  nn.GELU(),
            "relu":  nn.ReLU(),
        }
        act_fn = act_map[activation]

        # MLP: 3 → hidden → 2  (predicts small corrections ΔV, Δg)
        layers = [nn.Linear(3, hidden_dim), act_fn]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act_fn]
        layers.append(nn.Linear(hidden_dim, 2))

        self.net = nn.Sequential(*layers)

        # Initialise the final layer to near-zero → NN starts as pure Euler
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # Other layers: Xavier
        for m in self.net[:-1]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def _euler_step(
        self,
        V_c: torch.Tensor,
        g_c: torch.Tensor,
        I:   torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute one Euler step from the known physics ODEs."""
        p = self.params
        dV = (
            -(V_c - p.E_rest) / p.tau_m
            + (g_c / p.C_m) * (p.E_syn - V_c)
            + I / p.C_m
        )
        dg = -g_c / p.tau_s
        V_euler = V_c + p.dt * dV
        g_euler = g_c + p.dt * dg
        return V_euler, g_euler

    def forward(
        self,
        V_c: torch.Tensor,    # (B, 1) mV
        g_c: torch.Tensor,    # (B, 1) nS
        I:   torch.Tensor,    # (B, 1) nA
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        V_next : (B, 1) mV
        g_next : (B, 1) nS  (always ≥ 0)
        """
        p = self.params

        # ── Step 1: Physics Euler prior ──────────────────────────
        V_euler, g_euler = self._euler_step(V_c, g_c, I)

        # ── Step 2: MLP correction ───────────────────────────────
        V_n = p.V_norm(V_c)
        g_n = p.g_norm(g_c)
        I_n = p.I_norm(I)
        x   = torch.cat([V_n, g_n, I_n], dim=-1)   # (B, 3)
        raw = self.net(x)                            # (B, 2)

        # tanh constrains corrections to [-1, 1] * scale
        delta_V = self.delta_V_scale * torch.tanh(raw[:, 0:1])
        delta_g = self.delta_g_scale * torch.tanh(raw[:, 1:2])

        # ── Step 3: Final prediction ──────────────────────────────
        V_next = V_euler + delta_V
        g_next = torch.clamp(g_euler + delta_g, min=0.0)   # g ≥ 0 physically

        return V_next, g_next


# ─────────────────────────────────────────────
#  PINN Loss (Finite-difference ODE residuals)
# ─────────────────────────────────────────────

class StepPredictorLoss(nn.Module):
    """
    Composite loss for the step predictor.

    L_total = λ_data * L_data  +  λ_V * L_physics_V  +  λ_g * L_physics_g

    Physics residuals use Euler finite-difference approximation of ODEs:

      R_V = (V_next - V_current)/dt
              + (V_current - E_rest)/tau_m
              - (g_current/C_m)*(E_syn - V_current)
              - I_input/C_m
            → 0

      R_g = (g_next - g_current)/dt  +  g_current / tau_s  → 0
    """

    def __init__(
        self,
        params: StepPhysicsParams,
        lambda_data: float = 1.0,
        lambda_V:    float = 0.5,
        lambda_g:    float = 0.5,
    ):
        super().__init__()
        self.params = params
        self.lambda_data = lambda_data
        self.lambda_V    = lambda_V
        self.lambda_g    = lambda_g

    def forward(
        self,
        V_c:    torch.Tensor,   # (B, 1) current V (mV)
        g_c:    torch.Tensor,   # (B, 1) current g (nS)
        I:      torch.Tensor,   # (B, 1) external current input (nA)
        V_next: torch.Tensor,   # (B, 1) predicted next V (mV)
        g_next: torch.Tensor,   # (B, 1) predicted next g (nS)
        V_tgt:  Optional[torch.Tensor] = None,  # (B, 1) target next V
        g_tgt:  Optional[torch.Tensor] = None,  # (B, 1) target next g
    ) -> Tuple[torch.Tensor, dict]:
        p = self.params
        dt = p.dt

        # ── 1. Physics Residual Losses ──────────────────────────────
        # RC circuit (membrane voltage)
        dV_dt_approx = (V_next - V_c) / dt
        dV_dt_physics = (
            -(V_c - p.E_rest) / p.tau_m
            + (g_c / p.C_m) * (p.E_syn - V_c)
            + I / p.C_m
        )
        R_V = dV_dt_approx - dV_dt_physics
        L_V = torch.mean(R_V ** 2)

        # Synaptic conductance decay
        dg_dt_approx  = (g_next - g_c) / dt
        dg_dt_physics = -g_c / p.tau_s
        R_g = dg_dt_approx - dg_dt_physics
        L_g = torch.mean(R_g ** 2)

        total_loss = self.lambda_V * L_V + self.lambda_g * L_g
        loss_dict = {
            "L_physics_V": L_V.item(),
            "L_physics_g": L_g.item(),
        }

        # ── 2. Data Loss ────────────────────────────────────────────
        L_data = torch.tensor(0.0, device=V_c.device)
        if V_tgt is not None:
            L_data = L_data + torch.mean((V_next - V_tgt) ** 2)
        if g_tgt is not None:
            L_data = L_data + torch.mean((g_next - g_tgt) ** 2)
        total_loss = total_loss + self.lambda_data * L_data
        loss_dict["L_data"] = L_data.item()

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict


# ─────────────────────────────────────────────
#  Trainer
# ─────────────────────────────────────────────

class StepPredictorTrainer:
    """Trains the StepPredictorPINN from paired (current, next) samples."""

    def __init__(
        self,
        model:    StepPredictorPINN,
        loss_fn:  StepPredictorLoss,
        lr:       float = 1e-3,
        device:   str   = "cpu",
    ):
        self.model   = model.to(device)
        self.loss_fn = loss_fn
        self.device  = device
        self.history = []

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=5000, eta_min=1e-5
        )

    def train(
        self,
        epochs:    int,
        V_c_data:  np.ndarray,   # (N,) current V
        g_c_data:  np.ndarray,   # (N,) current g
        I_data:    np.ndarray,   # (N,) input current
        V_n_data:  np.ndarray,   # (N,) next V
        g_n_data:  np.ndarray,   # (N,) next g
        batch_size: int = 256,
        log_every:  int = 200,
    ):
        N = len(V_c_data)

        def to_t(arr):
            return torch.tensor(arr.reshape(-1, 1).astype(np.float32), device=self.device)

        V_c_t = to_t(V_c_data)
        g_c_t = to_t(g_c_data)
        I_t   = to_t(I_data)
        V_n_t = to_t(V_n_data)
        g_n_t = to_t(g_n_data)

        print(f"{'訓練輪次':>10} | {'總損失':>12} | {'RC殘差(V)':>12} | {'g衰減殘差':>12} | {'資料損失':>10}")
        print("-" * 66)

        for epoch in range(1, epochs + 1):
            # Mini-batch sampling
            idx = torch.randperm(N, device=self.device)[:batch_size]
            V_c_b = V_c_t[idx]; g_c_b = g_c_t[idx]
            I_b   = I_t[idx]
            V_n_b = V_n_t[idx]; g_n_b = g_n_t[idx]

            self.model.train()
            self.optimizer.zero_grad()

            V_next_pred, g_next_pred = self.model(V_c_b, g_c_b, I_b)
            total_loss, loss_dict = self.loss_fn(
                V_c_b, g_c_b, I_b, V_next_pred, g_next_pred,
                V_tgt=V_n_b, g_tgt=g_n_b
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            self.history.append(loss_dict)

            if epoch % log_every == 0 or epoch == 1:
                print(
                    f"{epoch:>8d} | "
                    f"{loss_dict['total']:>10.6f} | "
                    f"{loss_dict['L_physics_V']:>10.6f} | "
                    f"{loss_dict['L_physics_g']:>10.6f} | "
                    f"{loss_dict['L_data']:>10.6f}"
                )

        print("\n[完成] 訓練結束！")
        return self.history

    @torch.no_grad()
    def rollout(
        self,
        V0: float,
        g0: float,
        I_seq: np.ndarray,   # (T,) input current array for T steps
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Auto-regressive 1000-step prediction.

        Returns
        -------
        V_pred : (T+1,) membrane potential trajectory
        g_pred : (T+1,) conductance trajectory
        """
        self.model.eval()
        device = self.device

        V_traj = [V0]
        g_traj = [g0]

        V_cur = torch.tensor([[V0]], dtype=torch.float32, device=device)
        g_cur = torch.tensor([[g0]], dtype=torch.float32, device=device)

        for I_val in I_seq:
            I_t = torch.tensor([[I_val]], dtype=torch.float32, device=device)
            V_next, g_next = self.model(V_cur, g_cur, I_t)
            V_cur = V_next
            g_cur = g_next
            V_traj.append(V_next.item())
            g_traj.append(g_next.item())

        return np.array(V_traj), np.array(g_traj)
