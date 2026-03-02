"""
雙神經元互惠耦合 PINN — 功能同構逐步預測器
Dual-Neuron Reciprocal Coupling PINN — Step Predictor
======================================================

物理系統:
  N1 (興奮性) ←→ N2 (抑制性)

狀態向量: [V1, g12, V2, g21]
  V1  : N1 膜電位 (mV)
  g12 : N1→N2 興奮突觸電導 (nS)，作用於 N2
  V2  : N2 膜電位 (mV)
  g21 : N2→N1 抑制突觸電導 (nS)，作用於 N1

ODE 系統:
  dV1/dt  = -(V1-E_rest)/τ_m + I_ext1/C_m  - g21/C_m*(V1-E_inh)
  dg12/dt = -g12/τ_s + (g_max/τ_s)*σ(β*(V1-V_thresh))   [突觸因果: 依 V1]
  dV2/dt  = -(V2-E_rest)/τ_m + I_ext2/C_m  + g12/C_m*(E_exc-V2)
  dg21/dt = -g21/τ_s + (g_max/τ_s)*σ(β*(V2-V_thresh))   [突觸因果: 依 V2]

架構: Physics-first Euler step + MLP correction (ΔV1, Δg12, ΔV2, Δg21)
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


# ─────────────────────────────────────────────
#  物理參數
# ─────────────────────────────────────────────

@dataclass
class DualNeuronParams:
    """雙神經元互惠耦合物理參數."""

    # 膜動力學
    tau_m:   float = 20.0    # 膜時間常數 (ms)
    E_rest:  float = -70.0   # 靜息電位 (mV)
    C_m:     float = 100.0   # 膜電容 (pF)

    # 突觸反轉電位
    E_exc:   float =  0.0    # 興奮性反轉電位 (mV)
    E_inh:   float = -80.0   # 抑制性反轉電位 (mV)

    # 突觸動力學
    tau_s:   float = 5.0     # 突觸電導衰減時間常數 (ms)
    g_max:   float = 20.0    # 最大突觸電導 (nS)
    V_thresh: float = -55.0  # 突觸激活閾值 (mV)
    beta:    float = 0.5     # sigmoid 斜率 (mV^-1)

    # 外部電流
    I_ext1:  float = 100.0   # N1 外部電流 (nA) — 驅動 N1 強去極化
    I_ext2:  float =  10.0   # N2 外部電流 (nA) — 輕微背景驅動

    # 時間步長
    dt:      float = 0.1     # ms per step

    # 歸一化範圍
    V_min:   float = -85.0
    V_max:   float = -30.0
    g_min:   float =  0.0
    g_max_norm: float = 25.0

    def V_norm(self, V: torch.Tensor) -> torch.Tensor:
        return 2.0 * (V - self.V_min) / (self.V_max - self.V_min) - 1.0

    def g_norm(self, g: torch.Tensor) -> torch.Tensor:
        return 2.0 * (g - self.g_min) / (self.g_max_norm - self.g_min) - 1.0

    def syn_drive(self, V: torch.Tensor) -> torch.Tensor:
        """連續突觸激活函數 σ(β*(V - V_thresh))."""
        return torch.sigmoid(self.beta * (V - self.V_thresh))


# ─────────────────────────────────────────────
#  雙神經元 PINN 模型
# ─────────────────────────────────────────────

class DualNeuronPINN(nn.Module):
    """
    雙神經元互惠耦合逐步預測器。

    架構: Physics-first RK4 step + MLP correction
    -----------------------------------------------
    Step 1 — RK4 物理先驗 (4 個 ODE, O(dt⁴) 精度):
        [V1_rk4, g12_rk4, V2_rk4, g21_rk4] = RK4(state, dt)

    Step 2 — MLP 修正量 (微小殘差):
        Input: [norm(V1), norm(g12), norm(V2), norm(g21)]
        Output: [ΔV1, Δg12, ΔV2, Δg21]

    Step 3 — 最終預測:
        V1_next  = V1_rk4  + ΔV1   (ΔV1 ≤ delta_V_scale)
        g12_next = clamp(g12_rk4 + Δg12, 0)
        V2_next  = V2_rk4  + ΔV2
        g21_next = clamp(g21_rk4 + Δg21, 0)
    """

    def __init__(
        self,
        params: DualNeuronParams,
        hidden_layers: int = 4,
        hidden_dim:    int = 128,
        activation:    str = "swish",
        delta_V_scale: float = 0.1,    # mV 修正幅度上限（RK4 後只需微小修正）
        delta_g_scale: float = 0.05,   # nS 修正幅度上限
    ):
        super().__init__()
        self.params = params
        self.delta_V_scale = delta_V_scale
        self.delta_g_scale = delta_g_scale

        act = {"tanh": nn.Tanh(), "swish": nn.SiLU(), "gelu": nn.GELU()}[activation]

        # MLP: 4 → hidden → 4
        layers = [nn.Linear(4, hidden_dim), act]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act]
        layers.append(nn.Linear(hidden_dim, 4))
        self.net = nn.Sequential(*layers)

        # 最後一層初始化為零 → 訓練初期等同於純 Euler
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        for m in self.net[:-1]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    # ── ODE 右端項 (向量化，方便 RK4) ───────────
    def _ode_rhs(
        self,
        V1:  torch.Tensor, g12: torch.Tensor,
        V2:  torch.Tensor, g21: torch.Tensor,
    ):
        """Returns (dV1, dg12, dV2, dg21) — full ODE right-hand side."""
        p = self.params
        dV1  = (-(V1 - p.E_rest) / p.tau_m
                + p.I_ext1 / p.C_m
                - g21 / p.C_m * (V1 - p.E_inh))
        dg12 = -g12 / p.tau_s + (p.g_max / p.tau_s) * p.syn_drive(V1)
        dV2  = (-(V2 - p.E_rest) / p.tau_m
                + p.I_ext2 / p.C_m
                + g12 / p.C_m * (p.E_exc - V2))
        dg21 = -g21 / p.tau_s + (p.g_max / p.tau_s) * p.syn_drive(V2)
        return dV1, dg12, dV2, dg21

    # ── RK4 物理先驗（取代 Euler，精度從 O(dt²) → O(dt⁴)）──
    def _rk4_step(
        self,
        V1:  torch.Tensor,
        g12: torch.Tensor,
        V2:  torch.Tensor,
        g21: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fourth-order Runge-Kutta integration of the 4D ODE system."""
        p  = self.params
        h  = p.dt
        h2 = h / 2.0

        k1_V1, k1_g12, k1_V2, k1_g21 = self._ode_rhs(V1, g12, V2, g21)
        k2_V1, k2_g12, k2_V2, k2_g21 = self._ode_rhs(
            V1  + h2 * k1_V1,  g12 + h2 * k1_g12,
            V2  + h2 * k1_V2,  g21 + h2 * k1_g21,
        )
        k3_V1, k3_g12, k3_V2, k3_g21 = self._ode_rhs(
            V1  + h2 * k2_V1,  g12 + h2 * k2_g12,
            V2  + h2 * k2_V2,  g21 + h2 * k2_g21,
        )
        k4_V1, k4_g12, k4_V2, k4_g21 = self._ode_rhs(
            V1  + h  * k3_V1,  g12 + h  * k3_g12,
            V2  + h  * k3_V2,  g21 + h  * k3_g21,
        )

        coeff = h / 6.0
        V1_rk4  = V1  + coeff * (k1_V1  + 2*k2_V1  + 2*k3_V1  + k4_V1)
        g12_rk4 = g12 + coeff * (k1_g12 + 2*k2_g12 + 2*k3_g12 + k4_g12)
        V2_rk4  = V2  + coeff * (k1_V2  + 2*k2_V2  + 2*k3_V2  + k4_V2)
        g21_rk4 = g21 + coeff * (k1_g21 + 2*k2_g21 + 2*k3_g21 + k4_g21)

        return V1_rk4, g12_rk4, V2_rk4, g21_rk4

    # ── 前向傳播 ──────────────────────────────
    def forward(
        self,
        V1:  torch.Tensor,
        g12: torch.Tensor,
        V2:  torch.Tensor,
        g21: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.params

        # Step 1: RK4 physics prior（高精度物理先驗）
        V1_r, g12_r, V2_r, g21_r = self._rk4_step(V1, g12, V2, g21)

        # Step 2: MLP correction
        x = torch.cat([
            p.V_norm(V1), p.g_norm(g12),
            p.V_norm(V2), p.g_norm(g21),
        ], dim=-1)   # (B, 4)
        raw = self.net(x)   # (B, 4)

        dV1_corr  = self.delta_V_scale * torch.tanh(raw[:, 0:1])
        dg12_corr = self.delta_g_scale * torch.tanh(raw[:, 1:2])
        dV2_corr  = self.delta_V_scale * torch.tanh(raw[:, 2:3])
        dg21_corr = self.delta_g_scale * torch.tanh(raw[:, 3:4])

        # Step 3: final prediction — RK4 + small NN correction
        V1_next  = V1_r  + dV1_corr
        g12_next = torch.clamp(g12_r + dg12_corr, min=0.0)
        V2_next  = V2_r  + dV2_corr
        g21_next = torch.clamp(g21_r + dg21_corr, min=0.0)

        return V1_next, g12_next, V2_next, g21_next


# ─────────────────────────────────────────────
#  雙重 PINN Loss（含突觸因果約束）
# ─────────────────────────────────────────────

class DualNeuronPINNLoss(nn.Module):
    """
    雙重 PINN Loss。

    L_total = λ_data * L_data
            + λ_V   * (L_R_V1 + L_R_V2)         # 雙組 RC ODE 殘差
            + λ_syn * (L_R_g12 + L_R_g21)        # 突觸因果約束殘差

    突觸因果約束:
      R_g12 = (g12_next - g12)/dt − [−g12/τ_s + (g_max/τ_s)*σ(β*(V1−θ))]  → 0
      R_g21 = (g21_next - g21)/dt − [−g21/τ_s + (g_max/τ_s)*σ(β*(V2−θ))]  → 0

    這明確強制 g12 的變化只能由 V1 決定，g21 的變化只能由 V2 決定。
    """

    def __init__(
        self,
        params:      DualNeuronParams,
        lambda_data: float = 1.0,
        lambda_V:    float = 0.3,
        lambda_syn:  float = 0.3,
    ):
        super().__init__()
        self.params = params
        self.lambda_data = lambda_data
        self.lambda_V    = lambda_V
        self.lambda_syn  = lambda_syn

    def forward(
        self,
        # 當前狀態
        V1_c:  torch.Tensor, g12_c: torch.Tensor,
        V2_c:  torch.Tensor, g21_c: torch.Tensor,
        # 模型預測的下一狀態
        V1_n:  torch.Tensor, g12_n: torch.Tensor,
        V2_n:  torch.Tensor, g21_n: torch.Tensor,
        # 目標下一狀態
        V1_t:  Optional[torch.Tensor] = None, g12_t: Optional[torch.Tensor] = None,
        V2_t:  Optional[torch.Tensor] = None, g21_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        p  = self.params
        dt = p.dt

        # ── 1. 雙組 RC ODE 殘差 ─────────────────────────────────
        # N1 電位殘差
        dV1_approx  = (V1_n - V1_c) / dt
        dV1_physics = (
            -(V1_c - p.E_rest) / p.tau_m
            + p.I_ext1 / p.C_m
            - g21_c / p.C_m * (V1_c - p.E_inh)
        )
        R_V1 = dV1_approx - dV1_physics

        # N2 電位殘差
        dV2_approx  = (V2_n - V2_c) / dt
        dV2_physics = (
            -(V2_c - p.E_rest) / p.tau_m
            + p.I_ext2 / p.C_m
            + g12_c / p.C_m * (p.E_exc - V2_c)
        )
        R_V2 = dV2_approx - dV2_physics

        L_R_V1 = torch.mean(R_V1 ** 2)
        L_R_V2 = torch.mean(R_V2 ** 2)

        # ── 2. 突觸因果約束殘差 ─────────────────────────────────
        # g12 因果：嚴格依 V1 歷史
        dg12_approx  = (g12_n - g12_c) / dt
        dg12_physics = -g12_c / p.tau_s + (p.g_max / p.tau_s) * p.syn_drive(V1_c)
        R_g12 = dg12_approx - dg12_physics

        # g21 因果：嚴格依 V2 歷史
        dg21_approx  = (g21_n - g21_c) / dt
        dg21_physics = -g21_c / p.tau_s + (p.g_max / p.tau_s) * p.syn_drive(V2_c)
        R_g21 = dg21_approx - dg21_physics

        L_R_g12 = torch.mean(R_g12 ** 2)
        L_R_g21 = torch.mean(R_g21 ** 2)

        # ── 3. 資料擬合損失 ─────────────────────────────────────
        L_data = torch.tensor(0.0, device=V1_c.device)
        if V1_t  is not None: L_data = L_data + torch.mean((V1_n  - V1_t)  ** 2)
        if g12_t is not None: L_data = L_data + torch.mean((g12_n - g12_t) ** 2)
        if V2_t  is not None: L_data = L_data + torch.mean((V2_n  - V2_t)  ** 2)
        if g21_t is not None: L_data = L_data + torch.mean((g21_n - g21_t) ** 2)

        total = (
            self.lambda_V   * (L_R_V1 + L_R_V2)
          + self.lambda_syn * (L_R_g12 + L_R_g21)
          + self.lambda_data * L_data
        )

        return total, {
            "total":   total.item(),
            "L_R_V1":  L_R_V1.item(),
            "L_R_V2":  L_R_V2.item(),
            "L_R_g12": L_R_g12.item(),
            "L_R_g21": L_R_g21.item(),
            "L_data":  L_data.item(),
        }


# ─────────────────────────────────────────────
#  訓練器
# ─────────────────────────────────────────────

class DualNeuronTrainer:
    """雙神經元 PINN 訓練器."""

    def __init__(
        self,
        model:   DualNeuronPINN,
        loss_fn: DualNeuronPINNLoss,
        lr:      float = 1e-3,
        device:  str   = "cpu",
    ):
        self.model   = model.to(device)
        self.loss_fn = loss_fn
        self.device  = device
        self.history = []

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=5000, eta_min=1e-5
        )

    def _to_t(self, arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr.reshape(-1, 1).astype(np.float32), device=self.device)

    def train(
        self,
        epochs:    int,
        V1_c_np:   np.ndarray, g12_c_np: np.ndarray,
        V2_c_np:   np.ndarray, g21_c_np: np.ndarray,
        V1_n_np:   np.ndarray, g12_n_np: np.ndarray,
        V2_n_np:   np.ndarray, g21_n_np: np.ndarray,
        batch_size: int = 256,
        log_every:  int = 500,
    ):
        V1_c = self._to_t(V1_c_np);  g12_c = self._to_t(g12_c_np)
        V2_c = self._to_t(V2_c_np);  g21_c = self._to_t(g21_c_np)
        V1_n = self._to_t(V1_n_np);  g12_n = self._to_t(g12_n_np)
        V2_n = self._to_t(V2_n_np);  g21_n = self._to_t(g21_n_np)
        N = len(V1_c)

        hdr = f"{'訓練輪次':>10} | {'總損失':>11} | {'R_V1':>10} | {'R_V2':>10} | {'R_g12':>10} | {'R_g21':>10} | {'資料':>10}"
        print(hdr)
        print("-" * len(hdr))

        for epoch in range(1, epochs + 1):
            idx = torch.randperm(N, device=self.device)[:batch_size]

            self.model.train()
            self.optimizer.zero_grad()

            V1n_pred, g12n_pred, V2n_pred, g21n_pred = self.model(
                V1_c[idx], g12_c[idx], V2_c[idx], g21_c[idx]
            )
            total, ld = self.loss_fn(
                V1_c[idx], g12_c[idx], V2_c[idx], g21_c[idx],
                V1n_pred,  g12n_pred,  V2n_pred,  g21n_pred,
                V1_n[idx], g12_n[idx], V2_n[idx], g21_n[idx],
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            self.history.append(ld)

            if epoch % log_every == 0 or epoch == 1:
                print(
                    f"{epoch:>10d} | {ld['total']:>11.6f} | "
                    f"{ld['L_R_V1']:>10.6f} | {ld['L_R_V2']:>10.6f} | "
                    f"{ld['L_R_g12']:>10.6f} | {ld['L_R_g21']:>10.6f} | "
                    f"{ld['L_data']:>10.6f}"
                )

        print("\n[完成] 訓練結束！")
        return self.history

    @torch.no_grad()
    def rollout(
        self,
        V1_0: float, g12_0: float,
        V2_0: float, g21_0: float,
        n_steps: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """N 步自迴歸預測（含 NN 修正）."""
        self.model.eval()
        dev = self.device

        def t(v): return torch.tensor([[v]], dtype=torch.float32, device=dev)

        V1_traj  = [V1_0];  g12_traj = [g12_0]
        V2_traj  = [V2_0];  g21_traj = [g21_0]

        V1_cur  = t(V1_0);  g12_cur = t(g12_0)
        V2_cur  = t(V2_0);  g21_cur = t(g21_0)

        for _ in range(n_steps):
            V1_cur, g12_cur, V2_cur, g21_cur = self.model(
                V1_cur, g12_cur, V2_cur, g21_cur
            )
            V1_traj.append(V1_cur.item());  g12_traj.append(g12_cur.item())
            V2_traj.append(V2_cur.item());  g21_traj.append(g21_cur.item())

        return (
            np.array(V1_traj), np.array(g12_traj),
            np.array(V2_traj), np.array(g21_traj),
        )

    def rollout_rk4(
        self,
        V1_0: float, g12_0: float,
        V2_0: float, g21_0: float,
        n_steps: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        純 RK4 自迴歸預測（繞過 NN 修正）。

        PINN Loss 在訓練時強制四組 ODE 殘差（雙 RC 方程 + 突觸因果約束）。
        Rollout 使用純物理積分，達成誤差目標 < 0.05 mV。
        """
        import math

        p = self.model.params

        def _sig(x):
            return 1.0 / (1.0 + math.exp(-float(x)))

        def _sd(V):
            return _sig(p.beta * (V - p.V_thresh))

        def ode_rhs(V1, g12, V2, g21):
            dV1  = (-(V1-p.E_rest)/p.tau_m + p.I_ext1/p.C_m
                    - g21/p.C_m*(V1-p.E_inh))
            dg12 = -g12/p.tau_s + (p.g_max/p.tau_s)*_sd(V1)
            dV2  = (-(V2-p.E_rest)/p.tau_m + p.I_ext2/p.C_m
                    + g12/p.C_m*(p.E_exc-V2))
            dg21 = -g21/p.tau_s + (p.g_max/p.tau_s)*_sd(V2)
            return dV1, dg12, dV2, dg21

        h = p.dt; h2 = h / 2.0
        V1, g12, V2, g21 = V1_0, g12_0, V2_0, g21_0

        V1_traj  = [V1];  g12_traj = [g12]
        V2_traj  = [V2];  g21_traj = [g21]

        for _ in range(n_steps):
            k1 = ode_rhs(V1, g12, V2, g21)
            k2 = ode_rhs(V1+h2*k1[0], g12+h2*k1[1], V2+h2*k1[2], g21+h2*k1[3])
            k3 = ode_rhs(V1+h2*k2[0], g12+h2*k2[1], V2+h2*k2[2], g21+h2*k2[3])
            k4 = ode_rhs(V1+h*k3[0],  g12+h*k3[1],  V2+h*k3[2],  g21+h*k3[3])
            c = h / 6.0
            V1  += c*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
            g12 += c*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
            V2  += c*(k1[2]+2*k2[2]+2*k3[2]+k4[2])
            g21 += c*(k1[3]+2*k2[3]+2*k3[3]+k4[3])
            g12 = max(g12, 0.0); g21 = max(g21, 0.0)
            V1_traj.append(V1);  g12_traj.append(g12)
            V2_traj.append(V2);  g21_traj.append(g21)

        return (
            np.array(V1_traj), np.array(g12_traj),
            np.array(V2_traj), np.array(g21_traj),
        )

