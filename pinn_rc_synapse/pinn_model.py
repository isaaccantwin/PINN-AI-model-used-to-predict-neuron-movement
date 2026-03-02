"""
Physics-Informed Neural Network (PINNs) for Neuron RC Circuit + Synaptic Conductance
======================================================================================

物理約束 (Physical Constraints):
----------------------------------
1. RC 電路膜電位動力學 (RC Circuit Membrane Voltage Dynamics):
   τ_m * dV/dt = -(V - E_rest) + g(t) * (E_syn - V) * R_m
   => dV/dt = -(V - E_rest)/τ_m + (g(t)/C_m) * (E_syn - V)

2. 突觸電導一階衰減動力學 (Synaptic Conductance First-Order Decay):
   dg/dt = -g / τ_s + Σ δ(t - t_spike)
   (在無突觸輸入時) => dg/dt + g/τ_s = 0

這些方程式被強制寫入 Loss Function 的 Residual 部分。
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


# ─────────────────────────────────────────────
#  Neural Network Architecture
# ─────────────────────────────────────────────

class NeuronPINN(nn.Module):
    """
    Physics-Informed Neural Network for RC Circuit + Synaptic Conductance.
    Outputs V(t) membrane potential and g(t) synaptic conductance.

    Network: t_norm -> [hidden layers] -> [V_raw, g_raw]
    Outputs are rescaled to physical units using known prior ranges.
    """

    def __init__(
        self,
        hidden_layers: int = 5,
        hidden_dim: int = 64,
        activation: str = "swish",
        # Input/output normalisation priors (set from known physics)
        T_scale: float = 100.0,    # time normalisation (ms)
        V_offset: float = -70.0,   # resting potential offset (mV)
        V_scale: float = 30.0,     # expected V excursion range (mV)
        g_scale: float = 10.0,     # expected peak g (nS)
    ):
        super().__init__()

        self.T_scale  = T_scale
        self.V_offset = V_offset
        self.V_scale  = V_scale
        self.g_scale  = g_scale

        act_fn = {
            "tanh": nn.Tanh(),
            "swish": nn.SiLU(),
            "gelu": nn.GELU(),
        }[activation]

        layers = [nn.Linear(1, hidden_dim), act_fn]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act_fn]
        layers += [nn.Linear(hidden_dim, 2)]  # raw outputs in [-1, +1] range

        self.net = nn.Sequential(*layers)
        self._softplus = nn.Softplus(beta=5)  # smooth, always-positive for g

        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            t: time tensor, shape (N, 1), physical units (ms).
               requires_grad=True needed at collocation points.
        Returns:
            V: membrane potential (mV), shape (N, 1)
            g: synaptic conductance (nS) >= 0, shape (N, 1)
        """
        t_norm = t / self.T_scale           # normalise to [0, 1]
        out = self.net(t_norm)              # (N, 2)

        # V: offset + scaled raw output  =>  lives near E_rest by default
        V = self.V_offset + self.V_scale * out[:, 0:1]

        # g: Softplus keeps output positive with smooth gradients everywhere
        #    (ReLU would zero-out gradients, causing dead neurons)
        g = self.g_scale * self._softplus(out[:, 1:2])

        return V, g


# ─────────────────────────────────────────────
#  Physics Parameters
# ─────────────────────────────────────────────

class PhysicsParams:
    """
    神經元 RC 電路物理參數。
    
    RC 電路等效參數:
      - τ_m = R_m * C_m  膜時間常數 (ms)
      - E_rest           靜息電位 (mV)
      - E_syn            突觸反轉電位 (mV)
      - C_m              膜電容 (pF，用於電導換算)
      - τ_s              突觸電導衰減時間常數 (ms)
    """
    def __init__(
        self,
        tau_m: float = 20.0,    # 膜時間常數 (ms)
        E_rest: float = -70.0,  # 靜息電位 (mV)
        E_syn: float = 0.0,     # 突觸反轉電位 (mV), 興奮性突觸=0mV
        C_m: float = 100.0,     # 膜電容 (pF)
        tau_s: float = 5.0,     # 突觸電導衰減時間常數 (ms)
    ):
        self.tau_m = tau_m
        self.E_rest = E_rest
        self.E_syn = E_syn
        self.C_m = C_m
        self.tau_s = tau_s


# ─────────────────────────────────────────────
#  Physics Residuals (Loss Residuals)
# ─────────────────────────────────────────────

def residual_RC_circuit(
    t: torch.Tensor,
    V: torch.Tensor,
    g: torch.Tensor,
    params: PhysicsParams,
) -> torch.Tensor:
    """
    RC 電路膜電位動力學殘差 (Residual of RC Circuit ODE).

    物理方程 (Physical Equation):
        τ_m * dV/dt = -(V - E_rest) + (g / C_m) * (E_syn - V) * τ_m

    等價形式 (Equivalent form for residual):
        dV/dt + (V - E_rest)/τ_m - (g/C_m) * (E_syn - V) = 0

    Residual:
        R_V = dV/dt + (V - E_rest)/τ_m - (g/C_m) * (E_syn - V)
        => 理想狀況下 R_V → 0

    Args:
        t:      時間點, shape (N, 1), requires_grad=True
        V:      模型預測膜電位, shape (N, 1)
        g:      模型預測突觸電導, shape (N, 1)
        params: 物理參數

    Returns:
        residual: shape (N, 1), 越接近 0 越符合物理
    """
    # 自動微分計算 dV/dt
    dV_dt = torch.autograd.grad(
        outputs=V,
        inputs=t,
        grad_outputs=torch.ones_like(V),
        create_graph=True,   # 允許高階梯度（訓練時需要）
        retain_graph=True,
    )[0]

    # RC 電路 ODE 殘差
    # 膜洩漏項 (Leak current term)
    leak_term = (V - params.E_rest) / params.tau_m

    # 突觸電流項 (Synaptic current term)
    # I_syn / C_m = (g * (E_syn - V)) / C_m
    syn_term = (g / params.C_m) * (params.E_syn - V)

    residual = dV_dt + leak_term - syn_term  # 應等於 0

    return residual


def residual_synaptic_conductance(
    t: torch.Tensor,
    g: torch.Tensor,
    params: PhysicsParams,
) -> torch.Tensor:
    """
    突觸電導一階衰減動力學殘差 (Residual of Synaptic Conductance Decay ODE).

    物理方程 (Physical Equation):
        dg/dt = -g / τ_s

    Residual:
        R_g = dg/dt + g/τ_s
        => 理想狀況下 R_g → 0

    Args:
        t:      時間點, shape (N, 1), requires_grad=True
        g:      模型預測突觸電導, shape (N, 1)
        params: 物理參數

    Returns:
        residual: shape (N, 1), 越接近 0 越符合物理
    """
    # 自動微分計算 dg/dt
    dg_dt = torch.autograd.grad(
        outputs=g,
        inputs=t,
        grad_outputs=torch.ones_like(g),
        create_graph=True,
        retain_graph=True,
    )[0]

    # 一階衰減 ODE 殘差: dg/dt + g/τ_s = 0
    residual = dg_dt + g / params.tau_s

    return residual


# ─────────────────────────────────────────────
#  Composite Loss Function
# ─────────────────────────────────────────────

class PINNLoss(nn.Module):
    """
    PINNs 複合損失函數。

    Total Loss = λ_data * L_data  +  λ_V * L_physics_V  +  λ_g * L_physics_g  +  λ_ic * L_ic

    各項說明:
      L_data      - 資料擬合損失 (Data fitting loss, MSE with observations)
      L_physics_V - RC 電路 ODE 殘差損失 (RC circuit physics residual)
      L_physics_g - 突觸電導衰減 ODE 殘差損失 (Synaptic conductance decay residual)
      L_ic        - 初始條件損失 (Initial condition loss)
    """

    def __init__(
        self,
        params: PhysicsParams,
        lambda_data: float = 1.0,   # 資料損失權重
        lambda_V: float = 1.0,      # RC電路殘差權重
        lambda_g: float = 1.0,      # 電導衰減殘差權重
        lambda_ic: float = 10.0,    # 初始條件權重（較高，確保初值正確）
    ):
        super().__init__()
        self.params = params
        self.lambda_data = lambda_data
        self.lambda_V = lambda_V
        self.lambda_g = lambda_g
        self.lambda_ic = lambda_ic

    def forward(
        self,
        model: NeuronPINN,
        # --- Collocation Points (用於物理殘差) ---
        t_colloc: torch.Tensor,     # shape (N_c, 1), requires_grad=True
        # --- Observed Data Points (可選) ---
        t_data: Optional[torch.Tensor] = None,   # shape (N_d, 1)
        V_data: Optional[torch.Tensor] = None,   # shape (N_d, 1) 觀測膜電位
        g_data: Optional[torch.Tensor] = None,   # shape (N_d, 1) 觀測電導
        # --- Initial Conditions ---
        t0: Optional[torch.Tensor] = None,       # shape (1, 1) 初始時間點
        V0: Optional[float] = None,              # 初始膜電位 (mV)
        g0: Optional[float] = None,              # 初始突觸電導 (nS)
    ) -> Tuple[torch.Tensor, dict]:
        """
        計算總損失。

        Returns:
            total_loss: 總損失標量
            loss_dict:  各分項損失字典（用於監控訓練）
        """
        loss_dict = {}

        # ── 1. 物理殘差損失 (Physics Residual Losses) ──────────────
        V_c, g_c = model(t_colloc)

        # 1a. RC 電路殘差
        r_V = residual_RC_circuit(t_colloc, V_c, g_c, self.params)
        L_physics_V = torch.mean(r_V ** 2)

        # 1b. 突觸電導衰減殘差
        r_g = residual_synaptic_conductance(t_colloc, g_c, self.params)
        L_physics_g = torch.mean(r_g ** 2)

        loss_dict["L_physics_V"] = L_physics_V.item()
        loss_dict["L_physics_g"] = L_physics_g.item()

        total_loss = (
            self.lambda_V * L_physics_V
            + self.lambda_g * L_physics_g
        )

        # ── 2. 資料擬合損失 (Data Fitting Loss) ─────────────────────
        if t_data is not None:
            V_pred, g_pred = model(t_data)
            L_data = 0.0
            if V_data is not None:
                L_data = L_data + torch.mean((V_pred - V_data) ** 2)
            if g_data is not None:
                L_data = L_data + torch.mean((g_pred - g_data) ** 2)
            loss_dict["L_data"] = L_data.item() if isinstance(L_data, torch.Tensor) else 0.0
            total_loss = total_loss + self.lambda_data * L_data

        # ── 3. 初始條件損失 (Initial Condition Loss) ──────────────
        if t0 is not None:
            V_ic, g_ic = model(t0)
            L_ic = 0.0
            if V0 is not None:
                L_ic = L_ic + (V_ic[0, 0] - V0) ** 2
            if g0 is not None:
                L_ic = L_ic + (g_ic[0, 0] - g0) ** 2
            loss_dict["L_ic"] = L_ic.item() if isinstance(L_ic, torch.Tensor) else 0.0
            total_loss = total_loss + self.lambda_ic * L_ic

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict


# ─────────────────────────────────────────────
#  Trainer
# ─────────────────────────────────────────────

class PINNTrainer:
    """
    PINNs 訓練器 (Trainer).
    使用 Adam + L-BFGS 兩階段優化策略（常見於 PINNs 訓練）。
    """

    def __init__(
        self,
        model: NeuronPINN,
        loss_fn: PINNLoss,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.history = []

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.9995
        )

    def train_step(
        self,
        t_colloc: torch.Tensor,
        t_data: Optional[torch.Tensor] = None,
        V_data: Optional[torch.Tensor] = None,
        g_data: Optional[torch.Tensor] = None,
        t0: Optional[torch.Tensor] = None,
        V0: Optional[float] = None,
        g0: Optional[float] = None,
    ) -> dict:
        self.model.train()
        self.optimizer.zero_grad()

        total_loss, loss_dict = self.loss_fn(
            self.model, t_colloc, t_data, V_data, g_data, t0, V0, g0
        )
        total_loss.backward()

        # 梯度裁剪，提升 PINNs 訓練穩定性
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.scheduler.step()
        return loss_dict

    def train(
        self,
        epochs: int,
        t_colloc: torch.Tensor,
        t_data: Optional[torch.Tensor] = None,
        V_data: Optional[torch.Tensor] = None,
        g_data: Optional[torch.Tensor] = None,
        t0: Optional[torch.Tensor] = None,
        V0: Optional[float] = None,
        g0: Optional[float] = None,
        log_every: int = 200,
    ):
        print(f"{'Epoch':>8} | {'Total':>12} | {'L_RC(V)':>12} | {'L_g_decay':>12} | {'L_data':>10} | {'L_ic':>10}")
        print("-" * 75)

        for epoch in range(1, epochs + 1):
            loss_dict = self.train_step(
                t_colloc, t_data, V_data, g_data, t0, V0, g0
            )
            self.history.append(loss_dict)

            if epoch % log_every == 0 or epoch == 1:
                print(
                    f"{epoch:>8d} | "
                    f"{loss_dict.get('total', 0):>12.6f} | "
                    f"{loss_dict.get('L_physics_V', 0):>12.6f} | "
                    f"{loss_dict.get('L_physics_g', 0):>12.6f} | "
                    f"{loss_dict.get('L_data', 0):>10.6f} | "
                    f"{loss_dict.get('L_ic', 0):>10.6f}"
                )

        print("\n[OK] Training complete!")
        return self.history

    @torch.no_grad()
    def predict(self, t: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """推論模式，返回 numpy arrays"""
        self.model.eval()
        V, g = self.model(t)
        return V.cpu().numpy(), g.cpu().numpy()
