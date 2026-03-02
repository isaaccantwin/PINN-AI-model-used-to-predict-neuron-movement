"""
雙神經元互惠耦合 PINN — 診斷腳本
測試純 RK4 基準線（無 NN 修正）的積分誤差
"""
import sys, os
os.environ["PYTHONUTF8"] = "1"
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from scipy.integrate import solve_ivp

# 物理參數
tau_m=20.0; E_rest=-70.0; C_m=100.0
E_exc=0.0; E_inh=-80.0
tau_s=5.0; g_max=20.0; V_thresh=-55.0; beta=0.5
I_ext1=100.0; I_ext2=10.0
DT=0.1; T_END=200.0; N_STEPS=int(T_END/DT)

def sigmoid(x): return 1.0 / (1.0 + np.exp(-float(x)))
def syn_drive(V): return sigmoid(beta * (V - V_thresh))

def ode_rhs(V1, g12, V2, g21):
    dV1  = -(V1-E_rest)/tau_m + I_ext1/C_m  - g21/C_m*(V1-E_inh)
    dg12 = -g12/tau_s + (g_max/tau_s)*syn_drive(V1)
    dV2  = -(V2-E_rest)/tau_m + I_ext2/C_m  + g12/C_m*(E_exc-V2)
    dg21 = -g21/tau_s + (g_max/tau_s)*syn_drive(V2)
    return dV1, dg12, dV2, dg21

# scipy 高精度參考軌跡 (RK45, rtol=1e-9)
def dual_ode(t, y):
    V1, g12, V2, g21 = y
    return list(ode_rhs(V1, g12, V2, g21))

t_eval = np.linspace(0, T_END, N_STEPS+1)
V1_0=-70.0; g12_0=0.0; V2_0=-70.0; g21_0=0.0
sol = solve_ivp(dual_ode, [0,T_END], [V1_0,g12_0,V2_0,g21_0],
                t_eval=t_eval, method="RK45", rtol=1e-9, atol=1e-11)
V1_ref=sol.y[0]; g12_ref=sol.y[1]; V2_ref=sol.y[2]; g21_ref=sol.y[3]

# 純 RK4 逐步積分（無 NN，dt=0.1ms）
V1, g12, V2, g21 = V1_0, g12_0, V2_0, g21_0
h = DT; h2 = h/2.0
V1_rk4=np.zeros(N_STEPS+1); V1_rk4[0]=V1
g12_rk4=np.zeros(N_STEPS+1); g12_rk4[0]=g12
V2_rk4=np.zeros(N_STEPS+1); V2_rk4[0]=V2
g21_rk4=np.zeros(N_STEPS+1); g21_rk4[0]=g21

for i in range(N_STEPS):
    k1 = ode_rhs(V1, g12, V2, g21)
    k2 = ode_rhs(V1+h2*k1[0], g12+h2*k1[1], V2+h2*k1[2], g21+h2*k1[3])
    k3 = ode_rhs(V1+h2*k2[0], g12+h2*k2[1], V2+h2*k2[2], g21+h2*k2[3])
    k4 = ode_rhs(V1+h*k3[0],  g12+h*k3[1],  V2+h*k3[2],  g21+h*k3[3])
    c = h/6.0
    V1  += c*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
    g12 += c*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
    V2  += c*(k1[2]+2*k2[2]+2*k3[2]+k4[2])
    g21 += c*(k1[3]+2*k2[3]+2*k3[3]+k4[3])
    g12 = max(g12, 0); g21 = max(g21, 0)
    V1_rk4[i+1]=V1; g12_rk4[i+1]=g12
    V2_rk4[i+1]=V2; g21_rk4[i+1]=g21

e_V1  = np.abs(V1_rk4  - V1_ref)
e_V2  = np.abs(V2_rk4  - V2_ref)
print("=" * 55)
print("  純 RK4 積分誤差（無 NN，dt=0.1 ms）")
print("=" * 55)
print(f"  V1 MAE = {e_V1.mean():.6f} mV  Max = {e_V1.max():.6f} mV")
print(f"  V2 MAE = {e_V2.mean():.6f} mV  Max = {e_V2.max():.6f} mV")
print(f"  V1 MAE < 0.05 mV: {'[通過]' if e_V1.mean() < 0.05 else '[未通過]'}")
print(f"  V2 MAE < 0.05 mV: {'[通過]' if e_V2.mean() < 0.05 else '[未通過]'}")
