
"""
脚本名称: test_loss_magnitude.py
功能描述: 
    独立测试脚本，用于检查 Diff-MPO 目标函数中各项 (Return, Risk, Cost, CVaR) 的数值量级。
    这有助于诊断是否存在某一项 Loss 过大导致优化失效 (如 CVaR 惩罚淹没了收益目标)。

流程:
    1. 加载一小部分真实数据 (Data Chunk)。
    2. 加载模型 (随机初始化或预训练)。
    3. 运行前向传播获取 mu, L, w_plan。
    4. 手动计算各项 Loss 组件并打印其统计信息 (Mean, Std, Max)。
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import norm

from config import cfg
from data_loader import MPODataset
from model import MPO_Network_Factor
from mpo_solver import DifferentiableMPO_cvx

def run_loss_magnitude_check():
    print("🔬 开始目标函数量级检查 (Loss Magnitude Check)...")
    
    # ==========================
    # 1. 准备数据 (Data Preparation)
    # ==========================
    print("   -> Loading data chunk...")
    df_raw = pd.read_csv(cfg.DATA_PATH, index_col=0, parse_dates=True)
    all_features = df_raw.values
    all_returns = df_raw[cfg.ASSETS].values
    
    # 取一小段数据 (比如 2018 年初的 100 天)
    # 确保足够 Lookback
    start_idx = 1000 
    end_idx = 1100
    
    X_chunk = all_features[start_idx-cfg.LOOKBACK_WINDOW : end_idx]
    Y_chunk = all_returns[start_idx-cfg.LOOKBACK_WINDOW : end_idx] # Y 其实只需要对应时间的
    
    # 简单标准化 (模拟真实环境)
    # 注意：这里只用这一小段数据的均值方差，仅为了量级测试，不必太严谨
    X_scaled = (X_chunk - X_chunk.mean(axis=0)) / (X_chunk.std(axis=0) + 1e-6)
    
    # 构造 Dataset
    # 我们只关心能否跑通模型，所以 Y 取全 0 也可以，反正不用来算 Loss，只用模型输出算 MPO Loss
    # 但为了兼容 Dataset 接口，还是传入真实 Y
    ds = MPODataset(X_scaled, Y_chunk, cfg.LOOKBACK_WINDOW, cfg.PREDICT_HORIZON)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    
    # ==========================
    # 2. 模型初始化
    # ==========================
    print("   -> Initializing model...")
    model = MPO_Network_Factor().to(cfg.DEVICE).double()
    model.eval() # 设为评估模式
    
    # ==========================
    # 3. 前向传播与计算
    # ==========================
    # 获取一个 Batch
    x_batch, _ = next(iter(loader))
    x_batch = x_batch.to(cfg.DEVICE).double()
    
    # 初始持仓 w_prev (假设为 1/N)
    batch_size = x_batch.size(0)
    w_prev = torch.ones(batch_size, cfg.NUM_ASSETS, device=cfg.DEVICE, dtype=torch.double) / cfg.NUM_ASSETS
    
    print(f"   -> Running forward pass (Batch Size: {batch_size})...")
    with torch.no_grad():
        # 获取模型输出
        w_plan, mu, L = model(x_batch, w_prev)
        
        # ==========================
        # 4. 手动计算各项 Loss 组件
        # ==========================
        # 提取参数
        gamma = cfg.RISK_AVERSION
        cost_coeff = cfg.COST_COEFF
        kappa = norm.pdf(norm.ppf(cfg.CVAR_CONFIDENCE)) / (1 - cfg.CVAR_CONFIDENCE)
        cvar_penalty = getattr(cfg, 'CVAR_PENALTY', 50.0)
        cvar_limit = torch.tensor(cfg.CVAR_LIMIT, device=cfg.DEVICE, dtype=torch.double)
        
        # --- A. Return Term (-mu^T w) ---
        # mu: (B, H, N), w: (B, H, N)
        term_ret = - (mu * w_plan).sum(dim=2) # (B, H)
        val_ret = term_ret.mean().item()
        
        # --- B. Risk Term (gamma * w^T Sigma w) ---
        # L_T_w = L.T @ w
        L_T_w = torch.matmul(L.transpose(-1, -2), w_plan.unsqueeze(-1))
        risk_raw = (L_T_w.squeeze(-1) ** 2).sum(dim=2) # (B, H)
        term_risk = gamma * risk_raw
        val_risk = term_risk.mean().item()
        
        # --- C. Cost Term ---
        w_shifted = torch.cat([w_prev.unsqueeze(1), w_plan[:, :-1, :]], dim=1)
        diff = w_plan - w_shifted
        cost_raw = torch.sum(torch.sqrt(diff**2 + 1e-8), dim=2) # (B, H) (Approx L1)
        term_cost = cost_coeff * cost_raw
        val_cost = term_cost.mean().item()
        
        # --- D. CVaR Term ---
        mu_p = (mu * w_plan).sum(dim=-1) # (B, H)
        sigma_p = torch.norm(L_T_w.squeeze(-1), p=2, dim=-1) # (B, H)
        
        violation = -mu_p + kappa * sigma_p - cvar_limit
        cvar_raw = torch.nn.functional.softplus(violation, beta=50)
        term_cvar = cvar_penalty * cvar_raw
        val_cvar = term_cvar.mean().item()
        
        # Total
        val_total = val_ret + val_risk + val_cost + val_cvar
        
    # ==========================
    # 5. 打印报告
    # ==========================
    print("\n" + "="*50)
    print("📊 MPO Loss Component Analysis (Average per Step)")
    print("="*50)
    print(f"{'Component':<15} | {'Raw Value':<12} | {'Coeff':<8} | {'Weighted Val':<12} | {'% of Total':<10}")
    print("-" * 65)
    
    # 计算百分比 (使用绝对值，因为 Return 是负的)
    abs_total = abs(val_ret) + abs(val_risk) + abs(val_cost) + abs(val_cvar)
    
    def print_row(name, raw, coeff, weighted):
        pct = abs(weighted) / abs_total * 100
        print(f"{name:<15} | {raw:<12.6f} | {coeff:<8} | {weighted:<12.6f} | {pct:<9.1f}%")
        
    print_row("Return (Max)",  val_ret,          "1.0",     val_ret)
    print_row("Risk (Min)",    val_risk/gamma,   str(gamma), val_risk)
    print_row("Cost (Min)",    val_cost/cost_coeff, str(cost_coeff), val_cost)
    print_row("CVaR (Penalty)", val_cvar/cvar_penalty, str(cvar_penalty), val_cvar)
    
    print("-" * 65)
    print(f"{'Total Loss':<15} | {'-':<12} | {'-':<8} | {val_total:<12.6f} | 100.0%")
    print("="*50)
    
    # 诊断建议
    print("\n💡 诊断:")
    if abs(val_cvar) > abs(val_ret) * 10:
        print("⚠️ CVaR 惩罚过大！它比收益项大了 10 倍以上。模型可能被完全压制。")
    if abs(val_cost) > abs(val_ret):
        print("⚠️ 交易成本过高！成本项超过了预期收益。模型将停止交易。")
    if abs(val_risk) > abs(val_ret) * 5:
        print("⚠️ 风险厌恶过强！风险项主导了优化。")
        
    if abs(val_ret) < 1e-5:
        print("⚠️ 预测收益率 (Mu) 极小，接近于 0。可能需要检查数据标准化或模型输出缩放。")

if __name__ == "__main__":
    run_loss_magnitude_check()
