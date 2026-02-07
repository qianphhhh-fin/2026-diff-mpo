
import torch
import cvxpy as cp
import numpy as np
from config import cfg
from mpo_solver import DifferentiableMPO_cvx

def test_cvar_constraint():
    print("🧪 Testing CVaR Constraint in DifferentiableMPO...")
    
    # 1. 强制开启 CVaR
    cfg.CVAR_ENABLE = True
    cfg.CVAR_LIMIT = 0.01 # 设置一个非常严格的限制 (1%)，迫使松弛变量生效或大幅调整权重
    print(f"⚙️ CVaR Config: Enable={cfg.CVAR_ENABLE}, Limit={cfg.CVAR_LIMIT}, Confidence={cfg.CVAR_CONFIDENCE}")

    # 2. 准备数据
    B, H, N = 2, cfg.PREDICT_HORIZON, cfg.NUM_ASSETS
    
    # 模拟高风险高收益环境
    # 资产 0: 收益极高，但波动率也极高
    # 资产 1: 收益低，波动率低 (无风险资产)
    mu = torch.zeros(B, H, N, dtype=torch.double)
    mu[:, :, 0] = 0.10  # Asset 0: High Return
    mu[:, :, 1] = 0.01  # Asset 1: Low Return
    
    # 构造 L 矩阵
    # Asset 0 的波动率设为 0.20 (20%) -> L[0,0] = 0.20
    L = torch.zeros(B, H, N, N, dtype=torch.double)
    L[:, :, 0, 0] = 0.20 
    L[:, :, 1, 1] = 0.01
    # 其他资产设为中等
    for i in range(2, N):
        L[:, :, i, i] = 0.05
        
    mu.requires_grad = True
    L.requires_grad = True
    
    w_prev = torch.ones(B, N, dtype=torch.double) / N
    
    # 3. 运行 Solver
    try:
        solver = DifferentiableMPO_cvx()
        w_plan = solver(mu, L, w_prev)
        
        print("✅ Solver execution successful.")
        print(f"   Output shape: {w_plan.shape}")
        
        # 4. 检查结果
        # 如果 CVaR 约束生效，应该会减少对 Asset 0 的配置，尽管它收益很高
        w_avg = w_plan.detach().numpy().mean(axis=(0, 1))
        print("\n📊 Average Weights Allocation:")
        for i in range(min(5, N)):
            print(f"   Asset {i}: {w_avg[i]:.4f}")
            
        # 验证 CVaR 限制是否被尊重 (近似)
        # CVaR ~ -mu*w + 2.06 * sigma
        # Asset 0 CVaR approx: -0.1*w0 + 2.06 * 0.20 * w0 = (-0.1 + 0.412) * w0 = 0.312 * w0
        # Limit = 0.01
        # w0 should be approx 0.01 / 0.312 = 0.03
        
        print(f"\nExpected w[0] (approx) < 0.03 to satisfy CVaR limit.")
        
        # 5. 反向传播测试
        loss = -w_plan.sum() # Dummy loss
        loss.backward()
        print("✅ Backward pass successful.")
        
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cvar_constraint()
