import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from config import cfg

class DifferentiableMPO(nn.Module):
    def __init__(self):
        super(DifferentiableMPO, self).__init__()
        
        # ==========================================
        # 1. 定义符号变量
        # ==========================================
        H = cfg.PREDICT_HORIZON
        N = cfg.NUM_ASSETS
        
        # ⚠️ 修正点 1: 只有 mu, L, w0 是需要反向传播的 Parameter
        self.param_mu = cp.Parameter((H, N), name='mu') 
        self.param_L  = cp.Parameter((H, N, N), name='L') 
        self.param_w0 = cp.Parameter(N, name='w_prev') 
        
        # ⚠️ 修正点 2: gamma 和 cost_coeff 改为普通 Python 变量 (常量)
        # 不要用 cp.Parameter 包装，否则会破坏 DPP 结构
        gamma = cfg.RISK_AVERSION
        cost_coeff = cfg.COST_COEFF
        
        # 决策变量
        w = cp.Variable((H, N), name='w_plan')
        
        # ==========================================
        # 2. 构建目标函数与约束
        # ==========================================
        obj_ret = 0
        obj_risk = 0
        obj_cost = 0
        constraints = []
        
        w_current = self.param_w0
        
        for t in range(H):
            # A. 收益
            obj_ret += self.param_mu[t] @ w[t]
            
            # B. 风险
            obj_risk += cp.sum_squares(self.param_L[t].T @ w[t])
            
            # C. 交易成本
            obj_cost += cp.norm(w[t] - w_current, 1)
            
            # D. 约束
            constraints.append(cp.sum(w[t]) == 1.0)
            constraints.append(w[t] >= 0)
            
            w_current = w[t]
            
        # ⚠️ 修正点 3: 这里是 Float * Convex，符合 DPP
        objective = cp.Maximize(obj_ret - gamma * obj_risk - cost_coeff * obj_cost)
        
        # ==========================================
        # 3. 创建 CvxpyLayer
        # ==========================================
        problem = cp.Problem(objective, constraints)
        
        # 现在这一行应该能通过了
        assert problem.is_dpp(), "问题不符合 DPP 规则！请检查是否用 Parameter 乘以了凸项。"
        
        self.layer = CvxpyLayer(
            problem, 
            parameters=[self.param_mu, self.param_L, self.param_w0], 
            variables=[w]
        )
        
    def forward(self, mu, L, w_prev):
        # solver_args 加上 eps 可以防止数值问题报错
        w_plan, = self.layer(
            mu, L, w_prev, 
            solver_args={
                'solve_method': 'ECOS',
                'abstol': 1e-4, # 放宽一点精度，训练更快
                'reltol': 1e-4
            }
        )
        return w_plan

# ==========================
# 单元测试 (Unit Test)
# ==========================
if __name__ == "__main__":
    print("🧪 开始测试 mpo_solver 模块 (Gradient Check)...")
    
    # 1. 模拟 Batch 数据
    B, H, N = 2, cfg.PREDICT_HORIZON, cfg.NUM_ASSETS
    
    # 模拟预测的 Mu (需要梯度)
    mu = torch.randn(B, H, N, requires_grad=True, dtype=torch.double) # CVXPY 默认喜欢 double
    
    # 模拟预测的 L (需要梯度) - 初始化为单位阵附近
    # L 必须是下三角，这里简化，假设网络输出全矩阵，但逻辑上它是 Factor
    L = torch.eye(N).view(1, 1, N, N).repeat(B, H, 1, 1).double()
    L.requires_grad = True
    
    # 初始权重 (不需要梯度)
    w0 = torch.ones(B, N, dtype=torch.double) / N
    
    # 2. 实例化 Solver
    # try:
    solver = DifferentiableMPO()
    print("✅ Solver 初始化成功 (Problem Compiled)")
    
    # 3. 前向传播
    w_plan = solver(mu, L, w0)
    print(f"✅ 前向传播成功. Output Shape: {w_plan.shape} (Expected: {B, H, N})")
    
    # 4. 反向传播测试
    # 构造一个假的 Loss: 希望 w 的第一个资产权重越大越好
    loss = -w_plan[:, :, 0].sum()
    loss.backward()
    
    print("✅ 反向传播成功")
    print(f"   Gradient of mu exists: {mu.grad is not None}")
    print(f"   Gradient of L exists: {L.grad is not None}")
    print(f"   mu grad sample: {mu.grad[0,0,:]}")
    
    print("\n🚀 mpo_solver 模块通过！核心引擎就绪。")
        
    # except Exception as e:
    #     print(f"\n❌ 测试失败: {e}")
    #     print("提示：如果是 SolverError，可能是数据随机初始化导致无解，或者缺少 ECOS/SCS 求解器。")