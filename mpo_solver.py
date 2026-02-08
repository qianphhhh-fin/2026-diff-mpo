"""
脚本名称: mpo_solver.py
功能描述: 
    实现可微多周期优化 (Differentiable MPO) 的核心求解器。
    通过自定义 PyTorch Autograd Function，实现了前向求解 (Forward) 和反向传播 (Backward)。

主要组件:
    1. DifferentiableMPO (nn.Module): 
       - 封装了求解器的接口。
       - solve_forward_md: 实现基于镜像下降 (Mirror Descent) 的快速前向求解器，支持 GPU 加速。
    2. MDFPIdentity (autograd.Function):
       - forward: 调用 solve_forward_md 计算最优权重 w*。
       - backward: 使用隐函数定理 (Implicit Function Theorem) 和 Neumann 级数近似，
         高效计算 Loss 对输入参数 (mu, L, w_prev) 的梯度。

输入:
    - mu: 预测收益率。
    - L: 预测协方差的 Cholesky 因子。
    - w_prev: 初始持仓。
    - cvar_limit: 风险约束上限。

输出:
    - w_star: 最优投资组合权重，带有梯度信息。

与其他脚本的关系:
    - 被 model.py 调用，作为神经网络的最后一层 (Optimization Layer)。
    - 依赖 config.py 获取优化参数 (Gamma, Cost Coeff, CVaR Penalty)。
"""

import torch
import torch.nn as nn
from scipy.stats import norm
from config import cfg

class MDFPIdentity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu, L, w_prev, cvar_limit, w_star, H, N, cfg_dict):
        ctx.save_for_backward(mu, L, w_prev, cvar_limit, w_star)
        ctx.cfg = cfg_dict
        ctx.H = H
        ctx.N = N
        return w_star

    @staticmethod
    def backward(ctx, grad_output):
        # 2. Backward Pass: MDFP
        # Vectorized implementation
        with torch.enable_grad():
            mu, L, w_prev, cvar_limit, w_star = ctx.saved_tensors
            cfg_dict = ctx.cfg
            H, N = ctx.H, ctx.N
            
            eta = 1e-1 
            B_iter = 5     # Reduced iterations for speed, usually sufficient
            
            gamma = cfg_dict['gamma']
            cost_coeff = cfg_dict['cost_coeff']
            kappa = cfg_dict['kappa']
            cvar_penalty = cfg_dict.get('cvar_penalty', 50.0) # 使用 Config 配置
            
            # --- A. Define Objective Gradient \nabla_w F(w) ---
            w = w_star.detach().clone().requires_grad_(True)
            
            # Vectorized Objective Calculation
            # 1. Return: - mu^T w
            loss_ret = - (mu * w).sum()
            
            # 2. Risk: || L^T w ||^2
            # L: (B, H, N, N), w: (B, H, N) -> (B, H, N, 1)
            # matmul: (B, H, N, N).mT @ (B, H, N, 1) -> (B, H, N, 1)
            L_T_w = torch.matmul(L.transpose(-1, -2), w.unsqueeze(-1))
            loss_risk = (L_T_w.squeeze(-1) ** 2).sum()
            
            # 3. Cost: smoothed L1 of (w_t - w_{t-1})
            # Prepend w_prev to w along time dimension
            # w_prev: (B, N) -> (B, 1, N)
            w_shifted = torch.cat([w_prev.unsqueeze(1), w[:, :-1, :]], dim=1)
            diff = w - w_shifted
            loss_cost = torch.sum(torch.sqrt(diff**2 + 1e-8))
            
            # 4. CVaR Penalty: cvar_penalty * Softplus(-mu_p + kappa*sigma_p - limit)
            if cvar_penalty > 1e-6:
                mu_p = (mu * w).sum(dim=-1) # (B, H)
                sigma_p = torch.norm(L_T_w.squeeze(-1), p=2, dim=-1) # (B, H)
                
                limit_val = cvar_limit if cvar_limit.dim() > 0 else cvar_limit.unsqueeze(0)
                # Broadcast limit_val to (B, H) if necessary
                if limit_val.dim() == 1:
                    limit_val = limit_val.unsqueeze(1)
                
                violation = -mu_p + kappa * sigma_p - limit_val
                # Softplus approximation of ReLU for smoothness
                loss_cvar = cvar_penalty * torch.nn.functional.softplus(violation, beta=50).sum()
            else:
                loss_cvar = 0.0
            
            F = loss_ret + gamma * loss_risk + cost_coeff * loss_cost + loss_cvar
            
            if not F.requires_grad:
                return (None,) * 8
            
            grad_F, = torch.autograd.grad(F, w, create_graph=True)
            
            # --- B. Neumann Series for (I - J)^-1 ---
            # J^T v = (v - <z*, v>1) - eta * HVP(z* * (v - <z*, v>1))
            
            curr_v = grad_output.clone()
            sum_v = grad_output.clone()
            
            # Vectorized Neumann Loop
            for k in range(B_iter):
                # 1. Projection: v_temp = v - <v, w*> 1
                # w_star: (B, H, N), curr_v: (B, H, N)
                dot_val = (curr_v * w_star).sum(dim=-1, keepdim=True) # (B, H, 1)
                v_temp = curr_v - dot_val # Broadcasting subtract
                
                # 2. Hessian Vector Product
                vec = v_temp * w_star
                
                # Efficient HVP using autograd
                grad_F_dot_vec = torch.sum(grad_F * vec)
                # retain_graph=True is needed because we differentiate grad_F multiple times
                H_vec, = torch.autograd.grad(grad_F_dot_vec, w, retain_graph=True)
                
                next_v = v_temp - eta * H_vec
                
                curr_v = next_v
                sum_v = sum_v + curr_v
            
            # --- C. Parameter Gradients ---
            # u_P = sum_v - <sum_v, w*> 1
            dot_val_sum = (sum_v * w_star).sum(dim=-1, keepdim=True)
            u_P = sum_v - dot_val_sum
            
            u_hat = (u_P * w_star).detach()
            
            grad_F_dot_uhat = torch.sum(grad_F * u_hat)
            
            # Only compute gradients for inputs that require grad
            inputs_to_grad = []
            inputs_indices = []
            
            if mu.requires_grad: inputs_to_grad.append(mu); inputs_indices.append(0)
            if L.requires_grad: inputs_to_grad.append(L); inputs_indices.append(1)
            if w_prev.requires_grad: inputs_to_grad.append(w_prev); inputs_indices.append(2)
            if cvar_limit.requires_grad: inputs_to_grad.append(cvar_limit); inputs_indices.append(3)
            
            if len(inputs_to_grad) > 0:
                computed_grads = torch.autograd.grad(grad_F_dot_uhat, tuple(inputs_to_grad), retain_graph=False, allow_unused=True)
            else:
                computed_grads = []

            # Map back to full list
            d_mu, d_L, d_wprev, d_cvar = None, None, None, None
            
            # Helper to get grad from computed list
            curr_idx = 0
            if 0 in inputs_indices: d_mu = computed_grads[curr_idx]; curr_idx += 1
            if 1 in inputs_indices: d_L = computed_grads[curr_idx]; curr_idx += 1
            if 2 in inputs_indices: d_wprev = computed_grads[curr_idx]; curr_idx += 1
            if 3 in inputs_indices: d_cvar = computed_grads[curr_idx]; curr_idx += 1
            
            if d_mu is None: d_mu = torch.zeros_like(mu)
            if d_L is None: d_L = torch.zeros_like(L)
            if d_wprev is None: d_wprev = torch.zeros_like(w_prev)
            if d_cvar is None: d_cvar = torch.zeros_like(cvar_limit)
            
            return -eta * d_mu, -eta * d_L, -eta * d_wprev, -eta * d_cvar, None, None, None, None

class DifferentiableMPO(nn.Module):
    def __init__(self):
        super(DifferentiableMPO, self).__init__()
        self.H = cfg.PREDICT_HORIZON
        self.N = cfg.NUM_ASSETS
        self.cfg_dict = {
            'gamma': cfg.RISK_AVERSION,
            'cost_coeff': cfg.COST_COEFF,
            'kappa': norm.pdf(norm.ppf(cfg.CVAR_CONFIDENCE)) / (1 - cfg.CVAR_CONFIDENCE),
            'cvar_penalty': getattr(cfg, 'CVAR_PENALTY', 50.0) # [NEW]
        }
            
    def solve_forward_md(self, mu, L, w_prev, cvar_limit, max_iters=300, tol=1e-6):
        """
        Solve the forward problem using Mirror Descent (Entropic) on PyTorch.
        This avoids CvxpyLayer overhead and ensures the solution matches the backward pass objective.
        """
        B, H, N = mu.shape
        # Initialize w uniform
        w = torch.ones_like(mu) / N
        w.requires_grad_(False) # We don't track grad in forward solve
        
        eta = 0.05 # Tuned step size
        
        gamma = self.cfg_dict['gamma']
        cost_coeff = self.cfg_dict['cost_coeff']
        kappa = self.cfg_dict['kappa']
        cvar_penalty = self.cfg_dict.get('cvar_penalty', 50.0)
        
        for k in range(max_iters):
            # Compute Gradient of F w.r.t w
            # We can use autograd for convenience, but detach to avoid graph build up
            with torch.enable_grad():
                w_var = w.detach().requires_grad_(True)
                
                # Re-implement objective (same as backward)
                loss_ret = - (mu * w_var).sum()
                L_T_w = torch.matmul(L.transpose(-1, -2), w_var.unsqueeze(-1))
                loss_risk = (L_T_w.squeeze(-1) ** 2).sum()
                w_shifted = torch.cat([w_prev.unsqueeze(1), w_var[:, :-1, :]], dim=1)
                diff = w_var - w_shifted
                loss_cost = torch.sum(torch.sqrt(diff**2 + 1e-8))
                
                mu_p = (mu * w_var).sum(dim=-1)
                
                if cvar_penalty > 1e-6:
                    sigma_p = torch.norm(L_T_w.squeeze(-1), p=2, dim=-1)
                    limit_val = cvar_limit if cvar_limit.dim() > 0 else cvar_limit.unsqueeze(0)
                    if limit_val.dim() == 1: limit_val = limit_val.unsqueeze(1)
                    violation = -mu_p + kappa * sigma_p - limit_val
                    loss_cvar = cvar_penalty * torch.nn.functional.softplus(violation, beta=50).sum()
                else:
                    loss_cvar = 0.0
                
                F = loss_ret + gamma * loss_risk + cost_coeff * loss_cost + loss_cvar
                
                grad_F, = torch.autograd.grad(F, w_var)
            
            # Mirror Descent Step: w_{k+1} = w_k * exp(-eta * grad) / Norm
            # Log-space update for stability
            log_w = torch.log(w + 1e-10)
            log_w_new = log_w - eta * grad_F
            w_new = torch.softmax(log_w_new, dim=-1)
            
            # Check convergence
            dist = torch.norm(w_new - w)
            w = w_new
            if dist < tol:
                break
                
        return w

    def forward(self, mu, L, w_prev, cvar_limit=None):
        # 如果未提供 limit，使用 Config 默认值
        if cvar_limit is None:
            # 构造一个 scalar tensor
            cvar_limit = torch.tensor(cfg.CVAR_LIMIT, device=mu.device, dtype=mu.dtype)
        
        # 确保 cvar_limit 是 tensor 且维度正确
        if cvar_limit.dim() == 0:
            cvar_limit = cvar_limit.expand(mu.size(0)) # Expand to Batch Size
            
        # 1. Forward using custom MD solver (Fast)
        w_star_batch = self.solve_forward_md(mu, L, w_prev, cvar_limit)
            
        # 2. Attach MDFP Backward
        return MDFPIdentity.apply(mu, L, w_prev, cvar_limit, w_star_batch, self.H, self.N, self.cfg_dict)

# ==========================
# CvxpyLayer 实现 (Benchmark)
# ==========================
class DifferentiableMPO_cvx(nn.Module):
    def __init__(self):
        super(DifferentiableMPO_cvx, self).__init__()
        try:
            import cvxpy as cp
            from cvxpylayers.torch import CvxpyLayer
        except ImportError:
            raise ImportError("请先安装 cvxpy 和 cvxpylayers: pip install cvxpy cvxpylayers")

        self.H = cfg.PREDICT_HORIZON
        self.N = cfg.NUM_ASSETS
        self.cfg_dict = {
            'gamma': cfg.RISK_AVERSION,
            'cost_coeff': cfg.COST_COEFF,
            'kappa': norm.pdf(norm.ppf(cfg.CVAR_CONFIDENCE)) / (1 - cfg.CVAR_CONFIDENCE),
            'cvar_penalty': getattr(cfg, 'CVAR_PENALTY', 50.0)
        }
        
        # 1. 定义参数 (Parameters)
        self.mu_param = cp.Parameter((self.H, self.N))
        self.L_param = cp.Parameter((self.H, self.N, self.N))
        self.w_prev_param = cp.Parameter(self.N)
        self.cvar_limit_param = cp.Parameter() 
        
        # 2. 定义变量 (Variables)
        self.w_var = cp.Variable((self.H, self.N))
        
        # 3. 构建目标函数 (Loss_QP: Min Variance + L2 Cost)
        # 不再包含收益率项 (-mu^T w)
        # 神经网络只需要预测 Sigma (L)，通过调整风险结构来间接优化 Sortino
        obj = 0
        
        # (1) Risk: sum(||L_t^T w_t||^2)
        # 这现在是主要的驱动项
        risk_term = 0
        for t in range(self.H):
            # L_t: (N, N), w_t: (N,)
            risk_term += cp.sum_squares(self.L_param[t].T @ self.w_var[t])
        obj += risk_term
        
        # (2) Cost: L2 Penalty (Smooth)
        # 使用 L2 Norm 替代 L1，保证 QP 性质
        cost_term = 0
        # t=0
        cost_term += cp.sum_squares(self.w_var[0] - self.w_prev_param)
        # t=1..H-1
        for t in range(1, self.H):
            cost_term += cp.sum_squares(self.w_var[t] - self.w_var[t-1])
            
        # Cost Coeff 需要根据 L2 的量级重新调整，这里暂时保持 Config 读取
        # 但通常 L2 cost 需要更大的系数才能与 Risk 平衡
        obj += self.cfg_dict['cost_coeff'] * 10.0 * cost_term
        
        # (3) CVaR Penalty (REMOVED)
        # 移除了 CVaR 项，保持 Solver 为纯 QP
            
        # 4. 约束条件
        constraints = [
            cp.sum(self.w_var, axis=1) == 1,
            self.w_var >= 0
        ]
        
        # 5. 初始化 Layer
        # 注意：不再传入 mu_param 和 cvar_limit_param
        problem = cp.Problem(cp.Minimize(obj), constraints)
        self.layer = CvxpyLayer(
            problem, 
            parameters=[self.L_param, self.w_prev_param], 
            variables=[self.w_var]
        )
        
    def forward(self, mu, L, w_prev, cvar_limit=None):
        # 兼容接口：虽然不再使用 mu 和 cvar_limit，但保持函数签名一致
        # mu: (Batch, H, N) -> IGNORED
        
        # 调用 CvxpyLayer
        w_star, = self.layer(L, w_prev)
        return w_star

# ==========================
# 单元测试 (Unit Test)
# ==========================
if __name__ == "__main__":
    import time
    import numpy as np
    
    # 设置打印精度
    torch.set_printoptions(precision=4, sci_mode=False)
    
    print("🧪 开始对比测试: Mirror Descent (MD) vs CvxpyLayer (CVX)...")
    
    # 1. 模拟 Batch 数据
    # 使用较小的 Batch 以便 CVX 跑得动 (CVX Batch 性能较差)
    B, H, N = 4, cfg.PREDICT_HORIZON, cfg.NUM_ASSETS
    device = cfg.DEVICE
    print(f"   Batch={B}, Horizon={H}, Assets={N}, Device={device}")
    
    # 模拟输入 (需要梯度)
    mu = torch.randn(B, H, N, requires_grad=True, dtype=torch.float32, device=device)
    L = torch.eye(N, device=device).view(1, 1, N, N).repeat(B, H, 1, 1)
    # 增加一点随机性给 L
    L = L + 0.1 * torch.randn_like(L)
    L.requires_grad = True
    
    w0 = torch.ones(B, N, dtype=torch.float32, device=device) / N
    w0.requires_grad = True # 也可以测试对 w_prev 的梯度
    
    # 2. 实例化 Solvers
    print("\n📦 初始化 Solvers...")
    solver_md = DifferentiableMPO().to(device)
    
    try:
        solver_cvx = DifferentiableMPO_cvx().to(device)
        has_cvx = True
        print("   ✅ DifferentiableMPO_cvx 加载成功")
    except Exception as e:
        print(f"   ⚠️ DifferentiableMPO_cvx 加载失败: {e}")
        has_cvx = False
        
    if has_cvx:
        # ==========================
        # 3. 前向传播速度对比
        # ==========================
        print("\n🏎️  Forward Pass Speed Test (Avg of 10 runs)")
        
        # MD Warmup
        _ = solver_md(mu, L, w0)
        
        # MD Timing
        torch.cuda.synchronize() if device=='cuda' else None
        t0 = time.time()
        for _ in range(10):
            w_md = solver_md(mu, L, w0)
        torch.cuda.synchronize() if device=='cuda' else None
        t_md = (time.time() - t0) / 10
        print(f"   🔹 Mirror Descent (Ours): {t_md*1000:.2f} ms")
        
        # CVX Warmup
        # CVX 第一次运行通常很慢 (Canonicalization)，Warmup 很重要
        _ = solver_cvx(mu, L, w0)
        
        # CVX Timing
        torch.cuda.synchronize() if device=='cuda' else None
        t0 = time.time()
        for _ in range(10):
            w_cvx = solver_cvx(mu, L, w0)
        torch.cuda.synchronize() if device=='cuda' else None
        t_cvx = (time.time() - t0) / 10
        print(f"   🔸 CvxpyLayer (Ref)   : {t_cvx*1000:.2f} ms")
        print(f"   🚀 Speedup: {t_cvx / t_md:.1f}x")
        
        # ==========================
        # 4. 结果一致性对比
        # ==========================
        print("\n🔍 Result Consistency Check")
        # 比较 w_md 和 w_cvx
        diff = torch.norm(w_md - w_cvx) / (torch.norm(w_cvx) + 1e-8)
        print(f"   Rel. Norm Diff: {diff.item():.6f}")
        if diff < 1e-2:
            print("   ✅ Results match closely.")
        else:
            print("   ⚠️ Results might differ (check constraints/parameters).")
            
        # ==========================
        # 5. 反向传播速度与梯度对比
        # ==========================
        print("\n📉 Backward Pass & Gradient Check")
        
        # 构造 Loss
        target = torch.rand_like(w_md)
        target = target / target.sum(dim=-1, keepdim=True)
        
        # --- MD Backward ---
        loss_md = torch.sum((w_md - target)**2)
        
        # 清零梯度
        if mu.grad is not None: mu.grad.zero_()
        if L.grad is not None: L.grad.zero_()
        
        torch.cuda.synchronize() if device=='cuda' else None
        t0 = time.time()
        loss_md.backward(retain_graph=True)
        torch.cuda.synchronize() if device=='cuda' else None
        t_md_back = time.time() - t0
        print(f"   🔹 MD Backward Time : {t_md_back*1000:.2f} ms")
        
        grad_mu_md = mu.grad.clone()
        mu.grad.zero_() # Reset for CVX
        
        # --- CVX Backward ---
        # 必须重新计算 Graph，因为 w_cvx 和 w_md 是不同的计算图节点
        # 为了公平，我们这里直接对 w_cvx backward
        loss_cvx = torch.sum((w_cvx - target)**2)
        
        torch.cuda.synchronize() if device=='cuda' else None
        t0 = time.time()
        loss_cvx.backward()
        torch.cuda.synchronize() if device=='cuda' else None
        t_cvx_back = time.time() - t0
        print(f"   🔸 CVX Backward Time: {t_cvx_back*1000:.2f} ms")
        
        grad_mu_cvx = mu.grad.clone()
        
        # --- Gradient Comparison ---
        grad_diff = torch.norm(grad_mu_md - grad_mu_cvx) / (torch.norm(grad_mu_cvx) + 1e-6)
        print(f"   Gradient Rel. Diff (Mu): {grad_diff.item():.6f}")
        
        # Cosine Similarity
        cos_sim = torch.nn.functional.cosine_similarity(grad_mu_md.flatten(), grad_mu_cvx.flatten(), dim=0)
        print(f"   Gradient Cosine Sim    : {cos_sim.item():.4f}")
        
    print("\nDone.")
