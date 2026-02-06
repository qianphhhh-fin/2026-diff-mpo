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
            
            # 4. CVaR Penalty: 100 * Softplus(-mu_p + kappa*sigma_p - limit)
            # Using Softplus for better Hessian properties than ReLU
            mu_p = (mu * w).sum(dim=-1) # (B, H)
            sigma_p = torch.norm(L_T_w.squeeze(-1), p=2, dim=-1) # (B, H)
            
            limit_val = cvar_limit if cvar_limit.dim() > 0 else cvar_limit.unsqueeze(0)
            # Broadcast limit_val to (B, H) if necessary
            if limit_val.dim() == 1:
                limit_val = limit_val.unsqueeze(1)
            
            violation = -mu_p + kappa * sigma_p - limit_val
            # Softplus approximation of ReLU for smoothness
            loss_cvar = 100.0 * torch.nn.functional.softplus(violation, beta=50).sum()
            
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
            'kappa': norm.pdf(norm.ppf(cfg.CVAR_CONFIDENCE)) / (1 - cfg.CVAR_CONFIDENCE)
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
                sigma_p = torch.norm(L_T_w.squeeze(-1), p=2, dim=-1)
                limit_val = cvar_limit if cvar_limit.dim() > 0 else cvar_limit.unsqueeze(0)
                if limit_val.dim() == 1: limit_val = limit_val.unsqueeze(1)
                violation = -mu_p + kappa * sigma_p - limit_val
                loss_cvar = 100.0 * torch.nn.functional.softplus(violation, beta=50).sum()
                
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
# 单元测试 (Unit Test)
# ==========================
if __name__ == "__main__":
    print("🧪 开始测试 mpo_solver 模块 (Fast MDFP Implementation)...")
    
    # 1. 模拟 Batch 数据
    B, H, N = 2, cfg.PREDICT_HORIZON, cfg.NUM_ASSETS
    device = cfg.DEVICE
    
    # 模拟预测的 Mu (需要梯度)
    mu = torch.randn(B, H, N, requires_grad=True, dtype=torch.float32, device=device)
    
    # 模拟预测的 L (需要梯度) - 初始化为单位阵附近
    L = torch.eye(N, device=device).view(1, 1, N, N).repeat(B, H, 1, 1)
    L.requires_grad = True
    
    # 初始权重 (不需要梯度)
    w0 = torch.ones(B, N, dtype=torch.float32, device=device) / N
    
    # 2. 实例化 Solver
    solver = DifferentiableMPO().to(device)
    print("✅ Solver 初始化成功")
    
    # 3. 前向传播
    start_t = torch.cuda.Event(enable_timing=True) if device=='cuda' else None
    end_t = torch.cuda.Event(enable_timing=True) if device=='cuda' else None
    
    if start_t: start_t.record()
    w_plan = solver(mu, L, w0)
    if end_t: end_t.record(); torch.cuda.synchronize()
    
    print(f"✅ 前向传播成功. Output Shape: {w_plan.shape} (Expected: {B, H, N})")
    
    # 4. 反向传播测试
    # 构造一个假的 Loss: 希望 w 的第一个资产权重越大越好
    loss = -w_plan[:, :, 0].sum()
    loss.backward()
    
    print("✅ 反向传播成功")
    print(f"   Gradient of mu exists: {mu.grad is not None}")
    print(f"   Gradient of L exists: {L.grad is not None}")
    if mu.grad is not None:
        print(f"   mu grad sample norm: {mu.grad.norm().item()}")
    
    print("\n🚀 mpo_solver 模块升级完成！(FastDiffMPO Integrated)")
