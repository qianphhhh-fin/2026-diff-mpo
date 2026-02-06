import numpy as np
import pandas as pd
import cvxpy as cp
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
# 在 strategy.py 顶部添加 import
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

from config import cfg
from model import MPO_Network_Factor
# 我们需要复用 Loss 计算函数，避免代码重复
# 假设 train_diff_mpo.py 在同一目录下
from train_diff_mpo import calc_composite_loss 

class BaseStrategy:
    """
    策略基类
    定义了所有策略必须实现的两个核心接口：
    1. on_train_period: 当回测引擎认为需要重训时调用 (用于 Stateful 策略)
    2. get_weights: 每个交易日调用，输出目标仓位
    """
    def __init__(self, name):
        self.name = name
        self.device = cfg.DEVICE

    def on_train_period(self, train_loader: DataLoader):
        """
        [生命周期钩子] 训练/微调模型
        :param train_loader: 包含 (X, Y) 的 DataLoader，数据已做防泄露处理
        """
        pass

    def get_weights(self, history_df=None, feature_tensor=None):
        """
        [每日推理] 计算目标仓位
        :param history_df: (DataFrame) 截止到 t-1 的历史收益率数据 (用于传统策略)
        :param feature_tensor: (Tensor) 截止到 t-1 的标准化特征窗口 (用于深度策略)
        :return: (np.array) 目标权重向量
        """
        raise NotImplementedError

# ==============================================================================
# 1. 规则类策略 (Stateless)
# ==============================================================================
class RuleBasedStrategy(BaseStrategy):
    def __init__(self, name):
        super().__init__(name)

    def get_weights(self, history_df, feature_tensor=None):
        N = cfg.NUM_ASSETS
        
        # --- 1/N ---
        if '1/N' in self.name:
            return np.ones(N) / N
            
        # --- Vanilla Risk Parity (基于历史波动率倒数) ---
        elif 'Risk Parity' in self.name:
            # 使用过去 1 年 (252天) 或 Config 定义的窗口
            lookback = 252
            if len(history_df) < lookback:
                lookback = len(history_df)
            
            # 计算波动率 (Standard Deviation)
            # history_df 是收益率
            window_ret = history_df.iloc[-lookback:].values
            vols = np.std(window_ret, axis=0)
            
            # 防除零
            vols[vols < 1e-6] = 1e-6
            
            # 权重与波动率成反比
            raw_w = 1.0 / vols
            return raw_w / np.sum(raw_w)
            
        # --- Factor Momentum (动量因子) ---
        elif 'Momentum' in self.name:
            # 逻辑：买入过去 60 天累计收益最高的 3 个资产，等权持有
            lookback = 60
            if len(history_df) < lookback:
                return np.ones(N) / N
                
            cum_ret = history_df.iloc[-lookback:].sum()
            
            # 排序：从大到小
            # argsort 返回的是从小到大的索引，所以取最后 3 个
            top_k = 3
            top_indices = cum_ret.argsort()[-top_k:]
            
            w = np.zeros(N)
            w[top_indices] = 1.0 / top_k
            return w
            
        else:
            raise ValueError(f"Unknown RuleStrategy: {self.name}")

# ==============================================================================
# 2. 传统优化策略 (Stateless / Instant Fit)
# ==============================================================================
class OptimizationStrategy(BaseStrategy):
    def __init__(self, name, lookback=60, lambda_reg=1e-4):
        super().__init__(name)
        self.lookback = lookback
        self.lambda_reg = lambda_reg # 协方差正则化系数
        
    def get_weights(self, history_df, feature_tensor=None):
        # 1. 数据准备
        if len(history_df) < self.lookback:
            # 数据不足时降级为 1/N
            return np.ones(cfg.NUM_ASSETS) / cfg.NUM_ASSETS
            
        # 截取窗口
        returns = history_df.iloc[-self.lookback:].values
        
        # 检查 NaN
        if np.isnan(returns).any():
            returns = np.nan_to_num(returns)
            
        # 2. 统计量估计
        mu = np.mean(returns, axis=0)
        Sigma = np.cov(returns.T)
        
        # 正则化 (防止矩阵奇异)
        Sigma += self.lambda_reg * np.eye(len(mu))
        
        # 3. 构建优化问题
        N = cfg.NUM_ASSETS
        w = cp.Variable(N)
        
        # 通用约束
        constraints = [
            cp.sum(w) == 1.0,  # 传统模型通常假设满仓 (你可以改为 <= 1)
            w >= 0,            # 只做多
            w <= 0.30          # [重要] 强制分散约束，与 DeepMPO 保持公平
        ]
        
        objective = None
        
        # --- Mean-Variance ---
        if 'Mean-Variance' in self.name:
            # Max w.T*mu - lambda * w.T*Sigma*w
            gamma = cfg.RISK_AVERSION # 0.5
            ret_term = mu @ w
            risk_term = cp.quad_form(w, Sigma)
            objective = cp.Maximize(ret_term - gamma * risk_term)
            
        # --- Global Min Var ---
        elif 'Global Min Var' in self.name:
            # Min w.T*Sigma*w
            risk_term = cp.quad_form(w, Sigma)
            objective = cp.Minimize(risk_term)
            
        # --- Mean-CVaR (不简化逻辑) ---
        elif 'Mean-CVaR' in self.name:
            # CVaR (Conditional Value at Risk) 95%
            # 需要引入辅助变量 z_t 和 alpha
            # Minimize: alpha + 1/((1-c)*T) * sum(z_t)
            # Subject to: z_t >= 0, z_t >= -w.T * r_t - alpha
            
            alpha = cp.Variable()
            z = cp.Variable(self.lookback)
            c = 0.95
            
            # 投资组合在每一天的损失: Loss_t = - (w.T @ r_t)
            # 注意：returns 是 (T, N)
            port_returns = returns @ w 
            losses = -port_returns
            
            cvar_term = alpha + (1.0 / ((1.0 - c) * self.lookback)) * cp.sum(z)
            
            # 添加 CVaR 特有约束
            constraints.append(z >= 0)
            constraints.append(z >= losses - alpha)
            
            # 目标：最小化 CVaR (也可以是 Max Return - lambda * CVaR)
            # 这里为了纯粹性，我们做 Min CVaR
            objective = cp.Minimize(cvar_term)
            
        else:
            raise ValueError(f"Unknown OptimizationStrategy: {self.name}")
            
        # 4. 求解
        # try:
        prob = cp.Problem(objective, constraints)
        # 尝试 ECOS
        prob.solve(solver=cp.ECOS, abstol=1e-5)
        
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            prob.solve(solver=cp.SCS, eps=1e-4) # Fallback
            
        if w.value is None:
            raise ValueError("Solver failed")
            
        w_res = np.clip(w.value, 0, 1)
        return w_res / w_res.sum()
            
        # except Exception as e:
        #     # 求解失败时的兜底 (Fallback to 1/N)
        #     # print(f"Warning: {self.name} solver failed: {e}")
        #     return np.ones(N) / N



# ==============================================================================
# 4. 层次风险平价策略 (HRP - Hierarchical Risk Parity)
# ==============================================================================
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import numpy as np
import pandas as pd
from config import cfg

class HRPStrategy(BaseStrategy):
    """
    Hierarchical Risk Parity (HRP) - 修复版
    使用 Scipy 标准库进行树构建和叶节点排序，修复了之前的崩溃 Bug。
    逻辑：
    1. Clustering: Single Linkage
    2. Quasi-Diag: 使用 sch.leaves_list 获取排序
    3. Recursive Bisection: 逆方差分配
    """
    def __init__(self, name, lookback=252):
        super().__init__(name)
        self.lookback = lookback

    def get_weights(self, history_df, feature_tensor=None):
        # 1. 数据准备
        # HRP 需要一定的历史长度来计算相关性，如果太短可能会导致相关性矩阵奇异
        if len(history_df) < max(self.lookback, cfg.NUM_ASSETS + 5):
            return np.ones(cfg.NUM_ASSETS) / cfg.NUM_ASSETS
            
        # 截取窗口
        returns = history_df.iloc[-self.lookback:]
        
        # 2. 计算相关性和协方差
        # 填充 NaN 防止报错 (虽然理论上不该有)
        corr = returns.corr().fillna(0)
        cov = returns.cov().fillna(0)
        
        # 3. 核心逻辑
        try:
            # === Stage 1: Tree Clustering ===
            # 距离度量: d = sqrt(0.5 * (1 - rho))
            # 裁剪相关系数防止浮点误差导致 > 1 或 < -1
            corr_clipped = corr.clip(-1.0, 1.0)
            dist = np.sqrt(0.5 * (1 - corr_clipped))
            dist_vec = squareform(dist)
            
            # 使用 Single Linkage (Lopez de Prado 原版推荐)
            link = sch.linkage(dist_vec, 'single')
            
            # === Stage 2: Quasi-Diagonalization (修复点) ===
            # 直接使用 scipy 的 leaves_list 获取排序索引
            # 这比手写的 while 循环稳健得多
            sort_ix_indices = sch.leaves_list(link)
            
            # 将数字索引转换为资产名称列表
            sort_ix = returns.columns[sort_ix_indices].tolist()
            
            # 根据新顺序重排协方差矩阵
            cov_sorted = cov.loc[sort_ix, sort_ix]
            
            # === Stage 3: Recursive Bisection ===
            # 递归二分分配权重
            hrp_weights = self._get_rec_bisection(cov_sorted, sort_ix)
            
            # 4. 结果对齐
            # hrp_weights 是 Series，索引是资产名，需要对齐回原始 columns 顺序
            final_w = hrp_weights.reindex(history_df.columns).fillna(0).values
            
            # 归一化 (防止精度误差)
            return final_w / final_w.sum()
            
        except Exception as e:
            # 如果计算中途出错，打印错误并降级为 Risk Parity (对角线倒数)
            # 这样我们能知道 HRP 还在报错，而不是默默失败
            print(f"[HRP Warning] Calculation failed: {e}. Fallback to Diagonal RP.")
            # 简单的 Risk Parity 兜底
            vol = np.sqrt(np.diag(cov))
            w = 1.0 / (vol + 1e-8)
            return w / w.sum()

    def _get_rec_bisection(self, cov, sort_ix):
        """
        递归二分法 (Recursive Bisection)
        输入: 
            cov: 已排序的协方差矩阵 (DataFrame)
            sort_ix: 排序后的资产名列表
        """
        w = pd.Series(1.0, index=sort_ix)
        c_items = [sort_ix]  # 初始化为一个大簇
        
        while len(c_items) > 0:
            # 将每个簇一分为二
            # 列表推导式：解析出所有需要切分的左右子簇
            c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            
            # 遍历每一对子簇，分配权重
            for i in range(0, len(c_items), 2):
                c_items0 = c_items[i]   # 左子簇
                c_items1 = c_items[i+1] # 右子簇
                
                c_var0 = self._get_cluster_var(cov, c_items0)
                c_var1 = self._get_cluster_var(cov, c_items1)
                
                # 分配因子 alpha (波动率越小，权重越大)
                # 防止除零
                if c_var0 + c_var1 == 0:
                    alpha = 0.5
                else:
                    alpha = 1 - c_var0 / (c_var0 + c_var1)
                
                # 更新权重
                w[c_items0] *= alpha
                w[c_items1] *= 1 - alpha
                
        return w

    def _get_cluster_var(self, cov, c_items):
        """
        计算簇方差 (Cluster Variance)
        假设簇内资产权重为 逆方差权重 (IVP)
        """
        # 提取子协方差矩阵
        cov_slice = cov.loc[c_items, c_items]
        
        # 计算簇内每个资产的逆方差权重
        # np.diag 返回对角线元素 (方差)
        vars_ = np.diag(cov_slice)
        ivp = 1.0 / (vars_ + 1e-8)
        ivp /= ivp.sum()
        
        # 计算该权重下的组合方差: w.T @ Sigma @ w
        # reshape 确保是列向量
        w_ivp = ivp.reshape(-1, 1) 
        
        # Matrix multiplication: (1, N) @ (N, N) @ (N, 1) -> scalar
        cluster_var = np.dot(np.dot(w_ivp.T, cov_slice.values), w_ivp)[0, 0]
        
        return cluster_var
    

# ==============================================================================
# 3. 深度学习策略 (Stateful / Deep Learning)
# ==============================================================================
class DeepMPOStrategy(BaseStrategy):
    def __init__(self, name):
        super().__init__(name)
        
        # 初始化模型 (Factor Model)
        self.model = MPO_Network_Factor().to(self.device).double()
        
        # 维护当前持仓状态 (用于推理时的 w_prev 和 交易成本计算)
        # 初始状态为 1/N
        self.curr_w = torch.ones(1, cfg.NUM_ASSETS, device=self.device, dtype=torch.double) / cfg.NUM_ASSETS
        
        # 优化器在 on_train_period 中动态重新初始化，实现 Warm Start

    def on_train_period(self, train_loader: DataLoader):
        """
        [滚动微调]
        每年调用一次。基于当年的 scaler 和数据，对模型进行 Fine-tune。
        """
        self.model.train()
        
        # 使用较小的学习率进行微调，避免灾难性遗忘
        # 同时也因为我们是继承了去年的参数
        optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        
        # 训练循环
        # 增加 Epochs 以确保充分适应新分布
        fine_tune_epochs = 20 
        
        for epoch in range(fine_tune_epochs):
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device).double()
                y_batch = y_batch.to(self.device).double()
                batch_size = x_batch.size(0)

                # --- 1. 动态构造 w_prev (关键改进) ---
                # 为了让模型学会从任意持仓状态调整到最优，我们不能只输入 1/N。
                # 我们使用 Dirichlet 分布随机采样合法的持仓权重。
                if epoch < 5:
                    # 初期保持稳定，使用 1/N
                    w_prev_batch = torch.ones(batch_size, cfg.NUM_ASSETS, 
                                              device=self.device, dtype=torch.double) / cfg.NUM_ASSETS
                else:
                    # 后期引入噪声，模拟各种持仓情况
                    # alpha=1.0 -> 均匀分布在单纯形上
                    alpha = torch.ones(batch_size, cfg.NUM_ASSETS, device=self.device, dtype=torch.double)
                    w_prev_batch = torch.distributions.Dirichlet(alpha).sample()

                # --- 2. Forward ---
                # 接收 mu 和 L 的输出用于辅助监督
                w_plan, mu_pred, L_pred = self.model(x_batch, w_prev_batch)
                
                # --- 3. Loss Calculation ---
                # A. 组合优化 Loss (针对 w_plan)
                loss_mpo, _ = calc_composite_loss(w_plan, y_batch, w_prev_batch, cost_coeff=cfg.COST_COEFF)
                
                # B. 辅助监督 Loss (针对 mu_pred)
                # 直接监督收益率预测
                loss_mse = torch.nn.functional.mse_loss(mu_pred, y_batch)

                # C. [NEW] 真实风险惩罚 (Realized Risk Penalty)
                # 解决 "Fake L" 问题：如果模型预测的 L 偏小导致 Solver 输出了高风险权重，
                # 那么在真实数据 (y_batch) 上会出现大幅亏损。
                # 我们计算组合的实际亏损超过 CVAR_LIMIT 的部分，进行惩罚。
                # port_ret: (B, H)
                port_ret = (w_plan * y_batch).sum(dim=2) 
                # 惩罚项: ReLU(-Return - Limit) -> 亏损超过 Limit 的幅度
                violation = torch.relu(-port_ret - cfg.CVAR_LIMIT)
                loss_realized_risk = torch.mean(violation**2)

                # Total Loss
                # 增加 1000.0 * loss_realized_risk
                loss = loss_mpo + 1000.0 * loss_mse + 1000.0 * loss_realized_risk
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # 梯度裁剪
                optimizer.step()
                
        # 训练结束后，模型保留参数，等待 daily inference
        # print(f"[{self.name}] Fine-tuning finished.")

    def get_weights(self, history_df, feature_tensor=None):
        """
        [深度推理]
        :param feature_tensor: (1, Lookback, Features) 已经做过 lag 处理和归一化的 Tensor
        """
        if feature_tensor is None:
            raise ValueError("Deep strategy requires feature_tensor input")
            
        self.model.eval()
        
        with torch.no_grad():
            x_in = feature_tensor.to(self.device).double()
            
            # 使用内部维护的 curr_w 作为 w_prev
            # 这是一个 autoregressive 的过程
            w_pred, _, _ = self.model(x_in, self.curr_w)
            
            # 取第一步的动作
            w_action = w_pred[0, 0, :] 
            
            # 更新内部状态
            self.curr_w = w_action.unsqueeze(0)
            
            return w_action.cpu().numpy()
        
# 在 strategy.py 中添加

class DirectGradientStrategy(BaseStrategy):
    """
    直接梯度优化策略 (Direct Gradient Optimization)
    不预测未来参数，而是直接在历史窗口上寻找能最小化 Composite Loss 的最优静态权重。
    这相当于一个使用复杂目标函数（而非均值方差）的"历史最优"策略。
    """
    def __init__(self, name, lookback=60, lr=0.05, steps=100):
        super().__init__(name)
        self.lookback = lookback
        self.lr = lr
        self.steps = steps # 优化步数
        
        # 记录上一次的权重用于计算 Turnover Cost
        self.last_w = None 

    def get_weights(self, history_df, feature_tensor=None):
        """
        通过梯度下降寻找历史窗口上的最优权重
        """
        N = cfg.NUM_ASSETS
        
        # 1. 准备数据
        if len(history_df) < self.lookback:
            return np.ones(N) / N
            
        # 取过去 Lookback 天的收益率
        # shape: (Lookback, N)
        returns_np = history_df.iloc[-self.lookback:].values
        returns_tensor = torch.tensor(returns_np, device=self.device, dtype=torch.double)
        
        # 2. 初始化权重参数 (Learnable Parameter)
        # 初始化为 1/N 或 上一期的权重 (Warm Start)
        if self.last_w is not None:
            init_w = self.last_w.copy()
        else:
            init_w = np.ones(N) / N
            
        w_param = torch.tensor(init_w, device=self.device, dtype=torch.double, requires_grad=True)
        
        # 优化器
        optimizer = optim.Adam([w_param], lr=self.lr)
        
        # 上一期的实际持仓 (用于计算当前的 Turnover Penalty)
        # 如果是第一天，假设是 1/N
        w_prev_tensor = torch.tensor(
            self.last_w if self.last_w is not None else np.ones(N)/N,
            device=self.device, dtype=torch.double
        )

        # 3. 优化循环
        for _ in range(self.steps):
            optimizer.zero_grad()
            
            # 约束处理 (Projected Gradient Descent 的软版本)
            # 我们在计算 loss 前先过一遍 Softmax 或者 Sigmoid? 
            # 为了保持线性特性，我们直接用原始 w_param 计算，但在 step 后进行截断
            
            # --- 计算 Loss ---
            # 假设我们在整个历史窗口期间都持有这个静态权重 w_param
            # 投资组合历史收益序列 R_p = Returns @ w
            # (Lookback, N) @ (N,) -> (Lookback,)
            port_returns = torch.mv(returns_tensor, w_param)
            
            # 1. Sortino Loss (Max Return / Downside Std)
            mean_ret = port_returns.mean()
            downside = torch.clamp(port_returns, max=0)
            downside_std = torch.sqrt(torch.mean(downside**2) + 1e-8)
            sortino = mean_ret / (downside_std + 1e-6)
            loss_sortino = -sortino # Maximize Sortino
            
            # 2. MaxDD Loss
            cum_ret = torch.cumsum(port_returns, dim=0)
            cum_max = torch.cummax(cum_ret, dim=0)[0]
            drawdown = cum_ret - cum_max
            max_dd = torch.min(drawdown) # 负数
            loss_max_dd = torch.abs(max_dd) * cfg.LOSS_GAMMA_DD
            
            # 3. Turnover Penalty (只针对当前调仓的动作)
            # 惩罚 w_param 与 w_prev 的差异
            turnover = torch.norm(w_param - w_prev_tensor, p=1)
            loss_turnover = turnover * cfg.LOSS_GAMMA_TURNOVER * 0.1 # 稍微给小点权重，因为这是单步
            
            total_loss = loss_sortino + loss_max_dd + loss_turnover
            
            total_loss.backward()
            optimizer.step()
            
            # --- 4. 强制约束 (Projection) ---
            with torch.no_grad():
                # a. 单标的约束 w <= 0.3
                w_param.clamp_(0, 0.3)
                
                # b. 总仓位约束 sum(w) <= 1.0
                # 如果和大于1，归一化到1；如果小于1，允许（代表持有现金）
                w_sum = w_param.sum()
                if w_sum > 1.0:
                    w_param.div_(w_sum)
                    
        # 4. 输出最终结果
        w_final = w_param.detach().cpu().numpy()
        self.last_w = w_final
        return w_final
    
# 添加到 strategy.py

class DeepE2EStrategy(BaseStrategy):
    def __init__(self, name):
        super().__init__(name)
        from model import E2E_Network # 延迟导入防循环
        self.model = E2E_Network().to(self.device).double()
        self.curr_w = torch.ones(1, cfg.NUM_ASSETS, device=self.device, dtype=torch.double) / cfg.NUM_ASSETS

    def on_train_period(self, train_loader: DataLoader):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
        fine_tune_epochs = 10
        
        for epoch in range(fine_tune_epochs):
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device).double()
                y_batch = y_batch.to(self.device).double()
                
                # Forward
                w_plan, _, _ = self.model(x_batch) # Output: (Batch, 1, Assets)
                
                # w_plan 是 (Batch, 1, N)，我们需要 (Batch, N)
                w_curr = w_plan.squeeze(1)
                
                # --- 修复点开始 ---
                # y_batch 是 (Batch, Horizon, Assets)，例如 (64, 5, 10)
                # E2E 策略只优化下一天 (t+1) 的 Sharpe，所以只取第 0 个时间步
                y_target = y_batch[:, 0, :] # Shape: (Batch, Assets)
                
                # Portfolio Return: (Batch, N) * (Batch, N) -> sum -> (Batch,)
                port_ret = (w_curr * y_target).sum(dim=1)
                # --- 修复点结束 ---
                
                # Negative Sharpe Ratio
                mean_ret = port_ret.mean()
                std_ret = port_ret.std() + 1e-6
                sharpe = mean_ret / std_ret
                
                loss = -sharpe
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def get_weights(self, history_df, feature_tensor=None):
        if feature_tensor is None: return np.ones(cfg.NUM_ASSETS)/cfg.NUM_ASSETS
        
        self.model.eval()
        with torch.no_grad():
            x_in = feature_tensor.to(self.device).double()
            w_pred, _, _ = self.model(x_in)
            w_action = w_pred[0, 0, :]
            
            # 手动施加约束 (Fair Play)
            # 虽然 Softmax 保证了 sum=1, 但不保证 <= 0.3
            # 如果我们想看看纯黑盒的效果，可以不加；
            # 如果想做严格控制变量对比，建议加上类似 DirectGradient 的后处理
            
            # 这里我们保持原样，看看 Softmax 倾向于怎么分配
            return w_action.cpu().numpy()
        
