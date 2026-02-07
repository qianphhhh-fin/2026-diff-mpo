import torch
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.dates as mdates
import sys

from config import cfg
from data_loader import load_and_process_data
from model import MPO_Network
from mpo_solver import DifferentiableMPO

# 设置绘图风格
plt.style.use('seaborn-v0_8')

# ==========================================
# 1. 策略基类
# ==========================================
class BaseStrategy:
    def __init__(self, name):
        self.name = name
    
    def get_weights(self, prices_df, current_weights, context_data=None):
        raise NotImplementedError

# ==========================================
# 2. 深度学习策略组
# ==========================================
class DeepStrategy(BaseStrategy):
    def __init__(self, name, model_path, mode='mpo'):
        super().__init__(name)
        self.mode = mode
        self.device = cfg.DEVICE
        
        # 加载模型
        self.model = MPO_Network().to(self.device).double()
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"✅ [DeepStrategy] 成功加载模型: {name}")
        except Exception as e:
            print(f"⚠️ [DeepStrategy] 无法加载模型 {name}: {e}")
            # 如果加载失败，不要让它跑，直接报错，避免浪费时间
            raise RuntimeError(f"模型文件不存在或损坏: {model_path}")

        self.model.eval()
        
        if mode == 'two_stage':
            self.solver_layer = DifferentiableMPO() 
            
    def get_weights(self, prices_df, current_weights, context_data):
        # context_data is (x_window, _)
        x_tensor, _ = context_data
        
        # 增加维度检查
        if x_tensor.shape[-1] != cfg.INPUT_FEATURE_DIM:
            raise ValueError(f"输入特征维度错误! 模型期望 {cfg.INPUT_FEATURE_DIM}, 实际得到 {x_tensor.shape[-1]}")

        x_tensor = x_tensor.to(self.device).double()
        w_prev = torch.tensor(current_weights, device=self.device, dtype=torch.double).unsqueeze(0)
        
        with torch.no_grad():
            if 'Diff-MPO' in self.name or 'Ours' in self.name:
                w_plan, _, _ = self.model(x_tensor, w_prev)
                return w_plan[0, 0, :].cpu().numpy()
            
            elif 'Two-Stage' in self.name:
                _, (h_n, _) = self.model.lstm(x_tensor)
                context = h_n[-1]
                mu_pred = self.model.mu_head(context).view(1, cfg.PREDICT_HORIZON, cfg.NUM_ASSETS)
                L_flat = self.model.L_head(context)
                L_pred = L_flat.view(1, cfg.PREDICT_HORIZON, cfg.NUM_ASSETS, cfg.NUM_ASSETS)
                
                # 构造正定矩阵
                mask = torch.tril(torch.ones_like(L_pred))
                L_pred = L_pred * mask
                diag_mask = torch.eye(cfg.NUM_ASSETS, device=self.device).view(1, 1, cfg.NUM_ASSETS, cfg.NUM_ASSETS)
                L_pred = L_pred + diag_mask * (torch.nn.functional.softplus(L_pred) + 1e-5 - L_pred)

                w_plan = self.solver_layer(mu_pred.cpu(), L_pred.cpu(), w_prev.cpu())
                return w_plan[0, 0, :].detach().numpy()
                
            elif 'Neural Risk Parity' in self.name:
                _, (h_n, _) = self.model.lstm(x_tensor)
                L_flat = self.model.L_head(h_n[-1])
                L_pred = L_flat.view(1, cfg.PREDICT_HORIZON, cfg.NUM_ASSETS, cfg.NUM_ASSETS)
                L_t0 = L_pred[0, 0, :, :]
                Sigma_t0 = L_t0 @ L_t0.T
                vols = torch.sqrt(torch.diagonal(Sigma_t0))
                raw_w = 1.0 / (vols + 1e-6)
                return (raw_w / raw_w.sum()).cpu().numpy()

# ==========================================
# 3. 经典优化策略组 (Strict Mode)
# ==========================================
class OptimizationStrategy(BaseStrategy):
    def __init__(self, name, lookback=126, lambda_cost=0.0):
        super().__init__(name)
        self.lookback = lookback
        self.lambda_cost = lambda_cost
        
    def get_weights(self, history_df, current_weights, context_data=None):
        """
        history_df: 包含过去 N 天的收益率数据 (index是日期)
        """
        # 1. 数据检查
        if history_df is None or len(history_df) < self.lookback:
            # 如果是刚开始回测，数据不足，可以使用 1/N，但不应该一直发生
            return np.ones(cfg.NUM_ASSETS) / cfg.NUM_ASSETS
            
        # 截取窗口
        returns_window = history_df.iloc[-self.lookback:].values
        
        # 检查 NaN
        if np.isnan(returns_window).any():
            # 尝试填充
            returns_window = np.nan_to_num(returns_window)
            
        # 2. 参数估计
        mu_est = np.mean(returns_window, axis=0)
        cov_est = np.cov(returns_window.T)
        
        # === 关键修复：协方差正则化 ===
        # 防止矩阵奇异导致 Solver 失败
        cov_est += 1e-6 * np.eye(len(mu_est))
        
        N = cfg.NUM_ASSETS
        w = cp.Variable(N)
        w_prev = current_weights
        
        # 交易成本项 (L1 Norm)
        cost_term = cp.norm(w - w_prev, 1) 
        
        # 3. 构建问题
        if 'Mean-Variance' in self.name:
            risk_aversion = 1.0 # 稍微降低一点，太高容易导致数值问题
            ret = mu_est @ w
            risk = cp.quad_form(w, cov_est)
            obj_expr = ret - risk_aversion * risk
            if self.lambda_cost > 0:
                obj_expr -= self.lambda_cost * cost_term
            objective = cp.Maximize(obj_expr)
            
        elif 'Global Min Var' in self.name:
            risk = cp.quad_form(w, cov_est)
            obj_expr = risk
            if self.lambda_cost > 0:
                obj_expr += self.lambda_cost * cost_term
            objective = cp.Minimize(obj_expr)
            
        elif 'Mean-CVaR' in self.name:
            # CVaR 比较难解，需要引入辅助变量
            # CVaR(alpha) = alpha + 1/(1-c) * mean(max(-w*r - alpha, 0))
            # 这里简化处理，如果不收敛直接抛错
            alpha = cp.Variable()
            # 注意：returns_window 是 (T, N)，这里要变成 (T,)
            port_returns = returns_window @ w 
            losses = -port_returns
            
            cvar_limit = 0.95
            cvar_term = alpha + (1.0 / ((1.0 - cvar_limit) * self.lookback)) * cp.sum(cp.pos(losses - alpha))
            
            obj_expr = cvar_term
            if self.lambda_cost > 0:
                obj_expr += self.lambda_cost * cost_term
            objective = cp.Minimize(obj_expr)
            
        else:
            raise ValueError(f"未知的优化策略: {self.name}")
            
        # 约束条件
        constraints = [
            cp.sum(w) == 1, 
            w >= 0 
        ]
        
        prob = cp.Problem(objective, constraints)
        
        # 4. 求解 (Strict Mode)
        try:
            # 尝试 ECOS，如果失败尝试 SCS
            prob.solve(solver=cp.ECOS, abstol=1e-5)
            
            if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                # 如果 ECOS 失败，尝试 SCS
                prob.solve(solver=cp.SCS, eps=1e-4)
            
            if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                raise ValueError(f"Solver Status: {prob.status}")
                
            if w.value is None:
                raise ValueError("Solver returned None weights")
                
            # 归一化并处理微小负值
            res_w = np.clip(w.value, 0, 1)
            return res_w / res_w.sum()
            
        except Exception as e:
            # === 严厉报错 ===
            print(f"\n❌ [CRITICAL ERROR] 策略 {self.name} 优化失败！")
            print(f"   原因: {e}")
            print(f"   日期索引: 数据的最后一行是 {history_df.index[-1]}")
            print(f"   Cov 矩阵条件数: {np.linalg.cond(cov_est)}")
            raise e # 直接抛出，终止程序

# ==========================================
# 4. 规则策略
# ==========================================
class RuleStrategy(BaseStrategy):
    def get_weights(self, history_df, current_weights, context_data=None):
        N = cfg.NUM_ASSETS
        
        if '1/N' in self.name:
            return np.ones(N) / N
            
        elif 'Vanilla Risk Parity' in self.name:
            # 过去 1 年的波动率倒数
            # 这里的 history_df 已经是 returns
            window = 252
            if len(history_df) < window:
                window = len(history_df)
                
            returns = history_df.iloc[-window:].values
            vols = np.std(returns, axis=0)
            
            # 防除零
            vols[vols < 1e-6] = 1e-6
            
            w = 1.0 / vols
            return w / np.sum(w)
            
        elif 'Factor Momentum' in self.name:
            # 简单的截面动量：买过去 60 天涨幅最好的 3 个
            ret_accum = history_df.iloc[-60:].sum()
            # 选 top 3
            top_k_indices = ret_accum.argsort()[-3:]
            w = np.zeros(N)
            w[top_k_indices] = 1.0 / 3.0
            return w

# ==========================================
# 5. 回测主引擎
# ==========================================
def run_backtest():
    print("⚔️ 开启回测竞技场 (Strict Mode Debugging) ...")
    
    # 1. 准备数据
    _, test_loader, scaler = load_and_process_data()
    
    # 读取原始 CSV (用于 Optimization Strategy 的输入)
    df_raw = pd.read_csv(cfg.DATA_PATH, index_col=0, parse_dates=True)
    # 只保留资产列用于计算回报
    df_assets_ret = df_raw[cfg.ASSETS]
    
    # 确定测试集起始点
    split_date = pd.Timestamp(cfg.TRAIN_SPLIT_DATE)
    
    # 2. 初始化策略
    strategies = [
        # Deep Models
        DeepStrategy('Ours (Diff-MPO)', 'models/diff_mpo_sharpe.pth', mode='mpo'),
        DeepStrategy('Two-Stage (MSE)', 'models/baseline_mse_model.pth', mode='two_stage'),
        DeepStrategy('Neural Risk Parity', 'models/baseline_vol_model.pth', mode='nrp'),
        
        # Optimization Models (Classic)
        OptimizationStrategy('Mean-Variance', lookback=60, lambda_cost=0.0),
        OptimizationStrategy('Global Min Var', lookback=60, lambda_cost=0.0),
        # Mean-CVaR 计算太慢且容易无解，暂时注释掉，先调通上面两个
        # OptimizationStrategy('Mean-CVaR', lookback=60, lambda_cost=0.0), 
        
        # Rule Based
        RuleStrategy('1/N'),
        RuleStrategy('Vanilla Risk Parity (1Y)'), 
    ]
    
    results = {s.name: {'wealth': [cfg.INIT_WEALTH], 'turnover': []} for s in strategies}
    current_weights = {s.name: np.ones(cfg.NUM_ASSETS)/cfg.NUM_ASSETS for s in strategies}
    
    full_w_plans = {s.name: [] for s in strategies}
    
    # 找到测试集的起始索引
    # 注意：MPODataset 的 test_loader 是从 split_date 开始的
    # 我们需要找到 df_assets_ret 中对应 split_date 的位置
    test_indices = np.where(df_assets_ret.index >= split_date)[0]
    if len(test_indices) == 0:
        raise ValueError("测试集为空！请检查 TRAIN_SPLIT_DATE")
    start_idx = test_indices[0]
    
    print(f"   测试集起始日期: {df_assets_ret.index[start_idx]}")
    print(f"   总样本数: {len(test_loader) * cfg.BATCH_SIZE}")
    
    # --- 生成决策序列 ---
    print("   正在生成决策序列...")
    
    # 为了对齐，我们需要遍历 DataLoader
    # 每个 Batch 对应 test_loader 中的一段
    
    global_step = 0
    
    for batch_idx, (x_batch, _) in enumerate(tqdm(test_loader)):
        for i in range(x_batch.size(0)):
            # 当前在原始 DataFrame 中的绝对索引
            # 注意：test_loader 里的 x 是归一化后的，用于深度模型
            # 传统模型需要原始收益率数据
            
            curr_abs_idx = start_idx + global_step
            if curr_abs_idx >= len(df_assets_ret): break
            
            current_date = df_assets_ret.index[curr_abs_idx]
            
            # Deep Model Input
            x_sample = x_batch[i].unsqueeze(0)
            
            for strat in strategies:
                w_curr = current_weights[strat.name]
                
                try:
                    if isinstance(strat, DeepStrategy):
                        # 深度学习模型使用 Tensor 输入
                        w_target = strat.get_weights(None, w_curr, context_data=(x_sample, None))
                    else:
                        # 传统模型使用历史 DataFrame 切片
                        # 必须包含直到 current_date 的数据
                        history_slice = df_assets_ret.iloc[:curr_abs_idx+1] 
                        w_target = strat.get_weights(history_slice, w_curr)
                        
                except Exception as e:
                    print(f"\n❌ 策略 {strat.name} 在 {current_date} 崩溃！")
                    print(f"错误信息: {e}")
                    sys.exit(1) # 强制退出，方便你看报错

                # 格式转换与安全检查
                w_target = np.array(w_target, dtype=np.float64).reshape(-1)
                
                # 再次归一化防止浮点误差
                if w_target.sum() > 1e-6:
                    w_target = w_target / w_target.sum()
                
                full_w_plans[strat.name].append(w_target)
                current_weights[strat.name] = w_target
            
            global_step += 1
    
    # --- 计算净值 ---
    print("\n   正在计算净值与归因...")
    
    # 截取实际回测长度的收益率
    # 注意：w_t 决定的是 t+1 的收益
    # full_w_plans 长度为 T，对应的收益率应该是从 start_idx + 1 开始
    
    n_steps = len(full_w_plans['1/N'])
    realized_ret = df_assets_ret.iloc[start_idx+1 : start_idx+1+n_steps].values
    plot_dates = df_assets_ret.index[start_idx+1 : start_idx+1+n_steps]
    
    # 如果生成的权重比收益率多1个（最后一天决策），截断权重
    for k in full_w_plans:
        full_w_plans[k] = full_w_plans[k][:len(realized_ret)]
    
    metrics = []
    
    for strat_name in full_w_plans:
        weights_seq = np.array(full_w_plans[strat_name]) # (T, N)
        
        # 收益计算: R_p = sum(w_{t-1} * r_t)
        # 这里的 weights_seq[t] 是在 t 时刻做出的决策，享受 r_{t+1} 的收益
        # 但我们在循环里实际上是 aligned 的：
        # loop step k: current_date=k, 做出 w_target. 这个 w_target 也就是 w_k
        # 它的收益应该是 realized_ret[k] (即 k+1 天的收益)
        
        # 简单起见：
        port_ret = (weights_seq * realized_ret).sum(axis=1)
        
        # 换手率计算
        # w_diff = |w_t - w_{t-1}|
        # 注意：这里为了简化，假设 w_t 直接变成 w_{t+1}，忽略日内价格变动导致的权重漂移
        w_diff = np.abs(weights_seq[1:] - weights_seq[:-1]).sum(axis=1)
        # 补上第一天的换手
        w_diff = np.insert(w_diff, 0, 0.0) 
        
        turnover = w_diff
        cost = turnover * cfg.COST_COEFF
        net_ret = port_ret - cost
        
        wealth = np.cumprod(1 + net_ret)
        results[strat_name]['wealth'] = wealth
        
        # --- 修改点：增加 Sortino 和 Calmar 指标 ---
        ann_ret = np.mean(net_ret) * 252
        ann_vol = np.std(net_ret) * np.sqrt(252)
        sharpe = (ann_ret - 0.02) / (ann_vol + 1e-6)
        
        # 计算 Sortino (只看下行波动)
        downside_ret = net_ret.copy()
        downside_ret[downside_ret > 0] = 0
        downside_vol = np.std(downside_ret) * np.sqrt(252)
        sortino = (ann_ret - 0.02) / (downside_vol + 1e-6)
        
        # 计算回撤
        cum_max = np.maximum.accumulate(wealth)
        drawdown = (wealth - cum_max) / cum_max
        max_dd = drawdown.min()
        
        # 计算 Calmar (年化收益 / 最大回撤)
        calmar = ann_ret / (abs(max_dd) + 1e-6)
        
        avg_turnover = np.mean(turnover)
        
        metrics.append({
            'Strategy': strat_name,
            'Ann Return': f"{ann_ret:.2%}",
            'Sharpe': f"{sharpe:.2f}",
            'Sortino': f"{sortino:.2f}",  # 新增
            'Max DD': f"{max_dd:.2%}",
            'Calmar': f"{calmar:.2f}",    # 新增
            'Turnover': f"{avg_turnover:.2%}",
            '_sort_key': sortino          # 按 Sortino 排序 (这是我们的训练目标)
        })
        
    metrics_df = pd.DataFrame(metrics).sort_values('_sort_key', ascending=False).drop(columns='_sort_key')
    print("\n🏆 回测结果排行榜 (Test Set):")
    print(metrics_df)
    metrics_df.to_csv('backtest_metrics.csv', index=False)
    
    # --- 绘图 ---
    plt.figure(figsize=(12, 6))
    for strat_name in results:
        wealth = results[strat_name]['wealth']
        plt.plot(plot_dates, wealth, label=strat_name)
        
    plt.title('Cumulative Wealth Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('backtest_wealth_curve.png', dpi=300)
    print(f"\n📈 结果已保存。")

if __name__ == "__main__":
    run_backtest()