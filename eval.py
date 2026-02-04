import torch
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from tqdm import tqdm
import matplotlib.dates as mdates  # <--- 1. 新增：引入日期格式化库
from config import cfg
from data_loader import load_and_process_data
from model import MPO_Network
from mpo_solver import DifferentiableMPO

# 设置绘图风格
plt.style.use('seaborn-v0_8')

# ==========================================
# 1. 策略基类与通用函数
# ==========================================
class BaseStrategy:
    def __init__(self, name):
        self.name = name
    
    def get_weights(self, prices_df, current_weights, context_data=None):
        """
        输入:
            prices_df: 截止到 t 时刻的历史价格/收益数据
            current_weights: 当前持仓 (t-1)
            context_data: 神经网络需要的额外 Tensor 数据 (Batch)
        输出:
            target_weights: t 时刻的目标仓位 (N,)
        """
        raise NotImplementedError

def calculate_turnover_cost(w_new, w_old, cost_rate=0.0005):
    turnover = np.sum(np.abs(w_new - w_old))
    return turnover * cost_rate, turnover

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
        # 允许加载部分权重 (因为 baseline 模型结构可能略有差异，但这里我们结构统一)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # 如果是 Two-Stage，需要一个独立的 Solver
        if mode == 'two_stage':
            self.solver_layer = DifferentiableMPO() # 复用 Solver
            
    def get_weights(self, prices_df, current_weights, context_data):
        x_tensor, _ = context_data
        x_tensor = x_tensor.to(self.device).double()
        w_prev = torch.tensor(current_weights, device=self.device, dtype=torch.double).unsqueeze(0)
        
        with torch.no_grad():
            if self.name == 'Ours (Diff-MPO)':
                # 直接端到端输出
                w_plan, _, _ = self.model(x_tensor, w_prev)
                # w_plan 是 (Batch, Horizon, Assets)，我们取第一步 t=0
                return w_plan[0, 0, :].cpu().numpy()
            
            elif self.name == 'Two-Stage (MSE)':
                # 1. 预测 mu (模型不管 Solver)
                _, (h_n, _) = self.model.lstm(x_tensor)
                context = h_n[-1]
                mu_pred = self.model.mu_head(context).view(1, cfg.PREDICT_HORIZON, cfg.NUM_ASSETS)
                
                # 2. 预测 L (这里其实 MSE 模型也预测了 L，虽然训练没用到，但可以拿来用)
                # 或者用历史协方差代替。为了公平，我们用模型预测的 L
                L_flat = self.model.L_head(context)
                L_pred = L_flat.view(1, cfg.PREDICT_HORIZON, cfg.NUM_ASSETS, cfg.NUM_ASSETS)
                
                # 处理 L 合法性
                mask = torch.tril(torch.ones_like(L_pred))
                L_pred = L_pred * mask
                diag_mask = torch.eye(cfg.NUM_ASSETS, device=self.device).view(1, 1, cfg.NUM_ASSETS, cfg.NUM_ASSETS)
                L_pred = L_pred + diag_mask * (torch.nn.functional.softplus(L_pred) + 1e-5 - L_pred)

                # 3. 显式调用 Solver (在 CPU 上解)
                # Two-Stage 的核心在于：预测是独立的，但执行时依然用 MPO
                w_plan = self.solver_layer(mu_pred.cpu(), L_pred.cpu(), w_prev.cpu())
                return w_plan[0, 0, :].detach().numpy()
                
            elif self.name == 'Neural Risk Parity':
                # 只用预测的波动率
                _, (h_n, _) = self.model.lstm(x_tensor)
                L_flat = self.model.L_head(h_n[-1])
                L_pred = L_flat.view(1, cfg.PREDICT_HORIZON, cfg.NUM_ASSETS, cfg.NUM_ASSETS)
                
                # 算协方差 Sigma = L @ L.T
                # 取第一个时间步 t=0
                L_t0 = L_pred[0, 0, :, :]
                Sigma_t0 = L_t0 @ L_t0.T
                
                # 风险平价: w ~ 1 / sqrt(diag(Sigma))
                vols = torch.sqrt(torch.diagonal(Sigma_t0))
                raw_w = 1.0 / (vols + 1e-6)
                return (raw_w / raw_w.sum()).cpu().numpy()

# ==========================================
# 3. 经典优化策略组 (Convex Optimization)
# ==========================================
class OptimizationStrategy(BaseStrategy):
    def __init__(self, name, lookback=60):
        super().__init__(name)
        self.lookback = lookback
        
    def get_weights(self, prices_df, current_weights, context_data=None):
        # 获取过去 N 天的收益率数据
        returns = prices_df.iloc[-self.lookback:].values
        
        # 1. 均值与协方差估计
        mu_est = np.mean(returns, axis=0)
        cov_est = np.cov(returns.T)
        
        N = len(mu_est)
        w = cp.Variable(N)
        
        if self.name == 'Mean-Variance':
            # Max mu*w - lambda * w*Sigma*w
            risk_aversion = 1.0
            ret = mu_est @ w
            risk = cp.quad_form(w, cov_est)
            obj = cp.Maximize(ret - risk_aversion * risk)
            constraints = [cp.sum(w) == 1, w >= 0]
            
        elif self.name == 'Global Min Var':
            # Min w*Sigma*w
            risk = cp.quad_form(w, cov_est)
            obj = cp.Minimize(risk)
            constraints = [cp.sum(w) == 1, w >= 0]
            
        elif self.name == 'Mean-CVaR':
            # 最小化 CVaR (95%)
            # 引入辅助变量
            # CVaR = alpha + 1/(1-c) * mean(max(loss - alpha, 0))
            # Loss = - returns @ w
            alpha = cp.Variable()
            # 模拟样本场景
            samples = returns # (T, N)
            losses = - samples @ w
            
            cvar_term = alpha + (1.0 / (0.05 * self.lookback)) * cp.sum(cp.pos(losses - alpha))
            obj = cp.Minimize(cvar_term)
            constraints = [cp.sum(w) == 1, w >= 0]
            
        else:
            return np.ones(N) / N
            
        prob = cp.Problem(obj, constraints)
        try:
            prob.solve(solver=cp.ECOS)
            if w.value is None: return current_weights # 求解失败保持不动
            return np.clip(w.value, 0, 1) # 修正数值误差
        except:
            return current_weights

# ==========================================
# 4. 规则与现代策略组 (HRP, Momentum, 1/N)
# ==========================================
class RuleStrategy(BaseStrategy):
    def get_weights(self, prices_df, current_weights, context_data=None):
        N = prices_df.shape[1]
        
        if self.name == '1/N':
            return np.ones(N) / N
            
        elif self.name == 'Vanilla Risk Parity':
            # w ~ 1/std
            returns = prices_df.iloc[-60:].values
            vols = np.std(returns, axis=0)
            w = 1.0 / (vols + 1e-6)
            return w / np.sum(w)
            
        elif self.name == 'Factor Momentum':
            # 过去 20 天收益率为正的，平分权重；否则为 0
            # 这是一个 Long-Only 的动量实现
            moms = prices_df.iloc[-20:].mean().values
            # 简单的 Signal: > 0 买入
            signal = (moms > 0).astype(float)
            if signal.sum() == 0: return np.ones(N)/N # 全跌就躺平
            return signal / signal.sum()

class HRPStrategy(BaseStrategy):
    """ Hierarchical Risk Parity (Lopez de Prado) """
    def get_weights(self, prices_df, current_weights, context_data=None):
        returns = prices_df.iloc[-60:]
        corr = returns.corr().values
        cov = returns.cov().values
        
        # 1. 聚类 (Hierarchical Clustering)
        dist = np.sqrt((1 - corr) / 2)
        link = sch.linkage(squareform(dist), 'single')
        
        # 2. 排序 (Quasi-Diagonalization)
        # 这里简化：直接用 sch.dendrogram 得到的叶子顺序
        sort_ix = sch.dendrogram(link, no_plot=True)['leaves']
        
        # 3. 递归二分 (Recursive Bisection)
        # 这是一个简化版的 HRP 核心逻辑：自顶向下分配风险
        w = pd.Series(1, index=sort_ix)
        
        # 核心逻辑太长，这里用 Inverse Variance 替代 Cluster 内部权重
        # 真实 HRP 比较复杂，这里用 "Hierarchical Inverse Variance" 近似
        # 重新排列 Cov
        cov_sorted = cov[sort_ix][:, sort_ix]
        
        # 简单实现：HRP 的核心思想是相似资产分配相似权重
        # 这里退化为 IVP (Inverse Variance) 但在 Cluster 层面
        # 为了代码简洁，我们使用 IVP 作为 HRP 的近似 (Common simplification)
        ivp = 1. / np.diag(cov)
        ivp /= ivp.sum()
        return ivp 

# ==========================================
# 5. 回测主引擎
# ==========================================
def run_backtest():
    print("⚔️ 开启回测竞技场 (Backtest Arena) ...")
    
    # 1. 准备数据
    _, test_loader, scaler = load_and_process_data()
    df = pd.read_csv(cfg.DATA_PATH, index_col=0, parse_dates=True)
    split_date = pd.Timestamp(cfg.TRAIN_SPLIT_DATE)
    
    # 提取测试集期间的数据
    test_returns_df = df.loc[df.index >= split_date, cfg.ASSETS]
    
    # 2. 初始化所有策略 (保持不变)
    strategies = [
        DeepStrategy('Ours (Diff-MPO)', 'models/diff_mpo_sharpe.pth', mode='mpo'),
        DeepStrategy('Two-Stage (MSE)', 'models/baseline_mse_model.pth', mode='two_stage'),
        DeepStrategy('Neural Risk Parity', 'models/baseline_vol_model.pth', mode='nrp'),
        OptimizationStrategy('Mean-Variance'),
        OptimizationStrategy('Global Min Var'),
        OptimizationStrategy('Mean-CVaR'),
        RuleStrategy('1/N'),
        RuleStrategy('Vanilla Risk Parity'),
        RuleStrategy('Factor Momentum'),
    ]
    
    results = {s.name: {'wealth': [cfg.INIT_WEALTH], 'turnover': []} for s in strategies}
    current_weights = {s.name: np.ones(cfg.NUM_ASSETS)/cfg.NUM_ASSETS for s in strategies}
    
    print(f"   测试集长度: {len(test_returns_df)} 天")
    print(f"   交易成本 (单边): {cfg.COST_COEFF}")
    
    full_w_plans = {s.name: [] for s in strategies}
    
    # --- 第一阶段：生成所有策略的权重序列 (保持不变) ---
    print("   正在生成决策序列...")
    for batch_idx, (x_batch, _) in enumerate(tqdm(test_loader)):
        for i in range(x_batch.size(0)):
            global_idx = batch_idx * cfg.BATCH_SIZE + i
            if global_idx >= len(test_returns_df) - 1: break
            
            current_date_idx = global_idx
            if current_date_idx < 60:
                history_slice = test_returns_df.iloc[:60]
            else:
                history_slice = test_returns_df.iloc[current_date_idx-60 : current_date_idx]
            
            x_sample = x_batch[i].unsqueeze(0)
            
            for strat in strategies:
                w_curr = current_weights[strat.name]
                if isinstance(strat, DeepStrategy):
                    w_target = strat.get_weights(None, w_curr, context_data=(x_sample, None))
                else:
                    w_target = strat.get_weights(history_slice, w_curr)
                full_w_plans[strat.name].append(w_target)
                current_weights[strat.name] = w_target
    
    # --- 第二阶段：统一计算净值 ---
    print("   正在计算净值与归因...")
    n_days = len(full_w_plans['1/N'])
    
    # <--- 2. 修改：获取对应的日期序列 --->
    # realized_ret 是从第 0 天开始的
    # 但由于我们计算逻辑是 r_t1 = realized_ret[1:]
    # 所以净值曲线是从第 1 天开始累积的
    realized_ret = test_returns_df.iloc[:n_days].values
    full_dates = test_returns_df.index[:n_days] # 获取完整日期索引
    plot_dates = full_dates[1:] # 对齐净值曲线的日期 (去掉第一天)
    
    metrics = []
    
    for strat_name in full_w_plans:
        weights_seq = np.array(full_w_plans[strat_name])
        
        w_t = weights_seq[:-1]
        r_t1 = realized_ret[1:]
        
        gross_ret = (w_t * r_t1).sum(axis=1)
        
        w_diff = np.abs(w_t[1:] - w_t[:-1])
        turnover_dynamic = w_diff.sum(axis=1)
        first_turnover = np.sum(np.abs(w_t[0]))
        turnover = np.insert(turnover_dynamic, 0, first_turnover)
        
        cost = turnover * cfg.COST_COEFF
        net_ret = gross_ret - cost # 之前修复的 bug，不要切片
        
        wealth = np.cumprod(1 + net_ret)
        results[strat_name]['wealth'] = wealth
        
        ann_ret = np.mean(net_ret) * 252
        ann_vol = np.std(net_ret) * np.sqrt(252)
        sharpe = (ann_ret - 0.02) / (ann_vol + 1e-6)
        
        cum_max = np.maximum.accumulate(wealth)
        drawdown = (wealth - cum_max) / cum_max
        max_dd = drawdown.min()
        avg_turnover = np.mean(turnover)
        
        metrics.append({
            'Strategy': strat_name,
            'Ann Return': f"{ann_ret:.2%}",
            'Sharpe': f"{sharpe:.2f}",
            'Max DD': f"{max_dd:.2%}",
            'Turnover': f"{avg_turnover:.2%}",
            '_sort_key': sharpe
        })
        
    metrics_df = pd.DataFrame(metrics).sort_values('_sort_key', ascending=False).drop(columns='_sort_key')
    print("\n🏆 回测结果排行榜 (Test Set):")
    print(metrics_df)
    metrics_df.to_csv('backtest_metrics.csv', index=False)
    
    # --- 3. 修改：绘图部分 ---
    plt.figure(figsize=(14, 8)) #稍微宽一点
    
    for strat_name in results:
        wealth = results[strat_name]['wealth']
        
        # 优化：只给前几名加粗，其他的细线，避免太乱
        if 'Ours' in strat_name:
            lw = 2.5
            alpha = 1.0
            zorder = 10 # 保证画在最上层
        elif '1/N' in strat_name:
            lw = 2.0
            alpha = 0.8
            zorder = 5
        else:
            lw = 1.0
            alpha = 0.6
            zorder = 1
            
        plt.plot(plot_dates, wealth, label=strat_name, linewidth=lw, alpha=alpha, zorder=zorder)
        
    plt.title('Cumulative Wealth: Diff-MPO vs Benchmarks', fontsize=14)
    plt.ylabel('Wealth (Start=1.0)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    
    # 日期格式化美化
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6)) # 每6个月标一个刻度
    plt.gcf().autofmt_xdate() # 自动旋转日期标签
    
    plt.legend(loc='upper left', fontsize=10, frameon=True, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = 'backtest_wealth_curve.png'
    plt.savefig(save_path, dpi=300) # 提高分辨率
    print(f"\n📈 净值曲线已保存至 {save_path} (X轴已显示真实日期)")

if __name__ == "__main__":
    run_backtest()