import torch
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from tqdm import tqdm
import matplotlib.dates as mdates

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
        except Exception as e:
            print(f"⚠️ Warning: Could not load model {name} from {model_path}. Error: {e}")
        self.model.eval()
        
        if mode == 'two_stage':
            self.solver_layer = DifferentiableMPO() 
            
    def get_weights(self, prices_df, current_weights, context_data):
        x_tensor, _ = context_data
        x_tensor = x_tensor.to(self.device).double()
        w_prev = torch.tensor(current_weights, device=self.device, dtype=torch.double).unsqueeze(0)
        
        with torch.no_grad():
            if 'Diff-MPO' in self.name:
                w_plan, _, _ = self.model(x_tensor, w_prev)
                return w_plan[0, 0, :].cpu().numpy()
            
            elif 'Two-Stage' in self.name:
                _, (h_n, _) = self.model.lstm(x_tensor)
                context = h_n[-1]
                mu_pred = self.model.mu_head(context).view(1, cfg.PREDICT_HORIZON, cfg.NUM_ASSETS)
                L_flat = self.model.L_head(context)
                L_pred = L_flat.view(1, cfg.PREDICT_HORIZON, cfg.NUM_ASSETS, cfg.NUM_ASSETS)
                
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
# 3. 经典优化策略组 (修复 Name Mismatch Bug)
# ==========================================
class OptimizationStrategy(BaseStrategy):
    def __init__(self, name, lookback=126, lambda_cost=0.0):
        super().__init__(name)
        self.lookback = lookback
        self.lambda_cost = lambda_cost
        
    def get_weights(self, prices_df, current_weights, context_data=None):
        if prices_df.empty or len(prices_df) < 2:
            return current_weights
            
        returns = prices_df.iloc[-self.lookback:].values
        if np.isnan(returns).all(): return current_weights
        returns = np.nan_to_num(returns)
        
        mu_est = np.mean(returns, axis=0)
        cov_est = np.cov(returns.T)
        
        N = len(mu_est)
        if N != cfg.NUM_ASSETS: return current_weights

        w = cp.Variable(N)
        w_prev = current_weights
        cost_term = cp.norm(w - w_prev, 1) 
        
        # ⚠️ 修复：使用 'in' 进行模糊匹配
        if 'Mean-Variance' in self.name:
            risk_aversion = 2.0 
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
            alpha = cp.Variable()
            losses = - returns @ w
            cvar_term = alpha + (1.0 / (0.05 * self.lookback)) * cp.sum(cp.pos(losses - alpha))
            obj_expr = cvar_term
            if self.lambda_cost > 0:
                obj_expr += self.lambda_cost * cost_term
            objective = cp.Minimize(obj_expr)
            
        else:
            return np.ones(N) / N
            
        constraints = [cp.sum(w) == 1, w >= 0]
        
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.ECOS, abstol=1e-4)
            if w.value is None: return current_weights
            return np.clip(w.value, 0, 1)
        except:
            return current_weights

# ==========================================
# 4. HRP 与 规则策略 (修复 Name Mismatch Bug)
# ==========================================
class RuleStrategy(BaseStrategy):
    def get_weights(self, prices_df, current_weights, context_data=None):
        N = prices_df.shape[1]
        
        if '1/N' in self.name:
            return np.ones(N) / N
            
        elif 'Vanilla Risk Parity' in self.name:
            returns = prices_df.iloc[-252:].values
            vols = np.std(returns, axis=0)
            if np.any(vols < 1e-8): return np.ones(N)/N
            w = 1.0 / (vols + 1e-8)
            return w / np.sum(w)
            
        elif 'Factor Momentum' in self.name:
            moms = prices_df.iloc[-60:].mean().values
            signal = (moms > 0).astype(float)
            if signal.sum() == 0: return np.ones(N)/N 
            return signal / signal.sum()

# ==========================================
# 5. 回测主引擎
# ==========================================
def run_backtest():
    print("⚔️ 开启回测竞技场 (Final Fix) ...")
    
    # 1. 准备数据
    _, test_loader, scaler = load_and_process_data()
    df = pd.read_csv(cfg.DATA_PATH, index_col=0, parse_dates=True)
    split_date = pd.Timestamp(cfg.TRAIN_SPLIT_DATE)
    test_returns_df = df.loc[df.index >= split_date, cfg.ASSETS]
    
    # 2. 初始化策略
    strategies = [
        DeepStrategy('Ours (Diff-MPO)', 'models/diff_mpo_sharpe.pth', mode='mpo'),
        DeepStrategy('Two-Stage (MSE)', 'models/baseline_mse_model.pth', mode='two_stage'),
        DeepStrategy('Neural Risk Parity', 'models/baseline_vol_model.pth', mode='nrp'),
        
        OptimizationStrategy('Mean-Variance (Robust)', lookback=126, lambda_cost=0.005),
        OptimizationStrategy('Global Min Var (Robust)', lookback=126, lambda_cost=0.005),
        OptimizationStrategy('Mean-CVaR (Robust)', lookback=126, lambda_cost=0.005),
        
        RuleStrategy('1/N'),
        RuleStrategy('Vanilla Risk Parity (1Y)'), 
    ]
    
    results = {s.name: {'wealth': [cfg.INIT_WEALTH], 'turnover': []} for s in strategies}
    current_weights = {s.name: np.ones(cfg.NUM_ASSETS)/cfg.NUM_ASSETS for s in strategies}
    
    print(f"   测试集长度: {len(test_returns_df)} 天")
    print(f"   交易成本 (单边): {cfg.COST_COEFF}")
    
    full_w_plans = {s.name: [] for s in strategies}
    
    # --- 生成决策序列 ---
    print("   正在生成决策序列...")
    for batch_idx, (x_batch, _) in enumerate(tqdm(test_loader)):
        for i in range(x_batch.size(0)):
            global_idx = batch_idx * cfg.BATCH_SIZE + i
            if global_idx >= len(test_returns_df) - 1: break
            
            current_date_idx = global_idx
            current_date = test_returns_df.index[current_date_idx]
            
            x_sample = x_batch[i].unsqueeze(0)
            
            for strat in strategies:
                w_curr = current_weights[strat.name]
                
                try:
                    if isinstance(strat, DeepStrategy):
                        w_target = strat.get_weights(None, w_curr, context_data=(x_sample, None))
                    else:
                        history_slice = df.loc[:current_date, cfg.ASSETS].iloc[:-1]
                        if len(history_slice) < 2:
                            w_target = w_curr
                        else:
                            w_target = strat.get_weights(history_slice, w_curr)
                except Exception as e:
                    # print(f"Err: {strat.name} - {e}")
                    w_target = w_curr

                # 强制转换
                try:
                    w_target = np.array(w_target, dtype=np.float64).reshape(-1)
                except:
                    w_target = w_curr # Fallback if None

                if w_target.shape[0] != cfg.NUM_ASSETS:
                    w_target = w_curr
                
                if w_target.sum() > 1e-6:
                    w_target = w_target / w_target.sum()
                else:
                    w_target = np.ones(cfg.NUM_ASSETS) / cfg.NUM_ASSETS

                full_w_plans[strat.name].append(w_target)
                current_weights[strat.name] = w_target
    
    # --- 计算净值 ---
    print("   正在计算净值与归因...")
    n_days = len(full_w_plans['1/N'])
    
    realized_ret = test_returns_df.iloc[:n_days].values
    full_dates = test_returns_df.index[:n_days]
    plot_dates = full_dates[1:] 
    
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
        net_ret = gross_ret - cost
        
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
    
    # --- 绘图 ---
    plt.figure(figsize=(14, 8))
    for strat_name in results:
        wealth = results[strat_name]['wealth']
        if 'Ours' in strat_name:
            lw, alpha, zorder = 2.5, 1.0, 10
        elif '1/N' in strat_name:
            lw, alpha, zorder = 2.0, 0.8, 5
        elif 'Robust' in strat_name: 
            lw, alpha, zorder = 1.5, 0.7, 4
        else:
            lw, alpha, zorder = 1.0, 0.5, 1
        plt.plot(plot_dates, wealth, label=strat_name, linewidth=lw, alpha=alpha, zorder=zorder)
        
    plt.title('Cumulative Wealth: Diff-MPO vs Steel-Manned Benchmarks', fontsize=14)
    plt.ylabel('Wealth (Start=1.0)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.gcf().autofmt_xdate()
    plt.legend(loc='upper left', fontsize=10, frameon=True, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('backtest_wealth_curve.png', dpi=300)
    print(f"\n📈 净值曲线已保存至 backtest_wealth_curve.png")

if __name__ == "__main__":
    run_backtest()