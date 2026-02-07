"""
脚本名称: eval_rolling_all.py
功能描述: 
    "Grand Challenge" 滚动回测主引擎。
    在真实的时间轴上模拟交易，对比 Diff-MPO 与其他基准策略的绩效。

主要流程:
    1. 初始化所有策略 (Diff-MPO, Mean-Var, 1/N 等)。
    2. 按年份进行滚动回测 (Walk-Forward Validation):
       - 每年初，使用过去的数据对 DeepMPO 进行微调 (Retraining)。
       - 每日进行推理，获取目标权重。
       - 模拟交易，计算每日收益和换手率。
    3. 绩效评估: 计算 Sharpe, Sortino, Calmar, MaxDD, Turnover 等指标。
    4. 可视化: 绘制净值曲线并保存结果。

输入:
    - 'mpo_experiment_data.csv' (原始数据)。
    - 策略定义 (strategy.py)。

输出:
    - 绩效指标表格 (控制台打印 & CSV 保存)。
    - 净值曲线图 'results/grand_challenge_wealth_curves.png'。
    - 每日收益和换手率序列 CSV。

与其他脚本的关系:
    - 项目的最终出口，整合了 data_loader, strategy, model, mpo_solver 等所有模块。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

# 引入自定义模块
from config import cfg
from data_loader import MPODataset
from strategy import (
    RuleBasedStrategy, 
    OptimizationStrategy, 
    DeepMPOStrategy,
    DirectGradientStrategy, # <--- 新增
    HRPStrategy,  # <--- 新增
    DeepE2EStrategy
)

# 设置绘图风格
plt.style.use('seaborn-v0_8')

def seed_everything(seed=42):
    """固定所有随机种子以保证结果可复现"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 保证 CuDNN 的确定性 (会牺牲一点速度)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_metrics(returns_series, turnover_series, name):
    """计算回测指标"""
    # 年化收益
    ann_ret = returns_series.mean() * 252
    
    # 年化波动
    ann_vol = returns_series.std() * np.sqrt(252)
    
    # Sharpe (无风险利率假定为 2%)
    rf = 0.02
    sharpe = (ann_ret - rf) / (ann_vol + 1e-6)
    
    # Sortino (只考虑下行波动)
    downside = returns_series.copy()
    downside[downside > 0] = 0
    downside_std = downside.std() * np.sqrt(252)
    sortino = (ann_ret - rf) / (downside_std + 1e-6)
    
    # Max Drawdown
    # wealth 是 numpy array
    wealth = (1 + returns_series).cumprod()
    
    # --- 修改点开始 ---
    # numpy 没有 .cummax()，要用 np.maximum.accumulate()
    cummax = np.maximum.accumulate(wealth) 
    # --- 修改点结束 ---
    
    drawdown = (wealth - cummax) / cummax
    max_dd = drawdown.min()
    
    # Calmar Ratio
    calmar = ann_ret / (abs(max_dd) + 1e-6)
    
    # Turnover
    avg_turnover = turnover_series.mean()
    
    return {
        "Strategy": name,
        "Return": f"{ann_ret:.2%}",
        "Sharpe": f"{sharpe:.2f}",
        "Sortino": f"{sortino:.2f}",
        "Calmar": f"{calmar:.2f}",
        "MaxDD": f"{max_dd:.2%}",
        "Turnover": f"{avg_turnover:.2%}"
    }

def run_comprehensive_backtest():
    # 1. 设置随机种子
    seed_everything(cfg.SEED)
    print(f"🔒 Random Seed set to {cfg.SEED}")

    print("⚔️ [Grand Challenge] 滚动回测竞技场启动 ...")
    print(f"   Device: {cfg.DEVICE}")
    print(f"   Transaction Cost: {cfg.COST_COEFF * 10000:.0f} bps")
    
    # ==========================================
    # 1. 初始化策略池
    # ==========================================
    # 这里我们实例化所有想要对比的策略
    strategies = [
        # --- 基准 (Benchmarks) ---
        RuleBasedStrategy('1/N Benchmark'),
        RuleBasedStrategy('Risk Parity (1Y)'),
        RuleBasedStrategy('Factor Momentum (Top3)'),
        
        # --- 传统优化 (Classic Optimization) ---
        # 注意：Mean-CVaR 计算较慢，如果你想快速跑通可以先注释掉
        OptimizationStrategy('Mean-Variance', lookback=60),
        OptimizationStrategy('Global Min Var', lookback=60),
        OptimizationStrategy('Mean-CVaR', lookback=60), 
        HRPStrategy('Hierarchical Risk Parity', lookback=252),
           
        # --- 深度学习 (Ours) ---
        DeepMPOStrategy('Diff-MPO (Factor Model)'),
        # 新增：直接在历史上优化 Loss 的策略
        DirectGradientStrategy('Direct Loss Opt (History)', lookback=60),
        DeepE2EStrategy('Deep E2E (Policy Net)'), # <--- 你的新对手
    ]
    
    # 初始化记录器
    # results[strat_name] = [daily_net_returns]
    results = {s.name: [] for s in strategies}
    turnovers = {s.name: [] for s in strategies}
    
    # 维护当前的持仓权重 (用于计算换手率和作为 w_prev)
    # 初始全部为 1/N
    current_weights = {
        s.name: np.ones(cfg.NUM_ASSETS) / cfg.NUM_ASSETS 
        for s in strategies
    }
    
    # ==========================================
    # 2. 数据准备
    # ==========================================
    df_raw = pd.read_csv(cfg.DATA_PATH, index_col=0, parse_dates=True)
    all_features = df_raw.values 
    all_returns = df_raw[cfg.ASSETS].values
    dates = df_raw.index
    
    # 设置回测区间
    TEST_START_YEAR = 2018
    TEST_END_YEAR = dates[-1].year
    
    print(f"   回测区间: {TEST_START_YEAR} -> {TEST_END_YEAR}")
    
    # ==========================================
    # 3. 年度滚动循环 (Walk-Forward Loop)
    # ==========================================
    for year in range(TEST_START_YEAR, TEST_END_YEAR + 1):
        print(f"\n📅 处理年份: {year} ...")
        
        # --- A. 时间切分 (Expanding Window) ---
        train_end_dt = pd.Timestamp(f"{year}-01-01")
        test_end_dt = pd.Timestamp(f"{year+1}-01-01")
        
        train_mask = dates < train_end_dt
        test_mask = (dates >= train_end_dt) & (dates < test_end_dt)
        
        # 确保数据足够
        if sum(test_mask) < cfg.LOOKBACK_WINDOW:
            print(f"   ⚠️ {year} 年数据不足，跳过。")
            continue
            
        # --- B. 训练集准备与标准化 (防泄露核心) ---
        scaler = StandardScaler()
        # Fit 仅在训练集上进行！
        X_train = scaler.fit_transform(all_features[train_mask])
        Y_train = all_returns[train_mask]
        
        # --- C. 策略重训 (Retraining Hook) ---
        # 对于 DeepMPO，这会触发 Fine-tuning
        # 对于传统策略，通常只是 Pass
        
        # 预先构建 DataLoader 供深度模型使用
        train_ds = MPODataset(X_train, Y_train, cfg.LOOKBACK_WINDOW, cfg.PREDICT_HORIZON)
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)
        
        for strat in strategies:
            # 只有 DeepMPOStrategy 实现了具体的 on_train_period
            # 我们不需要在这里判断类型，多态性会处理它
            if isinstance(strat, DeepMPOStrategy):
                print(f"   🔄 Retraining {strat.name}...")
            strat.on_train_period(train_loader)
            
        # --- D. 逐日推理 (Daily Rolling Inference) ---
        # 找到测试集在全量数据中的起始位置
        test_start_idx = np.where(test_mask)[0][0]
        # 当年所有的交易日
        test_indices = np.where(test_mask)[0]
        
        # 使用 tqdm 显示每日进度
        pbar = tqdm(test_indices, desc=f"   Trading {year}", leave=False)
        
        for t_abs in pbar:
            # t_abs 是绝对索引 (Absolute Index)
            # 目标：决定 t_abs 这一天的持仓，享受 t_abs 当天的收益 Y[t_abs]
            # 约束：只能看到 t_abs - 1 及以前的数据
            
            # 1. 准备历史输入窗口
            # 原始特征窗口 [t - Lookback : t] (不包含 t)
            # 比如 t=100, lookback=60 -> 取 [40:100], 即 indices 40...99
            x_window_raw = all_features[t_abs - cfg.LOOKBACK_WINDOW : t_abs]
            
            # 历史收益窗口 (用于传统优化器计算 Covariance)
            # DataFrame 切片是包含结尾的，所以用 iloc 需要小心
            # iloc[start : end] 不包含 end。
            # 我们需要 0 到 t-1 的数据。
            # 为了方便 OptimizationStrategy，我们传入 DataFrame
            history_df_slice = df_raw[cfg.ASSETS].iloc[:t_abs] 
            
            # 安全检查
            if len(x_window_raw) != cfg.LOOKBACK_WINDOW:
                continue
                
            # 2. 特征标准化 (使用当年的 scaler)
            x_window_scaled = scaler.transform(x_window_raw)
            # 转为 Tensor (Batch=1, Lookback, Features)
            feature_tensor = torch.tensor(x_window_scaled).unsqueeze(0)
            
            # 当天的真实收益 (用于结算)
            y_today = all_returns[t_abs]
            
            # 3. 遍历所有策略获取决策
            for strat in strategies:
                w_prev = current_weights[strat.name]
                
                try:
                    # === 核心调用 ===
                    # 多态调用：不同策略会使用不同的输入参数
                    # 传统策略忽略 feature_tensor，深度策略忽略 history_df
                    w_target = strat.get_weights(
                        history_df=history_df_slice, 
                        feature_tensor=feature_tensor
                    )
                except Exception as e:
                    # 如果策略崩溃 (极少见)，保持仓位不变或空仓
                    # print(f"Error in {strat.name}: {e}")
                    w_target = w_prev
                
                # 4. 结算 PnL
                # 计算换手
                turnover = np.sum(np.abs(w_target - w_prev))
                cost = turnover * cfg.COST_COEFF
                
                # 计算收益 (假设满仓或部分仓位)
                # 收益 = 股票收益 + 现金收益(0) - 交易成本
                gross_ret = np.sum(w_target * y_today)
                net_ret = gross_ret - cost
                
                # 记录
                results[strat.name].append(net_ret)
                turnovers[strat.name].append(turnover)
                
                # 更新持仓
                current_weights[strat.name] = w_target

    # ==========================================
    # 4. 结果汇总与可视化
    # ==========================================
    print("\n📊 计算最终指标排行榜...")
    
    metrics_list = []
    equity_curves = {}
    
    # 获取日期索引 (用于绘图)
    # 注意：results 列表可能比 dates 少一点点（因为开头 lookback 跳过）
    # 我们取最后 N 个日期对齐
    n_days = len(results['1/N Benchmark'])
    plot_dates = dates[-n_days:]
    
    for strat in strategies:
        ret_seq = np.array(results[strat.name])
        to_seq = np.array(turnovers[strat.name])
        
        # 计算指标
        m = calculate_metrics(ret_seq, to_seq, strat.name)
        metrics_list.append(m)
        
        # 计算净值曲线
        equity = (1 + ret_seq).cumprod()
        equity_curves[strat.name] = pd.Series(equity, index=plot_dates)
        
    # 生成 DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    # 按 Sortino 排序
    metrics_df = metrics_df.sort_values("Sortino", ascending=False)
    
    print("\n🏆 全策略回测结果 (2018 - End):")
    print(metrics_df.to_string(index=False))
    
    # 保存 CSV
    metrics_df.to_csv("rolling_backtest_metrics.csv", index=False)
    metrics_df.to_csv("results/rolling_backtest_metrics.csv", index=False)
    
    # --- 保存原始序列 (New) ---
    print("\n💾 保存原始序列数据...")
    try:
        # 构造 DataFrame (使用对齐后的日期索引)
        returns_df = pd.DataFrame(results, index=plot_dates)
        turnovers_df = pd.DataFrame(turnovers, index=plot_dates)
        
        returns_df.to_csv("results/backtest_daily_returns.csv")
        turnovers_df.to_csv("results/backtest_daily_turnovers.csv")
        print("   -> results/backtest_daily_returns.csv (日收益率)")
        print("   -> results/backtest_daily_turnovers.csv (日换手率)")
    except Exception as e:
        print(f"⚠️ 保存原始序列失败: {e}")

    # --- 绘图 ---
    plt.figure(figsize=(14, 8))
    
    # 定义颜色和线型，突出显示 Diff-MPO
    for strat_name, curve in equity_curves.items():
        if "Diff-MPO" in strat_name:
            plt.plot(curve, label=strat_name, linewidth=2.5, color='#d62728', alpha=1.0) # 红色加粗
        elif "1/N" in strat_name:
            plt.plot(curve, label=strat_name, linewidth=2.0, color='black', linestyle='--', alpha=0.7)
        else:
            plt.plot(curve, label=strat_name, linewidth=1.0, alpha=0.5)
            
    plt.title('Grand Challenge: Diff-MPO vs Traditional Strategies (Walk-Forward)', fontsize=16)
    plt.ylabel('Cumulative Wealth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = "results/grand_challenge_wealth_curves.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n📈 净值曲线已保存至: {save_path}")

if __name__ == "__main__":
    run_comprehensive_backtest()