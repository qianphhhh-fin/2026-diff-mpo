import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
import os
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

# 引入你的模块
from config import cfg
from data_loader import MPODataset
from model import MPO_Network
from train_diff_mpo import calc_composite_loss 

# 设置风格
plt.style.use('seaborn-v0_8')
device = cfg.DEVICE

def run_walk_forward_experiment():
    print("⚔️ [Walk-Forward Experiment] Diff-MPO vs 1/N 滚动对决开始...")
    
    # ==========================
    # 1. 准备全量数据
    # ==========================
    df_raw = pd.read_csv(cfg.DATA_PATH, index_col=0, parse_dates=True)
    
    # 提取特征和收益率
    # 注意：这里假设 df_raw 的列顺序和 data_loader 里的一致
    # 特征 = 全部列 (15列)
    all_features = df_raw.values 
    # 目标资产 = Config 里的 10 个资产
    all_returns = df_raw[cfg.ASSETS].values
    dates = df_raw.index
    
    # 设置回测时间轴
    # 建议：从 2018 年开始回测，意味着第一次训练用的是 1990-2017 的数据
    TEST_START_YEAR = 2018 
    TEST_END_YEAR = dates[-1].year
    
    # 初始化记录器
    results_dmpo = [] # [(date, return)]
    results_ew = []   # [(date, return)]
    
    # 初始化模型 (Warm Start)
    # 我们创建一个模型实例，每年在此基础上微调 (Fine-tune)，模拟基金经理的持续学习
    model = MPO_Network().to(device).double()
    
    print(f"   数据范围: {dates[0].date()} -> {dates[-1].date()}")
    print(f"   回测区间: {TEST_START_YEAR} -> {TEST_END_YEAR}")
    print(f"   特征维度: {all_features.shape[1]}, 资产数: {cfg.NUM_ASSETS}")
    
    # ==========================
    # 2. 滚动循环 (Year by Year)
    # ==========================
    for year in range(TEST_START_YEAR, TEST_END_YEAR + 1):
        print(f"\n📅 正在处理年份: {year} ...")
        
        # --- A. 时间切分 ---
        # 训练集: 直到去年末 (Expanding)
        train_end_dt = pd.Timestamp(f"{year}-01-01")
        # 测试集: 今年整年
        test_end_dt = pd.Timestamp(f"{year+1}-01-01")
        
        train_mask = dates < train_end_dt
        test_mask = (dates >= train_end_dt) & (dates < test_end_dt)
        
        if sum(test_mask) < cfg.LOOKBACK_WINDOW:
            print(f"   ⚠️ 数据不足，跳过 {year}")
            continue
            
        # --- B. 数据防泄漏处理 (Scaler) ---
        scaler = StandardScaler()
        # 严禁：使用全量数据 fit
        # 必须：只用截至去年的数据 fit
        X_train = scaler.fit_transform(all_features[train_mask])
        Y_train = all_returns[train_mask]
        
        X_test = scaler.transform(all_features[test_mask])
        Y_test = all_returns[test_mask] # 这里的 Y_test 是这一年的真实收益率
        test_dates_curr = dates[test_mask]
        
        # 构建 DataLoader (Train)
        train_ds = MPODataset(X_train, Y_train, cfg.LOOKBACK_WINDOW, cfg.PREDICT_HORIZON)
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)
        
        # --- C. 模型重训练 (Retrain/Fine-tune) ---
        # 每年仅需少量 Epochs 适应新风格 (例如 5-10 Epochs)
        # 学习率可以稍微小一点，防止灾难性遗忘
        optimizer = optim.Adam(model.parameters(), lr=5e-4) 
        model.train()
        
        train_pbar = tqdm(range(10), desc=f"   Training {year}", leave=False)
        for ep in train_pbar:
            ep_loss = 0
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(device).double(), y_b.to(device).double()
                
                # 假设 w_prev 每天重置为 1/N (或者你可以维护真实的 w_prev)
                w_prev_b = torch.ones(x_b.size(0), cfg.NUM_ASSETS, device=device, dtype=torch.double) / cfg.NUM_ASSETS
                
                w_plan, _, _ = model(x_b, w_prev_b)
                
                # 使用你之前改好的 Composite Loss
                loss, _ = calc_composite_loss(w_plan, y_b, w_prev_b, cost_coeff=cfg.COST_COEFF)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ep_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{ep_loss/len(train_loader):.4f}"})
            
        # --- D. 样本外预测 (Out-of-Sample Inference) ---
        model.eval()
        
        # 这里的 Inference 需要逐日进行 (Rolling Inference)
        # 为了简单，我们构建一个 Test Dataset，它本质上就是滑动窗口
        # 注意：MPODataset 会吃掉前 Lookback 天，所以我们要补上一点数据
        # 让我们取 X_test 加上前 Lookback-1 天的数据，保证预测从 1月1日 开始
        
        # 找到测试集在原始数据中的起始索引
        test_start_idx = np.where(test_mask)[0][0]
        # 回溯 Lookback-1 天
        infer_start_idx = max(0, test_start_idx - cfg.LOOKBACK_WINDOW + 1)
        
        # 准备 Inference 数据
        X_infer_raw = all_features[infer_start_idx : test_start_idx + len(test_dates_curr)]
        X_infer_scaled = scaler.transform(X_infer_raw) # 用旧的 scaler 转换
        
        # 真实收益率 (用于计算每天的 PnL)
        Y_realized = Y_test 
        
        # 维护一个滚动的 current_w (初始为 1/N)
        curr_w = torch.ones(1, cfg.NUM_ASSETS, device=device, dtype=torch.double) / cfg.NUM_ASSETS
        
        with torch.no_grad():
            for t in range(len(Y_realized)):
                # 构造输入: [t : t+Lookback]
                x_window = X_infer_scaled[t : t + cfg.LOOKBACK_WINDOW]
                
                # 安全检查
                if len(x_window) != cfg.LOOKBACK_WINDOW: break
                
                x_tensor = torch.tensor(x_window).unsqueeze(0).to(device).double()
                
                # 预测
                w_pred, _, _ = model(x_tensor, curr_w)
                w_action = w_pred[0, 0, :] # 取第一步动作 (H=0)
                
                # --- 记录结果 ---
                w_np = w_action.cpu().numpy()
                y_today = Y_realized[t]
                
                # DMPO 收益
                # 扣费：Turnover * Cost
                # 假设 curr_w 是昨天的仓位
                w_prev_np = curr_w[0].cpu().numpy()
                turnover = np.sum(np.abs(w_np - w_prev_np))
                cost = turnover * cfg.COST_COEFF
                
                # 考虑现金仓位: sum(w) <= 1, 剩余是现金(收益0)
                # port_ret = w * y + (1-sum(w))*0
                gross_ret = np.sum(w_np * y_today)
                net_ret = gross_ret - cost
                
                results_dmpo.append(net_ret)
                
                # 1/N 收益 (Benchmark)
                w_ew = np.ones(cfg.NUM_ASSETS) / cfg.NUM_ASSETS
                ret_ew = np.sum(w_ew * y_today) # 1/N 不扣费或扣极少，这里简化不扣
                results_ew.append(ret_ew)
                
                # 更新状态
                curr_w = w_action.unsqueeze(0)
                
    # ==========================
    # 3. 结果汇总与评估
    # ==========================
    print("\n📊 计算最终指标...")
    
    # 转换为 Series
    # 只有被预测的日子才有结果
    total_days = len(results_dmpo)
    # 对应的日期是所有测试年份的并集
    # 这里简单处理，直接取最后的 total_days 个日期（可能会有极其微小的对齐误差，但做实验足够了）
    # 更严谨的做法是在 loop 里存 date
    idx = dates[-total_days:]
    
    s_dmpo = pd.Series(results_dmpo, index=idx)
    s_ew = pd.Series(results_ew, index=idx)
    
    # 净值曲线
    wealth_dmpo = (1 + s_dmpo).cumprod()
    wealth_ew = (1 + s_ew).cumprod()
    
    # 计算指标函数
    def calc_metrics(series, name):
        ann_ret = series.mean() * 252
        ann_vol = series.std() * np.sqrt(252)
        sharpe = (ann_ret - 0.02) / (ann_vol + 1e-6)
        
        downside = series[series<0]
        sortino = (ann_ret - 0.02) / (downside.std() * np.sqrt(252) + 1e-6)
        
        cum = (1+series).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        max_dd = dd.min()
        
        calmar = ann_ret / (abs(max_dd) + 1e-6)
        
        return {
            "Strategy": name,
            "Return": f"{ann_ret:.2%}",
            "Sharpe": f"{sharpe:.2f}",
            "Sortino": f"{sortino:.2f}",
            "Calmar": f"{calmar:.2f}",
            "MaxDD": f"{max_dd:.2%}"
        }
    
    m1 = calc_metrics(s_dmpo, "Diff-MPO (Walk-Forward)")
    m2 = calc_metrics(s_ew, "1/N Benchmark")
    
    res_df = pd.DataFrame([m1, m2])
    print("\n🏆 滚动回测最终结果:")
    print(res_df)
    
    # 画图
    plt.figure(figsize=(12, 6))
    plt.plot(wealth_dmpo, label='Diff-MPO (Walk-Forward)', linewidth=2)
    plt.plot(wealth_ew, label='1/N Benchmark', linestyle='--', alpha=0.7)
    plt.title(f'Walk-Forward Validation ({TEST_START_YEAR}-{TEST_END_YEAR})\nDiff-MPO Retrained Annually', fontsize=14)
    plt.ylabel('Cumulative Wealth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存
    save_path = 'walk_forward_result.png'
    plt.savefig(save_path, dpi=300)
    print(f"\n📈 图片已保存至: {save_path}")

if __name__ == "__main__":
    run_walk_forward_experiment()