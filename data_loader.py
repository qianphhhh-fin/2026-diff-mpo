import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from config import cfg  # 导入配置

class MPODataset(Dataset):
    """
    针对多期优化的数据集类。
    输入 X: 过去 T 天的所有特征 (宏观 + 因子 + 技术面)
    输出 Y: 未来 H 天的【资产真实收益率】 (用于计算 Loss: Sharpe Ratio)
    """
    def __init__(self, features, returns, lookback=60, horizon=5):
        """
        features: (N, F) 归一化后的特征矩阵
        returns:  (N, A) 原始资产收益率矩阵 (不归一化！算钱必须用真值)
        """
        self.features = torch.FloatTensor(features)
        self.returns = torch.FloatTensor(returns)
        self.lookback = lookback
        self.horizon = horizon
        
        # 有效样本数 = 总长度 - 回看窗口 - 预测窗口
        self.length = len(features) - lookback - horizon + 1

    def __len__(self):
        return max(0, self.length)

    def __getitem__(self, idx):
        # 1. 输入 X: 从 idx 到 idx+lookback (过去T天)
        # 形状: (T, Num_Features)
        x_window = self.features[idx : idx + self.lookback]
        
        # 2. 标签 Y: 从 idx+lookback 到 idx+lookback+horizon (未来H天)
        # 形状: (H, Num_Assets)
        # 注意：这是 Solver 优化完之后，用来“对答案”的真实未来收益
        y_horizon = self.returns[idx + self.lookback : idx + self.lookback + self.horizon]
        
        return x_window, y_horizon

def load_and_process_data():
    """
    主函数：读取CSV -> 清洗 -> 拆分 -> 归一化 -> 构建DataLoader
    """
    print(f"🔄 正在加载数据: {cfg.DATA_PATH} ...")
    df = pd.read_csv(cfg.DATA_PATH, index_col=0, parse_dates=True)
    
    # 1. 拆分特征与目标
    # Target: 只有那5个我们要交易的资产
    asset_returns = df[cfg.ASSETS].values
    
    # Features: 包含宏观、技术面、以及资产自身的滞后收益
    # (在 fetch_data 中我们已经把所有列都拼在了一起，这里直接用全部列作为特征)
    # 注意：通常为了防止过拟合，可以只选部分列，这里先全用
    feature_data = df.values
    
    # 2. 划分训练集/测试集 (按时间切分，严禁 Shuffle!)
    split_date = pd.Timestamp(cfg.TRAIN_SPLIT_DATE)
    train_mask = df.index < split_date
    test_mask = df.index >= split_date
    
    print(f"   训练集截止: {cfg.TRAIN_SPLIT_DATE} (样本数: {sum(train_mask)})")
    print(f"   测试集开始: {cfg.TRAIN_SPLIT_DATE} (样本数: {sum(test_mask)})")
    
    X_train_raw = feature_data[train_mask]
    X_test_raw = feature_data[test_mask]
    
    # Y 不需要归一化，因为由于我们要算真实的夏普比率
    Y_train = asset_returns[train_mask]
    Y_test = asset_returns[test_mask]
    
    # 3. 归一化 (Z-Score)
    # 关键：Scaler 只能在训练集上 fit，然后 transform 到测试集！
    # 否则就是严重的 Look-ahead Bias (数据泄露)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw) # 用训练集的均值方差处理测试集
    
    print("✅ 数据归一化完成 (StandardScaler)")
    
    # 4. 构建 Dataset
    train_dataset = MPODataset(
        X_train_scaled, Y_train, 
        lookback=cfg.LOOKBACK_WINDOW, 
        horizon=cfg.PREDICT_HORIZON
    )
    
    test_dataset = MPODataset(
        X_test_scaled, Y_test, 
        lookback=cfg.LOOKBACK_WINDOW, 
        horizon=cfg.PREDICT_HORIZON
    )
    
    # 5. 构建 DataLoader
    # 训练集可以 shuffle，增加泛化能力
    # 测试集不要 shuffle，方便画出连续的资金曲线
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, drop_last=True)
    
    return train_loader, test_loader, scaler

# ==========================
# 单元测试 (Run this file directly)
# ==========================
if __name__ == "__main__":
    print("🧪 开始运行数据加载测试...")
    try:
        train_loader, test_loader, _ = load_and_process_data()
        
        # 取出一个 Batch 看看形状
        x_batch, y_batch = next(iter(train_loader))
        
        print(f"\n[测试通过] Batch Shapes:")
        print(f"   X (Input Features): {x_batch.shape}")
        print(f"     -> (Batch={cfg.BATCH_SIZE}, Lookback={cfg.LOOKBACK_WINDOW}, Feats={x_batch.shape[2]})")
        print(f"   Y (Future Returns): {y_batch.shape}")
        print(f"     -> (Batch={cfg.BATCH_SIZE}, Horizon={cfg.PREDICT_HORIZON}, Assets={cfg.NUM_ASSETS})")
        
        print("\n🚀 data_loader 模块工作正常！请继续下一步。")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()