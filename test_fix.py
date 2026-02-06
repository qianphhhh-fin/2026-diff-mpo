
import torch
from torch.utils.data import DataLoader
from strategy import DeepMPOStrategy
from config import cfg

def test_strategy_fix():
    print("🧪 Testing DeepMPOStrategy fixes...")
    
    # Mock Data
    B, H, N, F_dim = 4, cfg.PREDICT_HORIZON, cfg.NUM_ASSETS, cfg.INPUT_FEATURE_DIM
    
    # Create fake dataset
    # X: (B, Lookback, F)
    x_batch = torch.randn(B, cfg.LOOKBACK_WINDOW, F_dim)
    # Y: (B, H, N)
    y_batch = torch.randn(B, H, N) * 0.01
    
    # Mock Loader
    loader = [(x_batch, y_batch)]
    
    # Init Strategy
    strat = DeepMPOStrategy("Test_DiffMPO")
    
    try:
        # Trigger training
        print("   Running on_train_period...")
        strat.on_train_period(loader)
        print("✅ on_train_period completed without error.")
        
        # Trigger inference
        print("   Running get_weights...")
        # Mock feature tensor for inference: (1, Lookback, F)
        feat = torch.randn(1, cfg.LOOKBACK_WINDOW, F_dim)
        w = strat.get_weights(history_df=None, feature_tensor=feat)
        
        print(f"✅ get_weights output: {w.shape}, Sum: {w.sum():.4f}")
        assert abs(w.sum() - 1.0) < 1e-3, "Weights sum should be close to 1"
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_strategy_fix()
