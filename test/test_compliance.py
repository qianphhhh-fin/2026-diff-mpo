
import unittest
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from config import cfg
from strategy import DeepMPOStrategy
from train_diff_mpo import train
import inspect

class TestDiffMPOCompliance(unittest.TestCase):
    """
    Diff-MPO 规范一致性测试套件
    对应一致性审查报告中的 REQ 检查点
    """

    def test_01_config_completeness(self):
        """REQ-14, REQ-15, REQ-16, REQ-17: 检查关键配置项是否存在且值正确"""
        print("\n[Test] Checking Config Completeness...")
        
        # 风险参数
        self.assertTrue(hasattr(cfg, 'RISK_AVERSION'), "Missing RISK_AVERSION")
        self.assertEqual(cfg.RISK_AVERSION, 5.0, "RISK_AVERSION mismatch with Doc")
        
        # CVaR 参数
        self.assertTrue(hasattr(cfg, 'CVAR_LIMIT'), "Missing CVAR_LIMIT")
        self.assertEqual(cfg.CVAR_LIMIT, 0.05, "CVaR Limit mismatch with Doc")
        
        # 新增的统一控制参数
        self.assertTrue(hasattr(cfg, 'GRAD_CLIP'), "Missing GRAD_CLIP (Fix Check)")
        self.assertEqual(cfg.GRAD_CLIP, 0.5, "GRAD_CLIP default mismatch")
        
        self.assertTrue(hasattr(cfg, 'LR_FINE_TUNE_SCALE'), "Missing LR_FINE_TUNE_SCALE (Fix Check)")
        
    def test_02_strategy_lr_consistency(self):
        """REQ-16: 检查 Strategy 是否使用了 Config 中的 LR 配置"""
        print("\n[Test] Checking Strategy LR Consistency...")
        
        # 1. 实例化策略
        strat = DeepMPOStrategy("Test_Strat")
        
        # 2. 构造 Mock 数据
        # (Batch=2, Lookback=60, Feat=15)
        x_dummy = torch.randn(2, cfg.LOOKBACK_WINDOW, cfg.INPUT_FEATURE_DIM)
        # (Batch=2, Horizon=5, Assets=10)
        y_dummy = torch.randn(2, cfg.PREDICT_HORIZON, cfg.NUM_ASSETS)
        
        dataset = TensorDataset(x_dummy, y_dummy)
        loader = DataLoader(dataset, batch_size=2)
        
        # 3. Hook 优化器创建
        # 我们不能直接访问函数内部的 optimizer 变量，但我们可以通过 Monkey Patch 
        # torch.optim.Adam 来捕获传入的 lr
        
        captured_lr = None
        original_adam = optim.Adam
        
        def mock_adam(params, lr=None, **kwargs):
            nonlocal captured_lr
            captured_lr = lr
            return original_adam(params, lr=lr, **kwargs)
            
        optim.Adam = mock_adam
        
        try:
            # 运行一轮 (这就足够触发 optimizer 初始化)
            # 我们只需要它跑通初始化部分
            # 为了不让它真跑太久，我们可以 patch model 的 forward 或者让 loader 为空
            # 但这里数据很少，应该很快
            strat.on_train_period(loader)
        except Exception as e:
            # 即使报错也没关系，只要 optimizer 初始化了就行
            pass
        finally:
            # 还原
            optim.Adam = original_adam
            
        expected_lr = cfg.LEARNING_RATE * cfg.LR_FINE_TUNE_SCALE
        print(f"   Captured LR: {captured_lr}")
        print(f"   Expected LR: {expected_lr}")
        
        self.assertIsNotNone(captured_lr, "Optimizer was not initialized")
        self.assertAlmostEqual(captured_lr, expected_lr, places=6, 
                               msg="Strategy Fine-tune LR does not match Config calculation")

    def test_03_solver_constraints(self):
        """REQ-07, REQ-08: 验证 Solver 输出是否满足硬约束 (Sum=1, Positive)"""
        print("\n[Test] Checking Solver Constraints...")
        from mpo_solver import DifferentiableMPO_cvx
        
        solver = DifferentiableMPO_cvx().to(cfg.DEVICE)
        
        B, H, N = 2, cfg.PREDICT_HORIZON, cfg.NUM_ASSETS
        mu = torch.randn(B, H, N, device=cfg.DEVICE)
        L = torch.eye(N, device=cfg.DEVICE).view(1, 1, N, N).expand(B, H, N, N)
        w_prev = torch.ones(B, N, device=cfg.DEVICE) / N
        
        w_out = solver(mu, L, w_prev)
        
        # Check Shape
        self.assertEqual(w_out.shape, (B, H, N))
        
        # Check Sum = 1
        w_sum = w_out.sum(dim=-1)
        # float precision tolerance
        self.assertTrue(torch.allclose(w_sum, torch.ones_like(w_sum), atol=1e-5), "Budget Constraint Violated")
        
        # Check Positive
        self.assertTrue((w_out >= -1e-6).all(), "Long-only Constraint Violated")

if __name__ == '__main__':
    unittest.main()
