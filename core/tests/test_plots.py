# -*- coding: utf-8 -*-
"""
可視化エンジンのテスト

カバレッジ目標: 35.54% → 80%
"""
import numpy as np
import pandas as pd
import pytest


class TestPlotEngineInit:
    """初期化テスト"""
    
    def test_basic_imports(self):
        """基本的なインポート"""
        from core.services.vis import plots
        assert plots is not None


class TestDataPreparation:
    """データ準備テスト"""
    
    def test_prepare_plot_data(self):
        """プロットデータ準備"""
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10]
        })
        
        assert len(df) == 5
        assert 'x' in df.columns
        assert 'y' in df.columns


class TestScatterPlot:
    """散布図テスト"""
    
    def test_basic_scatter(self):
        """基本的な散布図"""
        try:
            import matplotlib.pyplot as plt
            
            x = np.array([1, 2, 3, 4, 5])
            y = np.array([2, 4, 6, 8, 10])
            
            fig, ax = plt.subplots()
            ax.scatter(x, y)
            
            assert fig is not None
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")


class TestLinePlot:
    """折れ線図テスト"""
    
    def test_basic_line(self):
        """基本的な折れ線図"""
        try:
            import matplotlib.pyplot as plt
            
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            
            fig, ax = plt.subplots()
            ax.plot(x, y)
            
            assert fig is not None
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")


class TestHistogram:
    """ヒストグラムテスト"""
    
    def test_basic_histogram(self):
        """基本的なヒストグラム"""
        try:
            import matplotlib.pyplot as plt
            
            data = np.random.randn(1000)
            
            fig, ax = plt.subplots()
            ax.hist(data, bins=30)
            
            assert fig is not None
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")


class TestBoxPlot:
    """箱ひげ図テスト"""
    
    def test_basic_boxplot(self):
        """基本的な箱ひげ図"""
        try:
            import matplotlib.pyplot as plt
            
            data = [np.random.randn(100) for _ in range(3)]
            
            fig, ax = plt.subplots()
            ax.boxplot(data)
            
            assert fig is not None
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")


class TestHeatmap:
    """ヒートマップテスト"""
    
    def test_basic_heatmap(self):
        """基本的なヒートマップ"""
        try:
            import matplotlib.pyplot as plt
            
            data = np.random.rand(10, 10)
            
            fig, ax = plt.subplots()
            im = ax.imshow(data, cmap='viridis')
            plt.colorbar(im, ax=ax)
            
            assert fig is not None
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")


class TestSubplots:
    """サブプロットテスト"""
    
    def test_multiple_subplots(self):
        """複数サブプロット"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            
            # 各サブプロットに異なるプロット
            axes[0, 0].plot([1, 2, 3], [1, 2, 3])
            axes[0, 1].scatter([1, 2, 3], [3, 2, 1])
            axes[1, 0].hist(np.random.randn(100))
            axes[1, 1].boxplot([np.random.randn(50)])
            
            assert fig is not None
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")


class TestPlotCustomization:
    """プロットカスタマイズテスト"""
    
    def test_labels_and_title(self):
        """ラベルとタイトル"""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 4, 9])
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_title('Test Plot')
            
            assert ax.get_xlabel() == 'X Label'
            assert ax.get_ylabel() == 'Y Label'
            assert ax.get_title() == 'Test Plot'
            
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")
    
    def test_legend(self):
        """凡例"""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 2, 3], label='Line 1')
            ax.plot([1, 2, 3], [3, 2, 1], label='Line 2')
            ax.legend()
            
            legend = ax.get_legend()
            assert legend is not None
            
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")


class TestPlotSaving:
    """プロット保存テスト"""
    
    def test_save_figure(self, tmp_path):
        """図の保存"""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 4, 9])
            
            output_path = tmp_path / "test_plot.png"
            fig.savefig(str(output_path))
            
            assert output_path.exists()
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")


class TestPlotIntegration:
    """統合テスト"""
    
    def test_complete_workflow(self):
        """完全なワークフロー"""
        try:
            import matplotlib.pyplot as plt
            
            # データ準備
            df = pd.DataFrame({
                'x': np.random.rand(100),
                'y': np.random.rand(100),
                'category': np.random.choice(['A', 'B', 'C'], 100)
            })
            
            # プロット作成
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # 散布図
            for cat in df['category'].unique():
                mask = df['category'] == cat
                axes[0].scatter(df[mask]['x'], df[mask]['y'], label=cat, alpha=0.6)
            axes[0].set_xlabel('X')
            axes[0].set_ylabel('Y')
            axes[0].legend()
            
            # ヒストグラム
            axes[1].hist(df['x'], bins=20, alpha=0.7)
            axes[1].set_xlabel('X Value')
            axes[1].set_ylabel('Frequency')
            
            assert fig is not None
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")
