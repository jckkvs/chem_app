"""
ハイパーパラメータチューナーテスト
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression

from core.services.ml.hyperparameter_tuner import HyperparameterTuner, TuningResult


class TestHyperparameterTuner:
    """ハイパーパラメータチューナーテスト"""
    
    @pytest.fixture
    def sample_data(self):
        """サンプルデータ作成"""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'f{i}' for i in range(10)])
        y_series = pd.Series(y, name='target')
        return X_df, y_series
    
    def test_random_forest_tuning(self, sample_data):
        """Random Forestチューニング"""
        X, y = sample_data
        tuner = HyperparameterTuner()
        
        result = tuner.tune(
            model_type='random_forest',
            X=X, y=y,
            method='random',
            n_iter=5,  # 高速化のため少なめ
            cv=3
        )
        
        assert isinstance(result, TuningResult)
        assert 'n_estimators' in result.best_params
        assert 'max_depth' in result.best_params
        assert result.best_score > 0
        assert result.total_trials == 5
        assert len(result.tuning_history) == 5
    
    def test_lightgbm_tuning(self, sample_data):
        """LightGBMチューニング"""
        X, y = sample_data
        tuner = HyperparameterTuner()
        
        result = tuner.tune(
            model_type='lightgbm',
            X=X, y=y,
            method='random',
            n_iter=5,
            cv=3
        )
        
        assert 'learning_rate' in result.best_params
        assert 'num_leaves' in result.best_params
        assert result.best_score > 0
    
    def test_grid_search(self, sample_data):
        """GridSearchテスト"""
        X, y = sample_data
        tuner = HyperparameterTuner()
        
        # 小さい探索空間でテスト
        custom_space = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10]
        }
        
        result = tuner.tune(
            model_type='random_forest',
            X=X, y=y,
            method='grid',
            custom_search_space=custom_space,
            cv=3
        )
        
        # 2 x 2 = 4通り
        assert result.total_trials == 4
        assert result.best_params['n_estimators'] in [50, 100]
        assert result.best_params['max_depth'] in [5, 10]
    
    def test_custom_search_space(self, sample_data):
        """カスタム探索空間"""
        X, y = sample_data
        tuner = HyperparameterTuner()
        
        custom_space = {
            'n_estimators': [100, 200],
            'max_depth': [10]
        }
        
        result = tuner.tune(
            model_type='random_forest',
            X=X, y=y,
            method='random',
            n_iter=3,
            custom_search_space=custom_space,
            cv=3
        )
        
        assert result.best_params['max_depth'] == 10
        assert result.best_params['n_estimators'] in [100, 200]
    
    def test_unsupported_model_type(self, sample_data):
        """サポートされていないモデル"""
        X, y = sample_data
        tuner = HyperparameterTuner()
        
        with pytest.raises(ValueError, match="not supported"):
            tuner.tune(
                model_type='unsupported_model',
                X=X, y=y
            )
    
    def test_preset_config(self):
        """プリセット設定"""
        tuner = HyperparameterTuner()
        
        fast = tuner.get_preset_config('fast')
        assert fast['method'] == 'random'
        assert fast['n_iter'] == 20
        assert fast['cv'] == 3
        
        balanced = tuner.get_preset_config('balanced')
        assert balanced['n_iter'] == 50
        assert balanced['cv'] == 5
        
        thorough = tuner.get_preset_config('thorough')
        assert thorough['method'] == 'grid'
    
    def test_invalid_preset(self):
        """無効なプリセット"""
        tuner = HyperparameterTuner()
        
        with pytest.raises(ValueError, match="Unknown level"):
            tuner.get_preset_config('invalid')
    
    def test_tuning_history_sorted(self, sample_data):
        """履歴がスコア順にソートされている"""
        X, y = sample_data
        tuner = HyperparameterTuner()
        
        result = tuner.tune(
            model_type='random_forest',
            X=X, y=y,
            method='random',
            n_iter=10,
            cv=3
        )
        
        # スコアが降順
        scores = [h['score'] for h in result.tuning_history]
        assert scores == sorted(scores, reverse=True)
    
    def test_improvement_calculation(self, sample_data):
        """改善率計算"""
        X, y = sample_data
        tuner = HyperparameterTuner()
        
        result = tuner.tune(
            model_type='random_forest',
            X=X, y=y,
            method='random',
            n_iter=5,
            cv=3
        )
        
        # 改善率が計算されている
        assert isinstance(result.improvement, float)
        # ランダムデータなので、時々デフォルトより悪くなることもある
        assert result.improvement >= -20  # 最悪でも-20%程度
