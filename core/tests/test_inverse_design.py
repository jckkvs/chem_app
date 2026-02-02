"""
逆解析テスト
"""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from core.services.ml.inverse_design import InverseDesign, Candidate


class MockModel:
    """テスト用モックモデル"""
    
    def predict(self, X):
        """簡単な予測（SMILES長に比例）"""
        if isinstance(X, list):
            return np.array([len(smiles) * 0.1 for smiles in X])
        return np.array([1.0] * len(X))


class TestInverseDesign:
    """逆解析テスト"""
    
    @pytest.fixture
    def mock_model(self):
        """モックモデル"""
        return MockModel()
    
    @pytest.fixture
    def designer(self, mock_model):
        """デザイナー"""
        return InverseDesign(mock_model, property_name='test_property')
    
    def test_exhaustive_search(self, designer):
        """全探索テスト"""
        library = [
            'CCO',  # エタノール
            'c1ccccc1',  # ベンゼン
            'CC(=O)O',  # 酢酸
            'c1ccc(O)cc1',  # フェノール
        ]
        
        candidates = designer.optimize(
            target_value=1.0,
            direction='maximize',
            method='exhaustive',
            library=library
        )
        
        assert len(candidates) <= len(library)
        assert all(isinstance(c, Candidate) for c in candidates)
        assert candidates[0].rank == 1
        # スコア順にソートされている
        scores = [c.score for c in candidates]
        assert scores == sorted(scores, reverse=True)
    
    def test_random_sampling(self, designer):
        """ランダムサンプリングテスト"""
        candidates = designer.optimize(
            target_value=1.0,
            direction='maximize',
            method='random',
            n_iterations=20
        )
        
        assert len(candidates) > 0
        assert all(isinstance(c, Candidate) for c in candidates)
        # スコア順にソートされている
        scores = [c.score for c in candidates]
        assert scores == sorted(scores, reverse=True)
    
    def test_bayesian_optimization(self, designer):
        """ベイズ最適化テスト"""
        candidates = designer.optimize(
            target_value=1.0,
            direction='maximize',
            method='bayesian',
            n_iterations=20
        )
        
        assert len(candidates) > 0
        assert all(isinstance(c, Candidate) for c in candidates)
    
    def test_constraints(self, designer):
        """制約条件テスト"""
        library = [
            'CCO',  # MW: 46
            'c1ccccc1',  # MW: 78
            'c1ccc(O)cc1',  # MW: 94
            'c1ccc(Cl)cc1',  # MW: 112
        ]
        
        constraints = {
            'molecular_weight': {'min': 50, 'max': 100}
        }
        
        candidates = designer.optimize(
            target_value=1.0,
            direction='maximize',
            method='exhaustive',
            library=library,
            constraints=constraints
        )
        
        # MW 50-100の範囲のみ
        for c in candidates:
            assert 50 <= c.properties['molecular_weight'] <= 100
    
    def test_direction_maximize(self, designer):
        """最大化方向テスト"""
        library = ['CCO', 'c1ccccc1']  # 長さ3 vs 8
        
        candidates = designer.optimize(
            target_value=1.0,
            direction='maximize',
            method='exhaustive',
            library=library
        )
        
        # 長い方（ベンゼン）が上位
        assert candidates[0].smiles == 'c1ccccc1'
    
    def test_direction_minimize(self, designer):
        """最小化方向テスト"""
        library = ['CCO', 'c1ccccc1']
        
        candidates = designer.optimize(
            target_value=1.0,
            direction='minimize',
            method='exhaustive',
            library=library
        )
        
        # 短い方（エタノール）が上位
        assert candidates[0].smiles == 'CCO'
    
    def test_direction_target(self, designer):
        """目標値方向テスト"""
        library = ['CCO', 'CCCO', 'c1ccccc1']  # 長さ3, 4, 8
        # 予測値: 0.3, 0.4, 0.8
        
        candidates = designer.optimize(
            target_value=0.4,
            direction='target',
            method='exhaustive',
            library=library
        )
        
        # 0.4に最も近いCCCOが上位
        assert candidates[0].smiles == 'CCCO'
    
    def test_properties_calculation(self, designer):
        """物性計算テスト"""
        library = ['CCO']
        
        candidates = designer.optimize(
            target_value=1.0,
            direction='maximize',
            method='exhaustive',
            library=library
        )
        
        properties = candidates[0].properties
        assert 'molecular_weight' in properties
        assert 'logP' in properties
        assert 'QED' in properties
    
    def test_invalid_method(self, designer):
        """無効な手法"""
        with pytest.raises(ValueError, match="Unknown method"):
            designer.optimize(
                target_value=1.0,
                method='invalid_method',
                library=['CCO']
            )
    
    def test_exhaustive_without_library(self, designer):
        """ライブラリなし全探索"""
        with pytest.raises(ValueError, match="Library is required"):
            designer.optimize(
                target_value=1.0,
                method='exhaustive'
            )
