"""
フィンガープリント計算エンジンのテスト

カバレッジ目標: 27.45% → 80%
"""
import numpy as np
import pandas as pd
import pytest

from core.services.features.fingerprint import FingerprintCalculator


class TestFingerprintCalculatorInit:
    """初期化テスト"""
    
    def test_init_default(self):
        """デフォルト初期化"""
        calc = FingerprintCalculator()
        assert calc.fp_type == 'morgan'
        assert calc.radius == 2
        assert calc.n_bits == 2048
    
    def test_init_custom(self):
        """カスタム初期化"""
        calc = FingerprintCalculator(fp_type='rdkit', radius=3, n_bits=1024)
        assert calc.fp_type == 'rdkit'
        assert calc.radius == 3
        assert calc.n_bits == 1024


class TestFingerprintCalculate:
    """単一SMILES計算テスト"""
    
    def test_calculate_valid_smiles(self):
        """妥当なSMILES"""
        calc = FingerprintCalculator(fp_type='morgan')
        fp = calc.calculate('CCO')
        
        assert fp is not None
        assert isinstance(fp, np.ndarray)
        assert len(fp) == 2048
        assert fp.dtype == np.int8
    
    def test_calculate_invalid_smiles(self):
        """不正なSMILES"""
        calc = FingerprintCalculator()
        fp = calc.calculate('INVALID_SMILES_XYZ')
        
        assert fp is None
    
    def test_calculate_rdkit_fp(self):
        """RDKit FP"""
        calc = FingerprintCalculator(fp_type='rdkit')
        fp = calc.calculate('c1ccccc1')
        
        assert fp is not None
        assert len(fp) == 2048
    
    def test_calculate_maccs_fp(self):
        """MACCS FP"""
        calc = FingerprintCalculator(fp_type='maccs')
        fp = calc.calculate('CCO')
        
        assert fp is not None
        assert len(fp) == 167  # MACCSキーは167ビット
    
    def test_calculate_atompair_fp(self):
        """AtomPair FP"""
        calc = FingerprintCalculator(fp_type='atompair')
        fp = calc.calculate('CC(C)O')
        
        assert fp is not None
        assert len(fp) == 2048
    
    def test_calculate_topological_fp(self):
        """Topological FP"""
        calc = FingerprintCalculator(fp_type='topological')
        fp = calc.calculate('c1ccccc1')
        
        assert fp is not None
        assert len(fp) == 2048
    
    def test_calculate_unknown_fp_type(self):
        """未知のFPタイプ"""
        calc = FingerprintCalculator(fp_type='unknown_type')
        fp = calc.calculate('CCO')
        
        assert fp is None


class TestFingerprintCalculateBatch:
    """バッチ計算テスト"""
    
    def test_calculate_batch_valid(self):
        """妥当なSMILESリスト"""
        calc = FingerprintCalculator()
        smiles = ['CCO', 'c1ccccc1', 'CC(C)O']
        
        df = calc.calculate_batch(smiles)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert df.shape[1] == 2048
        assert all(df.columns.str.startswith('FP_'))
    
    def test_calculate_batch_with_invalid(self):
        """一部不正なSMILES"""
        calc = FingerprintCalculator()
        smiles = ['CCO', 'INVALID', 'c1ccccc1']
        
        df = calc.calculate_batch(smiles)
        
        assert len(df) == 3
        # 不正なSMILESはゼロベクトル
        assert df.iloc[1].sum() == 0
    
    def test_calculate_batch_all_invalid(self):
        """全て不正なSMILES"""
        calc = FingerprintCalculator()
        smiles = ['INVALID1', 'INVALID2']
        
        df = calc.calculate_batch(smiles)
        
        # 空のDataFrameが返る
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    def test_calculate_batch_empty(self):
        """空リスト"""
        calc = FingerprintCalculator()
        df = calc.calculate_batch([])
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestTanimotoSimilarity:
    """Tanimoto類似度テスト"""
    
    def test_tanimoto_identical(self):
        """同一分子"""
        calc = FingerprintCalculator()
        sim = calc.tanimoto_similarity('CCO', 'CCO')
        
        assert sim == 1.0
    
    def test_tanimoto_similar(self):
        """類似分子"""
        calc = FingerprintCalculator()
        sim = calc.tanimoto_similarity('CCO', 'CCCO')
        
        assert 0.0 < sim < 1.0
    
    def test_tanimoto_different(self):
        """異なる分子"""
        calc = FingerprintCalculator()
        # 単純なアルコールと芳香環
        sim = calc.tanimoto_similarity('CCO', 'c1ccccc1')
        
        assert 0.0 <= sim < 0.5
    
    def test_tanimoto_invalid_smiles1(self):
        """1つ目が不正"""
        calc = FingerprintCalculator()
        sim = calc.tanimoto_similarity('INVALID', 'CCO')
        
        assert sim == 0.0
    
    def test_tanimoto_invalid_smiles2(self):
        """2つ目が不正"""
        calc = FingerprintCalculator()
        sim = calc.tanimoto_similarity('CCO', 'INVALID')
        
        assert sim == 0.0
    
    def test_tanimoto_both_invalid(self):
        """両方不正"""
        calc = FingerprintCalculator()
        sim = calc.tanimoto_similarity('INVALID1', 'INVALID2')
        
        assert sim == 0.0


class TestSimilarityMatrix:
    """類似度行列テスト"""
    
    def test_similarity_matrix_basic(self):
        """基本的な類似度行列"""
        calc = FingerprintCalculator()
        smiles = ['CCO', 'CCCO', 'c1ccccc1']
        
        matrix = calc.similarity_matrix(smiles)
        
        assert matrix.shape == (3, 3)
        # 対角線は1.0
        assert np.allclose(np.diag(matrix), 1.0)
        # 対称行列
        assert np.allclose(matrix, matrix.T)
    
    def test_similarity_matrix_single(self):
        """単一分子"""
        calc = FingerprintCalculator()
        matrix = calc.similarity_matrix(['CCO'])
        
        assert matrix.shape == (1, 1)
        assert matrix[0, 0] == 1.0
    
    def test_similarity_matrix_two(self):
        """2分子"""
        calc = FingerprintCalculator()
        smiles = ['CCO', 'CCCO']
        
        matrix = calc.similarity_matrix(smiles)
        
        assert matrix.shape == (2, 2)
        assert matrix[0, 1] == matrix[1, 0]  # 対称
        assert 0.0 < matrix[0, 1] < 1.0


class TestFindSimilar:
    """類似分子検索テスト"""
    
    def test_find_similar_basic(self):
        """基本的な検索"""
        calc = FingerprintCalculator()
        query = 'CCO'
        database = ['CCCO', 'c1ccccc1', 'CC(C)O', 'CCCCO']
        
        results = calc.find_similar(query, database, top_k=3)
        
        assert len(results) == 3
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)
        # スコアは降順
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_find_similar_top_k_larger(self):
        """top_kがDB数より大きい"""
        calc = FingerprintCalculator()
        query = 'CCO'
        database = ['CCCO', 'c1ccccc1']
        
        results = calc.find_similar(query, database, top_k=10)
        
        assert len(results) == 2
    
    def test_find_similar_exact_match(self):
        """完全一致が含まれる"""
        calc = FingerprintCalculator()
        query = 'CCO'
        database = ['CCO', 'CCCO', 'c1ccccc1']
        
        results = calc.find_similar(query, database, top_k=3)
        
        # 最初の結果は完全一致（スコア1.0）
        assert results[0][0] == 'CCO'
        assert results[0][1] == 1.0
    
    def test_find_similar_empty_database(self):
        """空のデータベース"""
        calc = FingerprintCalculator()
        results = calc.find_similar('CCO', [], top_k=5)
        
        assert results == []


class TestFingerprintIntegration:
    """統合テスト"""
    
    def test_workflow_basic(self):
        """基本的なワークフロー"""
        calc = FingerprintCalculator(fp_type='morgan', radius=2)
        
        # バッチ計算
        smiles_list = ['CCO', 'CCCO', 'c1ccccc1']
        df = calc.calculate_batch(smiles_list)
        
        assert len(df) == 3
        
        # 類似度行列
        matrix = calc.similarity_matrix(smiles_list)
        
        assert matrix.shape == (3, 3)
        
        # 類似検索
        results = calc.find_similar('CCO', smiles_list[1:], top_k=2)
        
        assert len(results) == 2
    
    def test_different_fp_types(self):
        """異なるFPタイプでの一貫性"""
        smiles = 'c1ccccc1'
        
        for fp_type in ['morgan', 'rdkit', 'maccs', 'atompair', 'topological']:
            calc = FingerprintCalculator(fp_type=fp_type)
            fp = calc.calculate(smiles)
            
            assert fp is not None
            assert isinstance(fp, np.ndarray)
