# -*- coding: utf-8 -*-
"""
MolT5翻訳エンジンのテスト

カバレッジ目標: 0% → 80%
"""
import pytest


class TestMolT5TranslatorInit:
    """初期化テスト"""
    
    def test_init_caption2smiles(self):
        """caption2smilesタスク初期化"""
        from core.services.generation.molt5_translator import MolT5Translator
        
        translator = MolT5Translator(task='caption2smiles')
        assert translator.task == 'caption2smiles'
        assert translator.device in ['cpu', 'cuda']
    
    def test_init_smiles2caption(self):
        """smiles2captionタスク初期化"""
        from core.services.generation.molt5_translator import MolT5Translator
        
        translator = MolT5Translator(task='smiles2caption')
        assert translator.task == 'smiles2caption'
    
    def test_init_invalid_task(self):
        """不正なタスク"""
        from core.services.generation.molt5_translator import MolT5Translator
        
        with pytest.raises(ValueError):
            MolT5Translator(task='invalid_task')


class TestMolT5TranslatorMocked:
    """モック化テスト（transformersなし環境向け）"""
    
    def test_load_model_no_transformers(self, monkeypatch):
        """transformersなし環境"""
        from core.services.generation.molt5_translator import MolT5Translator
        
        def mock_import(*args, **kwargs):
            raise ImportError("No module named 'transformers'")
        
        translator = MolT5Translator()
        monkeypatch.setattr('builtins.__import__', mock_import)
        
        with pytest.raises(ImportError):
            translator._load_model()


class TestMoleculeScorer:
    """分子スコアリングテスト"""
    
    def test_score_molecule_basic(self):
        """基本的なスコアリング"""
        from core.services.generation.molecule_scorer import MoleculeScorer
        
        scorer = MoleculeScorer()
        result = scorer.score_molecule('CCO')
        
        assert 'smiles' in result
        assert 'overall_score' in result
        assert 'valid' in result
    
    def test_qed_calculation(self):
        """QED計算"""
        from core.services.generation.molecule_scorer import MoleculeScorer
        
        scorer = MoleculeScorer()
        qed = scorer.calculate_qed('CCO')
        
        if qed is not None:
            assert 0.0 <= qed <= 1.0
    
    def test_sa_score_calculation(self):
        """SA Score計算"""
        from core.services.generation.molecule_scorer import MoleculeScorer
        
        scorer = MoleculeScorer()
        sa = scorer.calculate_sa_score('c1ccccc1')
        
        if sa is not None:
            assert 1.0 <= sa <= 10.0
    
    def test_lipinski_check(self):
        """Lipinskiチェック"""
        from core.services.generation.molecule_scorer import MoleculeScorer
        
        scorer = MoleculeScorer()
        lipinski = scorer.check_lipinski('CCO')
        
        if lipinski is not None:
            assert 'all_pass' in lipinski
            assert 'values' in lipinski
    
    def test_pains_check(self):
        """PAINSフィルタ"""
        from core.services.generation.molecule_scorer import MoleculeScorer
        
        scorer = MoleculeScorer()
        is_clean = scorer.check_pains('CCO')
        
        if is_clean is not None:
            assert isinstance(is_clean, bool)
    
    def test_rank_molecules(self):
        """分子ランキング"""
        from core.services.generation.molecule_scorer import MoleculeScorer
        
        scorer = MoleculeScorer()
        ranked = scorer.rank_molecules(['CCO', 'c1ccccc1', 'CC(C)O'])
        
        assert len(ranked) == 3
        # スコア降順
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)


class TestMolGPTWithMolT5Integration:
    """MolGPT × MolT5統合テスト"""
    
    def test_text_to_molecule_fallback(self):
        """text_to_molecule フォールバック（transformersなし環境OK）"""
        from core.services.generation.molgpt_engine import MolGPTGenerator
        
        gen = MolGPTGenerator()
        
        # MolT5が利用できない場合、基本生成にフォールバック
        try:
            molecules = gen.text_to_molecule(
                "aspirin-like molecule",
                n_molecules=3,
                use_scoring=False
            )
            
            assert len(molecules) > 0
            assert all(mol.smiles for mol in molecules)
        except ImportError:
            # transformersなし環境ではスキップ
            pytest.skip("transformers not available")
    
    def test_text_to_molecule_with_scoring(self):
        """スコアリング付きtext_to_molecule"""
        from core.services.generation.molgpt_engine import MolGPTGenerator
        
        gen = MolGPTGenerator()
        
        try:
            molecules = gen.text_to_molecule(
                "simple alcohol",
                n_molecules=2,
                use_scoring=True
            )
            
            assert len(molecules) > 0
            # スコアが降順
            scores = [m.score for m in molecules]
            assert scores == sorted(scores, reverse=True)
        except ImportError:
            # transformersなし環境ではスキップ
            pytest.skip("transformers not available")
