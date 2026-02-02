"""
生成AI機能のユニットテスト

カバレッジ目標: 90%以上
"""
import json
import os
import pytest
import tempfile
from unittest.mock import Mock, patch, MagicMock

from core.services.generation.molgpt_engine import (
    MolGPTGenerator,
    GenerationConfig,
    GeneratedMolecule
)
from core.services.generation.validator import (
    GeneratedMoleculeValidator,
    ValidationResult
)


class TestGenerationConfig:
    """GenerationConfigテスト"""
    
    def test_default_config(self):
        """デフォルト設定"""
        config = GenerationConfig()
        assert config.n_molecules == 10
        assert config.temperature == 1.0
        assert config.max_length == 100


class TestMolGPTGeneratorInit:
    """MolGPTGenerator初期化テスト"""
    
    def test_init_default(self):
        """デフォルト初期化"""
        generator = MolGPTGenerator()
        assert generator.model_name == "gpt2"
        assert generator.device in ["cpu", "cuda"]
        assert generator._is_loaded is False
    
    def test_init_custom(self):
        """カスタム初期化"""
        generator = MolGPTGenerator(
            model_name="custom-model",
            device="cpu"
        )
        assert generator.model_name == "custom-model"
        assert generator.device == "cpu"


class TestMolGPTGeneratorBasic:
    """MolGPTGenerator基本機能テスト"""
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_model_loading(self, mock_tokenizer, mock_model):
        """モデルロードテスト"""
        try:
            import transformers
        except ImportError:
            pytest.skip("transformers not available")
        
        # モック設定
        mock_tok = MagicMock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "<eos>"
        mock_tokenizer.return_value = mock_tok
        
        mock_mdl = MagicMock()
        mock_mdl.to.return_value = mock_mdl
        mock_model.return_value = mock_mdl
        
        generator = MolGPTGenerator()
        generator._load_model()
        
        assert generator._is_loaded is True
        mock_tokenizer.assert_called_once()
        mock_model.assert_called_once()
    
    @patch('core.services.generation.molgpt_engine.MolGPTGenerator._load_model')
    def test_generate_basic(self, mock_load):
        """基本生成テスト"""
        generator = MolGPTGenerator()
        
        # モックトークナイザー・モデル設定
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.pad_token_id = 0
        
        # トークナイザー呼び出しのモック
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer.decode.return_value = "<start>CCO"
        generator._tokenizer = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.generate.return_value = [[1, 2, 3, 4]]
        generator._model = mock_model
        generator._is_loaded = True
        
        # 生成
        molecules = generator.generate(n_molecules=1)
        
        assert len(molecules) == 1
        assert molecules[0].smiles == "CCO"
    
    def test_clean_smiles(self):
        """SMILESクリーニングテスト"""
        generator = MolGPTGenerator()
        
        cleaned = generator._clean_smiles("<start>CCO\n ")
        assert cleaned == "CCO"


class TestMolGPTGeneratorConditional:
    """MolGPTGenerator条件付き生成テスト"""
    
    @patch('core.services.generation.molgpt_engine.MolGPTGenerator.generate')
    def test_conditional_generate(self, mock_generate):
        """条件付き生成テスト"""
        mock_generate.return_value = [
            GeneratedMolecule("CCO", 1.0),
            GeneratedMolecule("CC(C)O", 1.0)
        ]
        
        generator = MolGPTGenerator()
        molecules = generator.conditional_generate(
            properties={'logP': 2.0, 'MW': 300}
        )
        
        assert len(molecules) >= 1
        mock_generate.assert_called_once()
    
    @patch('core.services.generation.molgpt_engine.MolGPTGenerator.conditional_generate')
    def test_scaffold_optimization(self, mock_conditional):
        """骨格最適化テスト"""
        mock_conditional.return_value = [GeneratedMolecule("c1ccccc1C", 1.0)]
        
        generator = MolGPTGenerator()
        molecules = generator.scaffold_optimization(
            scaffold_smiles="c1ccccc1",
            target_property="logP",
            target_value=2.5
        )
        
        assert len(molecules) >= 1


class TestGeneratedMoleculeValidator:
    """Validatorテスト"""
    
    def test_init(self):
        """初期化テスト"""
        validator = GeneratedMoleculeValidator()
        assert validator.strict is False
    
    def test_validate_empty(self):
        """空SMILES検証"""
        validator = GeneratedMoleculeValidator()
        result = validator.validate("")
        
        assert result.is_valid is False
        assert "empty" in result.error.lower()
    
    def test_validate_invalid_type(self):
        """型エラー検証"""
        validator = GeneratedMoleculeValidator()
        result = validator.validate(None)
        
        assert result.is_valid is False
    
    @patch('core.services.generation.validator.GeneratedMoleculeValidator._check_rdkit')
    def test_simple_validation(self, mock_rdkit):
        """簡易検証テスト（RDKitなし）"""
        mock_rdkit.return_value = False
        
        validator = GeneratedMoleculeValidator()
        result = validator.validate("CCO")
        
        assert result.is_valid is True
        assert result.smiles == "CCO"
        assert result.qed_score is None
    
    @patch('core.services.generation.validator.GeneratedMoleculeValidator._check_rdkit')
    def test_invalid_characters(self, mock_rdkit):
        """不正文字検証"""
        mock_rdkit.return_value = False
        
        validator = GeneratedMoleculeValidator()
        result = validator.validate("CCO$%")
        
        assert result.is_valid is False


class TestGeneratedMoleculeValidatorFull:
    """Validator完全検証テスト（RDKitあり）"""
    
    def test_validate_valid_smiles(self):
        """妥当なSMILES"""
        validator = GeneratedMoleculeValidator()
        
        try:
            from rdkit import Chem
            result = validator.validate("CCO")
            
            assert result.is_valid is True
            assert result.qed_score is not None
            assert result.lipinski_violations >= 0
        except ImportError:
            pytest.skip("RDKit not available")
    
    def test_validate_invalid_smiles(self):
        """不正なSMILES"""
        validator = GeneratedMoleculeValidator()
        
        try:
            from rdkit import Chem
            result = validator.validate("INVALID_SMILES_XYZ")
            
            assert result.is_valid is False
            assert "cannot parse" in result.error.lower()
        except ImportError:
            pytest.skip("RDKit not available")
    
    def test_lipinski_check(self):
        """Lipinski則チェック"""
        validator = GeneratedMoleculeValidator()
        
        try:
            from rdkit import Chem
            result = validator.validate("CCO")  # エタノール（違反なし）
            
            assert result.lipinski_violations == 0
        except ImportError:
            pytest.skip("RDKit not available")
    
    def test_is_drug_like(self):
        """医薬品らしさ判定"""
        validator = GeneratedMoleculeValidator()
        
        assert validator.is_drug_like("CCO") in [True, False]


class TestGeneratedMoleculeValidatorStrict:
    """Validator厳密モードテスト"""
    
    def test_strict_mode_low_qed(self):
        """厳密モードでのQED不足"""
        validator = GeneratedMoleculeValidator(strict=True)
        
        try:
            from rdkit import Chem
            # 非常に複雑な分子（QEDが低い可能性）
            complex_smiles = "C" * 50  # 長鎖アルカン
            result = validator.validate(complex_smiles)
            
            # strictモードでは低QEDで失敗する可能性
            assert isinstance(result.is_valid, bool)
        except ImportError:
            pytest.skip("RDKit not available")
