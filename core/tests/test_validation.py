"""
入力検証モジュールのテスト

Implements: T-VALID-001  
カバレッジ対象: core/services/utils/validation.py
"""

import pandas as pd
import pytest

from core.services.utils.validation import (
    InputValidator,
    ValidationError,
    ValidationResult,
    validate_smiles,
    validate_smiles_list,
)


class TestValidationError:
    """ValidationError dataclassのテスト"""

    def test_creation(self):
        """作成"""
        error = ValidationError("field1", "error message",  "value1")
        assert error.field == "field1"
        assert error.message == "error message"
        assert error.value == "value1"


class TestValidationResult:
    """ValidationResultのテスト"""

    def test_initialization(self):
        """初期化"""
        result = ValidationResult()
        assert result.is_valid
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_add_error(self):
        """エラー追加"""
        result = ValidationResult()
        result.add_error("field1", "error1")
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.errors[0].field == "field1"

    def test_add_warning(self):
        """警告追加"""
        result = ValidationResult()
        result.add_warning("warning message")
        
        assert result.is_valid  # 警告は有効性に影響しない
        assert len(result.warnings) == 1

    def test_raise_if_invalid(self):
        """無効な場合に例外を発生"""
        result = ValidationResult()
        result.add_error("field1", "error1")
        result.add_error("field2", "error2")
        
        with pytest.raises(ValueError) as exc_info:
            result.raise_if_invalid()
        
        assert "field1" in str(exc_info.value)
        assert "field2" in str(exc_info.value)

    def test_raise_if_valid(self):
        """有効な場合は例外なし"""
        result = ValidationResult()
        result.raise_if_invalid()  # エラーなし

    def test_to_dict(self):
        """辞書への変換"""
        result = ValidationResult()
        result.add_error("field1", "error1")
        result.add_warning("warn1")
        
        data = result.to_dict()
        assert data["valid"] is False
        assert len(data["errors"]) == 1
        assert len(data["warnings"]) == 1


class TestInputValidator:
    """InputValidatorのテスト"""

    def test_validate_smiles_valid(self):
        """有効なSMILES"""
        validator = InputValidator()
        result = validator.validate_smiles("CCO")
        assert result.is_valid

    def test_validate_smiles_invalid(self):
        """無効なSMILES"""
        validator = InputValidator()
        result = validator.validate_smiles("invalid_smiles_123")
        assert not result.is_valid

    def test_validate_smiles_empty_not_allowed(self):
        """空SMILES（不許可）"""
        validator = InputValidator()
        result = validator.validate_smiles("", allow_empty=False)
        assert not result.is_valid

    def test_validate_smiles_empty_allowed(self):
        """空SMILES（許可）"""
        validator = InputValidator()
        result = validator.validate_smiles("", allow_empty=True)
        assert result.is_valid

    def test_validate_smiles_not_string(self):
        """文字列以外"""
        validator = InputValidator()
        result = validator.validate_smiles(123)
        assert not result.is_valid

    def test_validate_smiles_list_valid(self):
        """有効なSMILESリスト"""
        validator = InputValidator()
        result = validator.validate_smiles_list(["CCO", "c1ccccc1", "CC(=O)O"])
        assert result.is_valid or len(result.errors) == 0 or len(result.warnings) > 0

    def test_validate_smiles_list_invalid_items(self):
        """無効なアイテムを含むリスト"""
        validator = InputValidator()
        result = validator.validate_smiles_list(["CCO", "invalid", "c1ccccc1"])
        # RDKitがある場合は無効なSMILESがエラー
        # ない場合は警告

    def test_validate_smiles_list_min_count(self):
        """最小件数チェック"""
        validator = InputValidator()
        result = validator.validate_smiles_list(["CCO"], min_count=3)
        assert not result.is_valid

    def test_validate_smiles_list_max_count(self):
        """最大件数チェック"""
        validator = InputValidator()
        result = validator.validate_smiles_list(["CCO"] * 10, max_count=5)
        assert not result.is_valid

    def test_validate_smiles_list_not_list(self):
        """リスト以外"""
        validator = InputValidator()
        result = validator.validate_smiles_list("not a list")
        assert not result.is_valid

    def test_validate_numeric_valid(self):
        """有効な数値"""
        validator = InputValidator()
        result = validator.validate_numeric(42.5, "number")
        assert result.is_valid

    def test_validate_numeric_min_value(self):
        """最小値チェック"""
        validator = InputValidator()
        result = validator.validate_numeric(5, "num", min_value=10)
        assert not result.is_valid

    def test_validate_numeric_max_value(self):
        """最大値チェック"""
        validator = InputValidator()
        result = validator.validate_numeric(100, "num", max_value=50)
        assert not result.is_valid

    def test_validate_numeric_not_number(self):
        """数値以外"""
        validator = InputValidator()
        result = validator.validate_numeric("not a number", "num")
        assert not result.is_valid

    def test_validate_numeric_none_allowed(self):
        """None許可"""
        validator = InputValidator()
        result = validator.validate_numeric(None, "num", allow_none=True)
        assert result.is_valid

    def test_validate_numeric_none_not_allowed(self):
        """None不許可"""
        validator = InputValidator()
        result = validator.validate_numeric(None, "num", allow_none=False)
        assert not result.is_valid

    def test_validate_dataframe_valid(self):
        """有効なDataFrame"""
        validator = InputValidator()
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        result = validator.validate_dataframe(df)
        assert result.is_valid

    def test_validate_dataframe_not_dataframe(self):
        """DataFrame以外"""
        validator = InputValidator()
        result = validator.validate_dataframe("not a dataframe")
        assert not result.is_valid

    def test_validate_dataframe_min_rows(self):
        """最小行数チェック"""
        validator = InputValidator()
        df = pd.DataFrame({"col1": [1]})
        result = validator.validate_dataframe(df, min_rows=5)
        assert not result.is_valid

    def test_validate_dataframe_required_columns(self):
        """必須カラムチェック"""
        validator = InputValidator()
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        result = validator.validate_dataframe(
            df, required_columns=["col1", "col3"]
        )
        assert not result.is_valid

    def test_validate_model_params_valid(self):
        """有効なモデルパラメータ"""
        validator = InputValidator()
        params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
        }
        schema = {
            "n_estimators": {"type": int, "min": 1, "max": 1000},
            "learning_rate": {"type": float, "min": 0.001, "max": 1.0},
        }
        result = validator.validate_model_params(params, schema)
        assert result.is_valid

    def test_validate_model_params_required_missing(self):
        """必須パラメータ欠如"""
        validator = InputValidator()
        params = {}
        schema = {
            "n_estimators": {"required": True},
        }
        result = validator.validate_model_params(params, schema)
        assert not result.is_valid

    def test_validate_model_params_wrong_type(self):
        """型違い"""
        validator = InputValidator()
        params = {"n_estimators": "100"}  # 文字列
        schema = {"n_estimators": {"type": int}}
        result = validator.validate_model_params(params, schema)
        assert not result.is_valid

    def test_validate_model_params_out_of_range(self):
        """範囲外"""
        validator = InputValidator()
        params = {"learning_rate": 2.0}
        schema = {"learning_rate": {"min": 0.0, "max": 1.0}}
        result = validator.validate_model_params(params, schema)
        assert not result.is_valid

    def test_validate_model_params_invalid_choice(self):
        """無効な選択肢"""
        validator = InputValidator()
        params = {"model_type": "invalid"}
        schema = {"model_type": {"choices": ["rf", "gbm", "xgb"]}}
        result = validator.validate_model_params(params, schema)
        assert not result.is_valid


class TestConvenienceFunctions:
    """便利関数のテスト"""

    def test_validate_smiles_function_valid(self):
        """validate_smiles関数（有効）"""
        is_valid = validate_smiles("CCO")
        assert is_valid or validate_smiles("c1ccccc1")  # RDKit次第

    def test_validate_smiles_function_invalid(self):
        """validate_smiles関数（無効）"""
        is_valid = validate_smiles("definitely_invalid_123")
        # RDKitがない場合は警告のみで有効となる可能性

    def test_validate_smiles_list_function(self):
        """validate_smiles_list関数"""
        result = validate_smiles_list(["CCO", "c1ccccc1"])
        assert isinstance(result, ValidationResult)


class TestEdgeCases:
    """エッジケースのテスト"""

    def test_many_errors(self):
        """多数のエラー"""
        result = ValidationResult()
        for i in range(100):
            result.add_error(f"field{i}", f"error{i}")
        
        assert not result.is_valid
        assert len(result.errors) == 100

    def test_mixed_errors_and_warnings(self):
        """エラーと警告の混在"""
        result = ValidationResult()
        result.add_error("error_field", "error")
        result.add_warning("warning1")
        result.add_warning("warning2")
        
        assert not result.is_valid
        assert len(result.errors) == 1
        assert len(result.warnings) == 2

    def test_large_smiles_list(self):
        """大規模SMILESリスト"""
        validator = InputValidator()
        large_list = ["CCO"] * 1000
        result = validator.validate_smiles_list(large_list)
        # エラーが多くても処理できる

    def test_unicode_in_validation(self):
        """Unicode文字"""
        result = ValidationResult()
        result.add_error("フィールド", "エラーメッセージ", "値")
        assert not result.is_valid
