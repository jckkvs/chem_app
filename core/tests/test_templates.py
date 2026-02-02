"""
テンプレート管理テスト
"""

import pytest
from core.services.ml.templates import ModelTemplate, TemplateManager


class TestTemplateManager:
    """テンプレート管理テスト"""
    
    def test_list_all(self):
        """全テンプレート取得"""
        templates = TemplateManager.list_all()
        
        assert len(templates) == 6
        assert all(isinstance(t, ModelTemplate) for t in templates)
    
    def test_get_template(self):
        """テンプレート個別取得"""
        template = TemplateManager.get('drug_discovery_basic')
        
        assert template is not None
        assert template.id == 'drug_discovery_basic'
        assert template.name_ja == '創薬基礎'
        assert template.icon == '🧪'
        assert template.model_type == 'random_forest'
        assert template.task_type == 'regression'
    
    def test_get_nonexistent_template(self):
        """存在しないテンプレート"""
        template = TemplateManager.get('nonexistent')
        assert template is None
    
    def test_list_by_difficulty(self):
        """難易度フィルタ"""
        beginners = TemplateManager.list_by_difficulty('beginner')
        intermediates = TemplateManager.list_by_difficulty('intermediate')
        
        assert len(beginners) >= 1
        assert len(intermediates) >= 1
        assert all(t.difficulty == 'beginner' for t in beginners)
        assert all(t.difficulty == 'intermediate' for t in intermediates)
    
    def test_list_by_task_type(self):
        """タスクタイプフィルタ"""
        regression = TemplateManager.list_by_task_type('regression')
        classification = TemplateManager.list_by_task_type('classification')
        
        assert len(regression) >= 1
        assert len(classification) >= 1
        assert all(t.task_type == 'regression' for t in regression)
        assert all(t.task_type == 'classification' for t in classification)
    
    def test_apply_to_config(self):
        """設定変換"""
        config = TemplateManager.apply_to_config('toxicity_screening')
        
        # 必須フィールド
        assert 'model_type' in config
        assert 'task_type' in config
        assert 'features' in config
        assert 'model_params' in config
        
        # 値確認
        assert config['model_type'] == 'lightgbm'
        assert config['task_type'] == 'classification'
        assert 'rdkit' in config['features']
        
        # パラメータ
        assert config['model_params']['n_estimators'] == 200
    
    def test_apply_nonexistent_template(self):
        """存在しないテンプレート適用"""
        with pytest.raises(ValueError, match="Template.*not found"):
            TemplateManager.apply_to_config('nonexistent')
    
    def test_get_summary(self):
        """統計情報"""
        summary = TemplateManager.get_summary()
        
        assert 'total' in summary
        assert 'regression' in summary
        assert 'classification' in summary
        assert 'beginner' in summary
        
        assert summary['total'] == 6
        assert summary['regression'] + summary['classification'] == 6


class TestTemplateContent:
    """テンプレート内容テスト"""
    
    def test_all_templates_have_required_fields(self):
        """全テンプレートが必須フィールドを持つ"""
        for template in TemplateManager.list_all():
            assert template.id
            assert template.name
            assert template.name_ja
            assert template.description
            assert template.icon
            assert template.model_type
            assert template.task_type
            assert len(template.features) > 0
            assert len(template.use_cases) > 0
            assert len(template.pros) > 0
    
    def test_toxicity_template_details(self):
        """毒性テンプレート詳細"""
        template = TemplateManager.get('toxicity_screening')
        
        assert template.task_type == 'classification'
        assert template.difficulty == 'intermediate'
        assert 'Tox21' in ' '.join(template.use_cases)
        assert '高精度' in template.pros
    
    def test_fast_screening_template_details(self):
        """高速スクリーニングテンプレート詳細"""
        template = TemplateManager.get('fast_screening')
        
        assert template.difficulty == 'beginner'
        assert template.model_params['n_estimators'] < 100  # 高速化のため少なめ
        assert '高速' in ' '.join(template.pros)
    
    def test_solubility_template_details(self):
        """溶解度テンプレート詳細"""
        template = TemplateManager.get('solubility_prediction')
        
        assert template.task_type == 'regression'
        assert template.model_type == 'lightgbm'
        assert 'logP' in ' '.join(template.use_cases) or '溶解度' in ' '.join(template.use_cases)
