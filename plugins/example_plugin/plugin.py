"""
サンプルプラグイン

このプラグインは予測結果にログを追加し、
結果を範囲内に制限する簡単な例です。

Author: Chemical ML Platform Team
License: MIT
"""

from core.services.plugin import Plugin
import logging

logger = logging.getLogger(__name__)


def create_plugin():
    """
    プラグインのエントリーポイント
    
    Returns:
        Plugin: プラグインインスタンス
    """
    return Plugin(
        name="example_plugin",
        version="1.0.0",
        description="予測結果のログ出力と範囲制限を行うサンプルプラグイン",
        author="Chemical ML Platform Team",
        license="MIT",
        hooks={
            "on_prediction": on_prediction_hook,
            "on_training": on_training_hook,
        },
        config={
            "min_value": 0.0,
            "max_value": 100.0,
            "log_predictions": True,
        }
    )


def on_prediction_hook(smiles: str, prediction: float, **kwargs) -> float:
    """
    予測後に実行されるフック
    
    機能:
    - 予測値をログに出力
    - 予測値を設定範囲内に制限
    
    Args:
        smiles: 入力SMILES
        prediction: 予測値
        **kwargs: その他のコンテキスト情報
        
    Returns:
        float: 調整後の予測値
    """
    plugin = kwargs.get('plugin')
    config = plugin.config if plugin else {}
    
    # ログ出力
    if config.get('log_predictions', True):
        logger.info(f"Prediction for {smiles}: {prediction:.3f}")
    
    # 範囲制限
    min_val = config.get('min_value', 0.0)
    max_val = config.get('max_value', 100.0)
    
    adjusted = max(min_val, min(prediction, max_val))
    
    if adjusted != prediction:
        logger.info(f"Adjusted prediction from {prediction:.3f} to {adjusted:.3f}")
    
    return adjusted


def on_training_hook(experiment, **kwargs) -> None:
    """
    学習完了後に実行されるフック
    
    機能:
    - 学習完了をログに出力
    - メトリクスを表示
    
    Args:
        experiment: 実験オブジェクト
        **kwargs: その他のコンテキスト情報
    """
    logger.info(f"🎉 Training completed for experiment: {experiment.name}")
    
    if hasattr(experiment, 'result') and experiment.result:
        metrics = experiment.result.metrics
        logger.info(f"Performance metrics: {metrics}")
    
    logger.info(f"Model type: {experiment.model_type}")
    logger.info(f"Feature type: {experiment.feature_type}")
