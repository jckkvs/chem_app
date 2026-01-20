"""
機械学習パイプライン - 前処理統合版

Implements: F-001, F-007
設計思想:
- SmartPreprocessorとの統合
- Cross-Validation + Final Fit
- 可視化アーティファクト生成
- MLflow追跡

参考文献:
- scikit-learn Pipeline documentation
- MLflow Model Tracking best practices
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Dict, Any, Optional, Literal, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_validate, learning_curve
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, f1_score, precision_score, recall_score,
)
from xgboost import XGBRegressor, XGBClassifier
import lightgbm as lgb

from .tracking import MLTracker
from .preprocessor import SmartPreprocessor, PreprocessorFactory
from .applicability_domain import ApplicabilityDomain, ADResult
from .uncertainty import UncertaintyQuantifier
from ..vis.shap_eng import SHAPEngine
from ..vis.pdp_eng import PDPEngine
from ..vis.plots import PlotEngine

logger = logging.getLogger(__name__)


# モデル定義
MODEL_REGISTRY: Dict[Tuple[str, str], type] = {
    ('regression', 'random_forest'): RandomForestRegressor,
    ('regression', 'xgboost'): XGBRegressor,
    ('regression', 'lightgbm'): lgb.LGBMRegressor,
    ('regression', 'lgbm'): lgb.LGBMRegressor,
    ('classification', 'random_forest'): RandomForestClassifier,
    ('classification', 'xgboost'): XGBClassifier,
    ('classification', 'lightgbm'): lgb.LGBMClassifier,
    ('classification', 'lgbm'): lgb.LGBMClassifier,
}

# デフォルトハイパーパラメータ
DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    'random_forest': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1},
    'xgboost': {'random_state': 42, 'n_jobs': -1},
    'lightgbm': {'random_state': 42, 'n_jobs': -1, 'verbose': -1},
    'lgbm': {'random_state': 42, 'n_jobs': -1, 'verbose': -1},
}


class MLPipeline:
    """
    統合機械学習パイプライン
    
    Features:
    - 自動前処理（SmartPreprocessor統合）
    - 複数モデルタイプ対応
    - Cross-Validation
    - 学習曲線・残差プロット生成
    - SHAP/PDP説明性
    - MLflow追跡
    
    Example (シンプル):
        >>> pipeline = MLPipeline()
        >>> metrics = pipeline.train(X, y)
    
    Example (カスタマイズ):
        >>> pipeline = MLPipeline(
        ...     model_type='lightgbm',
        ...     task_type='regression',
        ...     preprocessor_preset='robust',
        ...     cv_folds=10,
        ... )
        >>> metrics = pipeline.train(X, y)
    """
    
    def __init__(
        self,
        model_type: Literal['random_forest', 'xgboost', 'lightgbm', 'lgbm'] = 'lightgbm',
        task_type: Literal['regression', 'classification'] = 'regression',
        cv_folds: int = 5,
        preprocessor: Optional[SmartPreprocessor] = None,
        preprocessor_preset: str = 'tree_optimized',
        tracker: Optional[MLTracker] = None,
        model_params: Optional[Dict[str, Any]] = None,
        generate_plots: bool = True,
        enable_ad: bool = True,
        enable_uq: bool = True,
        ad_params: Optional[Dict[str, Any]] = None,
        uq_params: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,  # 後方互換性
    ):
        """
        Args:
            model_type: モデルタイプ
            task_type: タスクタイプ
            cv_folds: クロスバリデーションの分割数
            preprocessor: カスタム前処理器
            preprocessor_preset: 前処理プリセット
            tracker: MLflow追跡器
            model_params: モデルハイパーパラメータ
            generate_plots: 可視化を生成するか
            enable_ad: Applicability Domain を有効化
            enable_uq: Uncertainty Quantification を有効化
            ad_params: AD モデルのパラメータ
            uq_params: UQ モデルのパラメータ
            config: 後方互換性用の設定辞書
        """
        # 後方互換性: configから値を取得
        if config:
            model_type = config.get('model_type', model_type)
            task_type = config.get('task_type', task_type)
            cv_folds = config.get('cv_folds', cv_folds)
        
        self.model_type = model_type
        self.task_type = task_type
        self.cv_folds = cv_folds
        self.model_params = model_params or {}
        self.generate_plots = generate_plots
        self.enable_ad = enable_ad
        self.enable_uq = enable_uq
        self.ad_params = ad_params or {}
        self.uq_params = uq_params or {}
        self.config = config or {}
        
        # 前処理器
        if preprocessor:
            self.preprocessor = preprocessor
        else:
            self.preprocessor = PreprocessorFactory.create(preprocessor_preset)
        
        # 追跡器
        self.tracker = tracker or MLTracker()
        
        # 学習後の状態
        self.model: Optional[BaseEstimator] = None
        self.metrics_: Optional[Dict[str, float]] = None
        self.feature_names_: Optional[List[str]] = None
        self.ad_model_: Optional[ApplicabilityDomain] = None
        self.uq_model_: Optional[UncertaintyQuantifier] = None
    
    def _get_model(self) -> BaseEstimator:
        """モデルインスタンスを生成"""
        key = (self.task_type, self.model_type)
        
        if key not in MODEL_REGISTRY:
            logger.warning(f"不明なモデルタイプ: {key}、RandomForestを使用")
            key = (self.task_type, 'random_forest')
        
        model_class = MODEL_REGISTRY[key]
        default_params = DEFAULT_PARAMS.get(self.model_type, {})
        params = {**default_params, **self.model_params}
        
        return model_class(**params)
    
    def _get_cv_scoring(self) -> str:
        """CVスコアリング指標を取得"""
        if self.task_type == 'regression':
            return 'neg_mean_squared_error'
        return 'accuracy'
    
    def train(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        run_name: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        モデルを訓練
        
        Args:
            X: 特徴量DataFrame
            y: ターゲット変数
            run_name: MLflow run名
            
        Returns:
            Dict[str, float]: 評価指標
        """
        run_name = run_name or self.config.get('run_name', 'train_run')
        
        with self.tracker.start_run(run_name=run_name):
            # パラメータログ
            self.tracker.log_params({
                'model_type': self.model_type,
                'task_type': self.task_type,
                'cv_folds': self.cv_folds,
                **self.model_params,
            })
            
            # 1. 前処理
            logger.info("前処理を実行中...")
            X_processed = self.preprocessor.fit_transform(X, y)
            self.feature_names_ = list(X_processed.columns)
            
            # 2. モデル生成
            model = self._get_model()
            
            # 3. Cross-Validation
            logger.info(f"Cross-Validation ({self.cv_folds} folds)...")
            scoring = self._get_cv_scoring()
            cv_results = cross_validate(
                model, X_processed, y, 
                cv=self.cv_folds, 
                scoring=scoring,
                return_train_score=True,
            )
            
            # CV指標
            metrics: Dict[str, float] = {
                'cv_mean_score': float(np.mean(cv_results['test_score'])),
                'cv_std_score': float(np.std(cv_results['test_score'])),
                'cv_train_mean': float(np.mean(cv_results['train_score'])),
            }
            
            # 4. Final Fit
            logger.info("最終モデルを学習中...")
            model.fit(X_processed, y)
            self.model = model
            
            # 5. 訓練データ評価
            preds = model.predict(X_processed)
            metrics.update(self._calculate_metrics(y, preds))
            
            self.metrics_ = metrics
            self.tracker.log_metrics(metrics)
            
            # 6. Applicability Domain
            if self.enable_ad:
                logger.info("Applicability Domain を学習中...")
                ad_params = {'method': 'knn', **self.ad_params}
                self.ad_model_ = ApplicabilityDomain(**ad_params)
                self.ad_model_.fit(X_processed)
                self.tracker.log_params({'ad_enabled': True, **ad_params})
            
            # 7. Uncertainty Quantification
            if self.enable_uq and self.task_type == 'regression':
                logger.info("Uncertainty Quantification を学習中...")
                uq_params = {'method': 'quantile', **self.uq_params}
                self.uq_model_ = UncertaintyQuantifier(**uq_params)
                self.uq_model_.fit(X_processed, y)
                self.tracker.log_params({'uq_enabled': True, **uq_params})
            
            # 8. モデル保存
            self.tracker.log_model(model, "model")
            
            # 9. 可視化生成
            if self.generate_plots:
                self._generate_visualizations(X_processed, y, model)
            
            logger.info(f"訓練完了: {metrics}")
            return metrics
    
    def _calculate_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """評価指標を計算"""
        metrics = {}
        
        if self.task_type == 'regression':
            metrics['train_mse'] = float(mean_squared_error(y_true, y_pred))
            metrics['train_rmse'] = float(np.sqrt(metrics['train_mse']))
            metrics['train_mae'] = float(mean_absolute_error(y_true, y_pred))
            metrics['train_r2'] = float(r2_score(y_true, y_pred))
        else:
            metrics['train_accuracy'] = float(accuracy_score(y_true, y_pred))
            metrics['train_f1'] = float(f1_score(y_true, y_pred, average='weighted'))
            metrics['train_precision'] = float(precision_score(y_true, y_pred, average='weighted'))
            metrics['train_recall'] = float(recall_score(y_true, y_pred, average='weighted'))
        
        return metrics
    
    def _generate_visualizations(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        model: BaseEstimator
    ) -> None:
        """可視化アーティファクトを生成"""
        try:
            # サンプリング（大規模データ対策）
            n_samples = min(200, len(X))
            sample_idx = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X.iloc[sample_idx]
            y_sample = y.iloc[sample_idx]
            
            # 1. SHAP Summary
            try:
                shap_eng = SHAPEngine()
                shap_values, _ = shap_eng.explain(model, X_sample)
                fig_shap = shap_eng.plot_summary(shap_values, X_sample)
                self._save_figure(fig_shap, "shap_summary.png")
            except Exception as e:
                logger.warning(f"SHAP生成失敗: {e}")
            
            # 2. PDP (Top 3 features)
            try:
                pdp_eng = PDPEngine()
                # 特徴量重要度で上位3件を選択
                if hasattr(model, 'feature_importances_'):
                    top_idx = np.argsort(model.feature_importances_)[-3:]
                    top_features = [X.columns[i] for i in top_idx]
                else:
                    top_features = list(X.columns[:3])
                
                fig_pdp = pdp_eng.plot_pdp(model, X_sample, top_features)
                if fig_pdp:
                    self._save_figure(fig_pdp, "pdp.png")
            except Exception as e:
                logger.warning(f"PDP生成失敗: {e}")
            
            # 3. 学習曲線
            try:
                fig_lc = self._plot_learning_curve(model, X, y)
                self._save_figure(fig_lc, "learning_curve.png")
            except Exception as e:
                logger.warning(f"学習曲線生成失敗: {e}")
            
            # 4. 予測 vs 実測（回帰のみ）
            if self.task_type == 'regression':
                try:
                    preds = model.predict(X_sample)
                    fig_pred = PlotEngine.plot_predicted_vs_actual(
                        y_sample, pd.Series(preds), 'regression'
                    )
                    self._save_figure(fig_pred, "predicted_vs_actual.png")
                except Exception as e:
                    logger.warning(f"予測プロット生成失敗: {e}")
            
            # 5. 特徴量重要度
            try:
                fig_imp = self._plot_feature_importance(model, X.columns)
                if fig_imp:
                    self._save_figure(fig_imp, "feature_importance.png")
            except Exception as e:
                logger.warning(f"重要度プロット生成失敗: {e}")
                
        except Exception as e:
            logger.error(f"可視化生成エラー: {e}")
    
    def _plot_learning_curve(
        self, 
        model: BaseEstimator, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> plt.Figure:
        """学習曲線をプロット"""
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, 
            cv=min(5, self.cv_folds),
            train_sizes=np.linspace(0.1, 1.0, 5),
            scoring=self._get_cv_scoring(),
            n_jobs=-1,
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
        ax.plot(train_sizes, train_mean, 'o-', label='Training Score')
        ax.plot(train_sizes, val_mean, 'o-', label='Validation Score')
        
        ax.set_xlabel('Training Size')
        ax.set_ylabel('Score')
        ax.set_title('Learning Curve')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def _plot_feature_importance(
        self, 
        model: BaseEstimator, 
        feature_names: pd.Index
    ) -> Optional[plt.Figure]:
        """特徴量重要度をプロット"""
        if not hasattr(model, 'feature_importances_'):
            return None
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[-20:]  # Top 20
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(indices)), importances[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top 20 Feature Importances')
        
        return fig
    
    def _save_figure(self, fig: plt.Figure, filename: str) -> None:
        """図を保存してMLflowにログ"""
        path = os.path.join(tempfile.gettempdir(), filename)
        fig.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        self.tracker.log_artifact(path)
        logger.debug(f"図を保存: {filename}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測を実行
        
        Args:
            X: 特徴量DataFrame
            
        Returns:
            np.ndarray: 予測値
        """
        if self.model is None:
            raise RuntimeError("モデルが訓練されていません")
        
        X_processed = self.preprocessor.transform(X)
        return self.model.predict(X_processed)
    
    def get_applicability(self, X: pd.DataFrame) -> List[ADResult]:
        """
        Applicability Domain を評価
        
        Args:
            X: 特徴量DataFrame
            
        Returns:
            List[ADResult]: AD判定結果
        """
        if self.ad_model_ is None:
            raise RuntimeError("ADモデルが有効化されていません")
        
        X_processed = self.preprocessor.transform(X)
        return self.ad_model_.check(X_processed)
    
    def predict_uncertainty(
        self,
        X: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        不確実性付き予測 (回帰のみ)
        
        Args:
            X: 特徴量DataFrame
            
        Returns:
            (mean, lower, upper) 予測値、下限、上限
        """
        if self.uq_model_ is None:
            raise RuntimeError("UQモデルが有効化されていません")
        
        X_processed = self.preprocessor.transform(X)
        return self.uq_model_.predict_with_interval(X_processed)
    
    def save(self, path: str) -> None:
        """パイプライン全体を保存"""
        data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'model_type': self.model_type,
            'task_type': self.task_type,
            'feature_names': self.feature_names_,
            'metrics': self.metrics_,
            'ad_model': self.ad_model_,
            'uq_model': self.uq_model_,
        }
        joblib.dump(data, path)
    
    @classmethod
    def load(cls, path: str) -> 'MLPipeline':
        """保存済みパイプラインを読み込み"""
        data = joblib.load(path)
        
        pipeline = cls(
            model_type=data['model_type'],
            task_type=data['task_type'],
            preprocessor=data['preprocessor'],
        )
        pipeline.model = data['model']
        pipeline.feature_names_ = data.get('feature_names')
        pipeline.metrics_ = data.get('metrics')
        pipeline.ad_model_ = data.get('ad_model')
        pipeline.uq_model_ = data.get('uq_model')
        
        return pipeline
