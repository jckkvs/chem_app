"""
時系列モデル統合 - Prophet, tsfresh, ARIMA対応

Implements: F-TIME-SERIES-001
設計思想:
- 時系列データの自動検出
- Prophet、ARIMA、SARIMA、tsfresh等の統合
- 時系列特化の前処理・特徴量生成
- 時系列分割（TimeSeriesSplit）
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

logger = logging.getLogger(__name__)


class ProphetWrapper(BaseEstimator, RegressorMixin):
    """
    Prophetのscikit-learn互換ラッパー
    
    Example:
        >>> model = ProphetWrapper()
        >>> model.fit(X, y)  # Xには'ds'カラム（日付）、yは数値
        >>> y_pred = model.predict(X_future)
    """
    
    def __init__(
        self,
        growth: str = 'linear',
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        yearly_seasonality: Union[bool, str] = 'auto',
        weekly_seasonality: Union[bool, str] = 'auto',
        daily_seasonality: Union[bool, str] = 'auto',
        **kwargs
    ):
        self.growth = growth
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.kwargs = kwargs
        
        self.model_ = None
        self.date_column_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """学習"""
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("Prophet required: pip install prophet")
        
        # 日時カラム検出
        self.date_column_ = self._detect_date_column(X)
        
        # Prophet用のDataFrame作成
        df = pd.DataFrame({
            'ds': X[self.date_column_],
            'y': y.values
        })
        
        # モデル作成
        self.model_ = Prophet(
            growth=self.growth,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            **self.kwargs
        )
        
        # 学習
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_.fit(df)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測"""
        if self.model_ is None:
            raise ValueError("Model not fitted")
        
        # Prophet用のDataFrame
        future = pd.DataFrame({'ds': X[self.date_column_]})
        
        # 予測
        forecast = self.model_.predict(future)
        return forecast['yhat'].values
    
    def _detect_date_column(self, X: pd.DataFrame) -> str:
        """日時カラムを検出"""
        # 'ds'カラムがあればそれを使う
        if 'ds' in X.columns:
            return 'ds'
        
        # datetime型のカラムを探す
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                return col
        
        # 見つからない場合はエラー
        raise ValueError("日時カラムが見つかりません。'ds'カラムまたはdatetime型のカラムが必要です")


class TimeSeriesFeatureExtractor:
    """
    時系列特徴量抽出（tsfresh統合）
    
    Example:
        >>> extractor = TimeSeriesFeatureExtractor()
        >>> features = extractor.fit_transform(df, id_column='id', time_column='time')
    """
    
    def __init__(
        self,
        default_fc_parameters: str = 'efficient',  # 'minimal', 'efficient', 'comprehensive'
        n_jobs: int = -1,
    ):
        self.default_fc_parameters = default_fc_parameters
        self.n_jobs = n_jobs
        
        self.feature_names_ = None
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """学習（特徴量名の記録）"""
        return self
    
    def transform(
        self, 
        X: pd.DataFrame,
        id_column: str,
        time_column: str,
        value_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """変換"""
        try:
            from tsfresh import extract_features
            from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters, ComprehensiveFCParameters
        except ImportError:
            raise ImportError("tsfresh required: pip install tsfresh")
        
        # パラメータ選択
        if self.default_fc_parameters == 'minimal':
            fc_params = MinimalFCParameters()
        elif self.default_fc_parameters == 'efficient':
            fc_params = EfficientFCParameters()
        else:
            fc_params = ComprehensiveFCParameters()
        
        # 特徴量抽出
        features = extract_features(
            X,
            column_id=id_column,
            column_sort=time_column,
            default_fc_parameters=fc_params,
            n_jobs=self.n_jobs,
            disable_progressbar=True,
        )
        
        self.feature_names_ = features.columns.tolist()
        
        return features
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        id_column: str = 'id',
        time_column: str = 'time',
        value_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """学習+変換"""
        return self.fit(X, y).transform(X, id_column, time_column, value_columns)


class ARIMAWrapper(BaseEstimator, RegressorMixin):
    """
    ARIMA/SARIMAのscikit-learn互換ラッパー
    
    Example:
        >>> model = ARIMAWrapper(order=(1,1,1))
        >>> model.fit(X, y)
        >>> y_pred = model.predict(X_future)
    """
    
    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        **kwargs
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.kwargs = kwargs
        
        self.model_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """学習"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            raise ImportError("statsmodels required: pip install statsmodels")
        
        # ARIMAモデル作成
        self.model_ = ARIMA(
            y,
            order=self.order,
            seasonal_order=self.seasonal_order,
            **self.kwargs
        )
        
        # 学習
        self.model_ = self.model_.fit()
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測"""
        if self.model_ is None:
            raise ValueError("Model not fitted")
        
        # 予測期間
        n_periods = len(X)
        forecast = self.model_.forecast(steps=n_periods)
        
        return forecast.values


def create_timeseries_pipeline(
    model_type: str = 'prophet',
    date_column: str = 'ds',
    **model_kwargs
) -> Any:
    """
    時系列パイプライン作成
    
    Args:
        model_type: モデル種別 ('prophet', 'arima', 'sarima')
        date_column: 日時カラム名
        **model_kwargs: モデル固有のパラメータ
        
    Returns:
        時系列モデル
    """
    if model_type == 'prophet':
        return ProphetWrapper(**model_kwargs)
    
    elif model_type == 'arima':
        return ARIMAWrapper(**model_kwargs)
    
    elif model_type == 'sarima':
        # SARIMAは季節性パラメータを指定
        seasonal_order = model_kwargs.pop('seasonal_order', (1, 1, 1, 12))
        return ARIMAWrapper(seasonal_order=seasonal_order, **model_kwargs)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# 時系列データ検出ヘルパー
def is_timeseries_data(df: pd.DataFrame) -> bool:
    """時系列データかどうか判定"""
    # datetime型のカラムがあるか
    has_datetime = any(pd.api.types.is_datetime64_any_dtype(df[col]) for col in df.columns)
    
    # 'ds'や'date'などの時系列っぽいカラム名
    timeseries_columns = ['ds', 'date', 'datetime', 'time', 'timestamp']
    has_timeseries_column = any(col.lower() in timeseries_columns for col in df.columns)
    
    return has_datetime or has_timeseries_column


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """日時カラムを検出"""
    # datetime型のカラムを優先
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    
    # カラム名から推測
    timeseries_columns = ['ds', 'date', 'datetime', 'time', 'timestamp']
    for col in df.columns:
        if col.lower() in timeseries_columns:
            return col
    
    return None
