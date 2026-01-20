"""
データ品質ダッシュボードエンジン

Implements: F-DATAQUALITY-001
設計思想:
- 異常値・欠損・重複の自動検出
- 品質スコア計算
- 修正推奨の提示
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """データ品質レポート"""
    # デフォルトなしフィールドを先に
    overall_score: float
    n_samples: int
    n_features: int
    missing_count: int
    missing_percent: float
    duplicate_rows: int
    duplicate_percent: float
    outlier_count: int
    outlier_percent: float
    
    # デフォルトありフィールドを後に
    missing_by_column: Dict[str, float] = field(default_factory=dict)
    outlier_by_column: Dict[str, int] = field(default_factory=dict)
    constant_columns: List[str] = field(default_factory=list)
    highly_correlated: List[tuple] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'overall_score': self.overall_score,
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'missing_count': self.missing_count,
            'missing_percent': self.missing_percent,
            'duplicate_rows': self.duplicate_rows,
            'duplicate_percent': self.duplicate_percent,
            'outlier_count': self.outlier_count,
            'outlier_percent': self.outlier_percent,
            'constant_columns': self.constant_columns,
            'recommendations': self.recommendations,
        }


class DataQualityAnalyzer:
    """
    データ品質分析エンジン
    
    Features:
    - 欠損値検出・分析
    - 異常値検出（IQR/Z-score）
    - 重複行検出
    - 定数カラム検出
    - 高相関ペア検出
    - 品質スコア算出
    
    Example:
        >>> analyzer = DataQualityAnalyzer()
        >>> report = analyzer.analyze(df)
        >>> print(f"Quality Score: {report.overall_score}")
    """
    
    def __init__(
        self,
        outlier_method: str = 'iqr',
        outlier_threshold: float = 1.5,
        correlation_threshold: float = 0.95,
        missing_threshold: float = 0.5,
    ):
        """
        Args:
            outlier_method: 'iqr' or 'zscore'
            outlier_threshold: IQR係数 or Z-score閾値
            correlation_threshold: 高相関とみなす閾値
            missing_threshold: 欠損率がこれを超えるとカラム削除推奨
        """
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.correlation_threshold = correlation_threshold
        self.missing_threshold = missing_threshold
    
    def analyze(self, df: pd.DataFrame) -> QualityReport:
        """
        データフレームの品質を分析
        
        Args:
            df: 分析対象DataFrame
            
        Returns:
            QualityReport
        """
        n_samples, n_features = df.shape
        
        # 1. 欠損値分析
        missing_count, missing_percent, missing_by_col = self._analyze_missing(df)
        
        # 2. 重複行分析
        dup_count, dup_percent = self._analyze_duplicates(df)
        
        # 3. 異常値分析
        outlier_count, outlier_percent, outlier_by_col = self._analyze_outliers(df)
        
        # 4. 定数カラム
        const_cols = self._find_constant_columns(df)
        
        # 5. 高相関ペア
        high_corr = self._find_high_correlation(df)
        
        # 6. 品質スコア計算
        score = self._calculate_score(
            missing_percent, dup_percent, outlier_percent,
            len(const_cols) / max(n_features, 1),
        )
        
        # 7. 推奨アクション
        recommendations = self._generate_recommendations(
            missing_by_col, dup_percent, const_cols, high_corr,
        )
        
        return QualityReport(
            overall_score=score,
            n_samples=n_samples,
            n_features=n_features,
            missing_count=missing_count,
            missing_percent=missing_percent,
            missing_by_column=missing_by_col,
            duplicate_rows=dup_count,
            duplicate_percent=dup_percent,
            outlier_count=outlier_count,
            outlier_percent=outlier_percent,
            outlier_by_column=outlier_by_col,
            constant_columns=const_cols,
            highly_correlated=high_corr,
            recommendations=recommendations,
        )
    
    def _analyze_missing(self, df: pd.DataFrame) -> tuple:
        """欠損値分析"""
        missing = df.isnull().sum()
        total_cells = df.size
        total_missing = missing.sum()
        
        missing_by_col = {
            col: float(count / len(df))
            for col, count in missing.items()
            if count > 0
        }
        
        return (
            int(total_missing),
            float(total_missing / total_cells) if total_cells > 0 else 0,
            missing_by_col,
        )
    
    def _analyze_duplicates(self, df: pd.DataFrame) -> tuple:
        """重複行分析"""
        dup_count = df.duplicated().sum()
        dup_percent = float(dup_count / len(df)) if len(df) > 0 else 0
        return int(dup_count), dup_percent
    
    def _analyze_outliers(self, df: pd.DataFrame) -> tuple:
        """異常値分析"""
        numeric_df = df.select_dtypes(include=[np.number])
        outlier_by_col = {}
        total_outliers = 0
        
        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            if len(series) == 0:
                continue
            
            if self.outlier_method == 'iqr':
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - self.outlier_threshold * IQR
                upper = Q3 + self.outlier_threshold * IQR
                outliers = ((series < lower) | (series > upper)).sum()
            else:  # zscore
                z = np.abs(stats.zscore(series))
                outliers = (z > self.outlier_threshold).sum()
            
            if outliers > 0:
                outlier_by_col[col] = int(outliers)
                total_outliers += outliers
        
        total_cells = numeric_df.size
        return (
            int(total_outliers),
            float(total_outliers / total_cells) if total_cells > 0 else 0,
            outlier_by_col,
        )
    
    def _find_constant_columns(self, df: pd.DataFrame) -> List[str]:
        """定数カラムを検出"""
        return [col for col in df.columns if df[col].nunique() <= 1]
    
    def _find_high_correlation(self, df: pd.DataFrame) -> List[tuple]:
        """高相関ペアを検出"""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return []
        
        corr_matrix = numeric_df.corr().abs()
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > self.correlation_threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        float(corr_matrix.iloc[i, j]),
                    ))
        
        return high_corr_pairs
    
    def _calculate_score(
        self,
        missing_pct: float,
        dup_pct: float,
        outlier_pct: float,
        const_pct: float,
    ) -> float:
        """品質スコア計算（0-100）"""
        # 各要素のペナルティ
        missing_penalty = missing_pct * 50
        dup_penalty = dup_pct * 30
        outlier_penalty = outlier_pct * 20
        const_penalty = const_pct * 10
        
        score = 100 - (missing_penalty + dup_penalty + outlier_penalty + const_penalty)
        return max(0, min(100, score))
    
    def _generate_recommendations(
        self,
        missing_by_col: Dict[str, float],
        dup_pct: float,
        const_cols: List[str],
        high_corr: List[tuple],
    ) -> List[str]:
        """推奨アクションを生成"""
        recommendations = []
        
        # 欠損率が高いカラム
        high_missing = [
            col for col, pct in missing_by_col.items()
            if pct > self.missing_threshold
        ]
        if high_missing:
            recommendations.append(
                f"欠損率50%超のカラム削除推奨: {', '.join(high_missing[:5])}"
            )
        
        # 重複
        if dup_pct > 0.01:
            recommendations.append(
                f"重複行が{dup_pct*100:.1f}%存在。drop_duplicates()推奨。"
            )
        
        # 定数カラム
        if const_cols:
            recommendations.append(
                f"情報量なしカラム削除推奨: {', '.join(const_cols[:5])}"
            )
        
        # 高相関
        if high_corr:
            recommendations.append(
                f"高相関ペア({len(high_corr)}組)検出。VIF削除またはPCA推奨。"
            )
        
        if not recommendations:
            recommendations.append("データ品質は良好です。")
        
        return recommendations
    
    def get_summary_html(self, report: QualityReport) -> str:
        """品質レポートのHTML生成"""
        score_color = "green" if report.overall_score >= 80 else "orange" if report.overall_score >= 60 else "red"
        
        html = f"""
        <div style="font-family: Arial, sans-serif; padding: 20px;">
            <h2>データ品質レポート</h2>
            <div style="display: flex; gap: 20px; margin: 20px 0;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            color: white; padding: 30px; border-radius: 10px; text-align: center;">
                    <div style="font-size: 3em; font-weight: bold;">{report.overall_score:.0f}</div>
                    <div>品質スコア (0-100)</div>
                </div>
                <div style="flex: 1;">
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr><td>サンプル数</td><td>{report.n_samples:,}</td></tr>
                        <tr><td>特徴量数</td><td>{report.n_features}</td></tr>
                        <tr><td>欠損率</td><td>{report.missing_percent*100:.1f}%</td></tr>
                        <tr><td>重複率</td><td>{report.duplicate_percent*100:.1f}%</td></tr>
                        <tr><td>異常値率</td><td>{report.outlier_percent*100:.1f}%</td></tr>
                    </table>
                </div>
            </div>
            <h3>推奨アクション</h3>
            <ul>
                {"".join(f"<li>{r}</li>" for r in report.recommendations)}
            </ul>
        </div>
        """
        return html
