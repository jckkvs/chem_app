"""
データ品質チェック

Implements: F-QUALITY-001
設計思想:
- 分子データの品質評価
- 異常値・外れ値検出
- データクリーニング推奨

機能:
- SMILES妥当性検証
- 分子特性の分布チェック
- 重複検出
- 欠損値分析
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """データ品質レポート"""
    total_samples: int
    valid_samples: int
    invalid_samples: int
    duplicate_count: int
    missing_values: Dict[str, int]
    outlier_count: int
    issues: List[Dict[str, Any]]
    recommendations: List[str]
    
    @property
    def validity_rate(self) -> float:
        return self.valid_samples / self.total_samples if self.total_samples > 0 else 0.0
    
    @property
    def quality_score(self) -> float:
        """品質スコア (0-100)"""
        score = 100.0
        
        # 無効データのペナルティ
        score -= (1 - self.validity_rate) * 30
        
        # 重複のペナルティ
        dup_rate = self.duplicate_count / self.total_samples if self.total_samples > 0 else 0
        score -= dup_rate * 20
        
        # 外れ値のペナルティ
        outlier_rate = self.outlier_count / self.total_samples if self.total_samples > 0 else 0
        score -= outlier_rate * 10
        
        return max(0, score)
    
    def summary(self) -> str:
        return (
            f"Quality Score: {self.quality_score:.1f}/100\n"
            f"Valid: {self.valid_samples}/{self.total_samples} ({self.validity_rate:.1%})\n"
            f"Duplicates: {self.duplicate_count}\n"
            f"Outliers: {self.outlier_count}\n"
            f"Issues: {len(self.issues)}"
        )


class DataQualityChecker:
    """
    データ品質チェッカー
    
    Usage:
        checker = DataQualityChecker()
        report = checker.check(smiles_list, target_values)
        print(report.summary())
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Args:
            strict_mode: 厳格モード（警告もエラーとして扱う）
        """
        self.strict_mode = strict_mode
    
    def check(
        self,
        smiles_list: List[str],
        target_values: np.ndarray = None,
        feature_df: pd.DataFrame = None,
    ) -> QualityReport:
        """
        データ品質をチェック
        
        Args:
            smiles_list: SMILESリスト
            target_values: ターゲット値
            feature_df: 特徴量データフレーム
            
        Returns:
            QualityReport
        """
        issues = []
        recommendations = []
        
        total = len(smiles_list)
        
        # SMILES妥当性チェック
        valid_count, invalid_indices = self._check_smiles_validity(smiles_list)
        if invalid_indices:
            issues.append({
                'type': 'invalid_smiles',
                'count': len(invalid_indices),
                'indices': invalid_indices[:10],  # 最初の10件
            })
            recommendations.append(
                f"Remove or fix {len(invalid_indices)} invalid SMILES"
            )
        
        # 重複チェック
        duplicates = self._check_duplicates(smiles_list)
        if duplicates:
            issues.append({
                'type': 'duplicates',
                'count': len(duplicates),
                'examples': list(duplicates)[:5],
            })
            recommendations.append(
                f"Remove {len(duplicates)} duplicate SMILES"
            )
        
        # ターゲット値チェック
        outlier_count = 0
        missing_values = {}
        
        if target_values is not None:
            # 欠損値
            nan_count = np.isnan(target_values).sum()
            if nan_count > 0:
                missing_values['target'] = int(nan_count)
                issues.append({
                    'type': 'missing_target',
                    'count': nan_count,
                })
                recommendations.append(
                    f"Handle {nan_count} missing target values"
                )
            
            # 外れ値
            outlier_indices = self._detect_outliers(target_values)
            outlier_count = len(outlier_indices)
            if outlier_count > 0:
                issues.append({
                    'type': 'outliers',
                    'count': outlier_count,
                    'indices': outlier_indices[:10],
                })
                if outlier_count > total * 0.05:
                    recommendations.append(
                        f"Review {outlier_count} potential outliers"
                    )
        
        # 特徴量チェック
        if feature_df is not None:
            # 欠損値
            for col in feature_df.columns:
                nan_count = feature_df[col].isna().sum()
                if nan_count > 0:
                    missing_values[col] = int(nan_count)
            
            # 高相関チェック
            high_corr = self._check_high_correlation(feature_df)
            if high_corr:
                issues.append({
                    'type': 'high_correlation',
                    'pairs': high_corr[:10],
                })
                recommendations.append(
                    f"Consider removing {len(high_corr)} highly correlated features"
                )
        
        return QualityReport(
            total_samples=total,
            valid_samples=valid_count,
            invalid_samples=total - valid_count,
            duplicate_count=len(duplicates),
            missing_values=missing_values,
            outlier_count=outlier_count,
            issues=issues,
            recommendations=recommendations,
        )
    
    def _check_smiles_validity(
        self,
        smiles_list: List[str],
    ) -> Tuple[int, List[int]]:
        """SMILESの妥当性チェック"""
        try:
            from rdkit import Chem
        except ImportError:
            return len(smiles_list), []
        
        valid_count = 0
        invalid_indices = []
        
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                valid_count += 1
            else:
                invalid_indices.append(i)
        
        return valid_count, invalid_indices
    
    def _check_duplicates(
        self,
        smiles_list: List[str],
    ) -> set:
        """重複チェック"""
        counter = Counter(smiles_list)
        return {smi for smi, count in counter.items() if count > 1}
    
    def _detect_outliers(
        self,
        values: np.ndarray,
        method: str = 'iqr',
    ) -> List[int]:
        """外れ値検出"""
        values = np.asarray(values)
        valid_values = values[~np.isnan(values)]
        
        if len(valid_values) < 4:
            return []
        
        if method == 'iqr':
            q1 = np.percentile(valid_values, 25)
            q3 = np.percentile(valid_values, 75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            
            outlier_mask = (values < lower) | (values > upper)
        else:  # zscore
            mean = np.nanmean(values)
            std = np.nanstd(values)
            z_scores = np.abs((values - mean) / std)
            outlier_mask = z_scores > 3
        
        return list(np.where(outlier_mask)[0])
    
    def _check_high_correlation(
        self,
        df: pd.DataFrame,
        threshold: float = 0.95,
    ) -> List[Tuple[str, str, float]]:
        """高相関特徴量チェック"""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return []
        
        corr = numeric_df.corr().abs()
        
        # 上三角のみ
        upper = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool)
        )
        
        high_corr = []
        for col in upper.columns:
            for idx in upper.index:
                if upper.loc[idx, col] > threshold:
                    high_corr.append((idx, col, upper.loc[idx, col]))
        
        return high_corr
    
    def clean_data(
        self,
        smiles_list: List[str],
        target_values: np.ndarray = None,
        remove_invalid: bool = True,
        remove_duplicates: bool = True,
        remove_outliers: bool = False,
    ) -> Tuple[List[str], Optional[np.ndarray]]:
        """
        データをクリーニング
        
        Args:
            smiles_list: SMILESリスト
            target_values: ターゲット値
            remove_invalid: 無効SMILESを削除
            remove_duplicates: 重複を削除
            remove_outliers: 外れ値を削除
            
        Returns:
            (cleaned_smiles, cleaned_targets)
        """
        indices_to_keep = set(range(len(smiles_list)))
        
        # 無効SMILES
        if remove_invalid:
            _, invalid = self._check_smiles_validity(smiles_list)
            indices_to_keep -= set(invalid)
        
        # 重複
        if remove_duplicates:
            seen = {}
            for i, smi in enumerate(smiles_list):
                if smi in seen:
                    indices_to_keep.discard(i)
                else:
                    seen[smi] = i
        
        # 外れ値
        if remove_outliers and target_values is not None:
            outliers = self._detect_outliers(target_values)
            indices_to_keep -= set(outliers)
        
        # フィルタリング
        indices = sorted(indices_to_keep)
        cleaned_smiles = [smiles_list[i] for i in indices]
        
        cleaned_targets = None
        if target_values is not None:
            cleaned_targets = target_values[indices]
        
        logger.info(
            f"Cleaned: {len(smiles_list)} -> {len(cleaned_smiles)} samples"
        )
        
        return cleaned_smiles, cleaned_targets


def check_data_quality(
    smiles_list: List[str],
    target_values: np.ndarray = None,
) -> QualityReport:
    """便利関数: データ品質チェック"""
    checker = DataQualityChecker()
    return checker.check(smiles_list, target_values)


def clean_molecular_data(
    smiles_list: List[str],
    target_values: np.ndarray = None,
) -> Tuple[List[str], Optional[np.ndarray]]:
    """便利関数: データクリーニング"""
    checker = DataQualityChecker()
    return checker.clean_data(smiles_list, target_values)
