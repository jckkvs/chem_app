"""
生成分子の検証

Implements: F-GEN-002
設計思想:
- SMILES妥当性チェック
- 医薬品らしさ（QED）
- Lipinski則
- 合成可能性（SA Score）

引用:
- Lipinski, C. A., et al. (2001). Advanced Drug Delivery Reviews
- Bickerton, G. R., et al. (2012). Nature Chemistry (QED)
- Ertl, P., & Schuffenhauer, A. (2009). Journal of Cheminformatics (SA Score)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """検証結果"""
    is_valid: bool
    smiles: str
    error: Optional[str] = None
    qed_score: Optional[float] = None
    lipinski_violations: int = 0
    sa_score: Optional[float] = None
    molecular_weight: Optional[float] = None
    logp: Optional[float] = None


class GeneratedMoleculeValidator:
    """
    生成分子の検証
    
    Features:
    - SMILES妥当性
    - 医薬品らしさ（QED）
    - Lipinski則（Rule of Five）
    - 合成可能性
    
    Example:
        >>> validator = GeneratedMoleculeValidator()
        >>> result = validator.validate("CCO")
        >>> print(result.is_valid)
        True
    """
    
    def __init__(self, strict: bool = False):
        """
        初期化
        
        Args:
            strict: 厳密モード（QED/SAスコア必須）
        """
        self.strict = strict
        self._rdkit_available = None
    
    def _check_rdkit(self) -> bool:
        """RDKit利用可能性チェック"""
        if self._rdkit_available is None:
            try:
                from rdkit import Chem
                self._rdkit_available = True
            except ImportError:
                logger.warning("RDKit not available. Validation will be limited.")
                self._rdkit_available = False
        
        return self._rdkit_available
    
    def validate(self, smiles: str) -> ValidationResult:
        """
        包括的検証
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            検証結果
        
        Example:
            >>> result = validator.validate("CCO")
            >>> if result.is_valid:
            ...     print(f"QED: {result.qed_score}")
        """
        # 基本チェック
        if not smiles or not isinstance(smiles, str):
            return ValidationResult(
                is_valid=False,
                smiles=smiles or "",
                error="Invalid SMILES: empty or not a string"
            )
        
        # RDKit利用不可の場合は簡易チェック
        if not self._check_rdkit():
            return self._simple_validation(smiles)
        
        # RDKitによる完全検証
        return self._full_validation(smiles)
    
    def _simple_validation(self, smiles: str) -> ValidationResult:
        """簡易検証（RDKitなし）"""
        # 基本的な文字チェック
        valid_chars = set('CNOPSFClBrIcnops()[]=#-+@/\\0123456789%')
        invalid_chars = set(smiles) - valid_chars
        
        if invalid_chars:
            return ValidationResult(
                is_valid=False,
                smiles=smiles,
                error=f"Invalid characters: {invalid_chars}"
            )
        
        return ValidationResult(
            is_valid=True,
            smiles=smiles
        )
    
    def _full_validation(self, smiles: str) -> ValidationResult:
        """完全検証（RDKitあり）"""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, QED
            
            # 分子パース
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return ValidationResult(
                    is_valid=False,
                    smiles=smiles,
                    error="Invalid SMILES: cannot parse"
                )
            
            # 基本記述子
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            
            # QEDスコア
            qed_score = QED.qed(mol)
            
            # Lipinski則チェック
            lipinski_violations = self._check_lipinski(mol)
            
            # 合成可能性スコア（オプション）
            sa_score = self._calculate_sa_score(mol)
            
            # 厳密モードでの追加チェック
            if self.strict:
                if qed_score < 0.3:
                    return ValidationResult(
                        is_valid=False,
                        smiles=smiles,
                        error=f"QED score too low: {qed_score:.2f}",
                        qed_score=qed_score,
                        lipinski_violations=lipinski_violations,
                        sa_score=sa_score,
                        molecular_weight=mw,
                        logp=logp
                    )
            
            return ValidationResult(
                is_valid=True,
                smiles=smiles,
                qed_score=qed_score,
                lipinski_violations=lipinski_violations,
                sa_score=sa_score,
                molecular_weight=mw,
                logp=logp
            )
        
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return ValidationResult(
                is_valid=False,
                smiles=smiles,
                error=f"Validation exception: {str(e)}"
            )
    
    def _check_lipinski(self, mol) -> int:
        """
        Lipinski則チェック
        
        Rule of Five:
        1. MW <= 500
        2. logP <= 5
        3. HBD <= 5
        4. HBA <= 10
        
        Returns:
            違反数（0-4）
        """
        from rdkit.Chem import Descriptors
        
        violations = 0
        
        if Descriptors.MolWt(mol) > 500:
            violations += 1
        
        if Descriptors.MolLogP(mol) > 5:
            violations += 1
        
        if Descriptors.NumHDonors(mol) > 5:
            violations += 1
        
        if Descriptors.NumHAcceptors(mol) > 10:
            violations += 1
        
        return violations
    
    def _calculate_sa_score(self, mol) -> Optional[float]:
        """
        合成可能性スコア計算
        
        Returns:
            SAスコア（1-10、低いほど合成容易）
        
        Note:
            完全な実装には別途SAScoreモジュールが必要。
            ここでは簡易版を提供。
        """
        try:
            # 簡易版：リング数とヘテロ原子数から推定
            from rdkit.Chem import Descriptors
            
            n_rings = Descriptors.RingCount(mol)
            n_hetero = Descriptors.NumHeteroatoms(mol)
            complexity = Descriptors.BertzCT(mol)
            
            # 簡易スコア（実際のSAScoreとは異なる）
            score = 1.0 + (n_rings * 0.5) + (n_hetero * 0.3) + (complexity / 1000)
            score = min(10.0, max(1.0, score))
            
            return score
        
        except Exception:
            return None
    
    def is_drug_like(self, smiles: str) -> bool:
        """
        医薬品らしさ判定
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            医薬品らしさ（True/False）
        
        Criteria:
        - Valid SMILES
        - QED >= 0.5
        - Lipinski violations <= 1
        """
        result = self.validate(smiles)
        
        if not result.is_valid:
            return False
        
        if result.qed_score is not None and result.qed_score < 0.5:
            return False
        
        if result.lipinski_violations > 1:
            return False
        
        return True
