"""
分子スコアリングエンジン

Implements: F-SCORE-001
設計思想:
- QED（定量的推定薬物適性）
- SA Score（合成容易性）
- Lipinski's Rule of Five
- PAINS フィルタ

参考文献:
- QED: Bickerton, G. R., et al. (2012). Nature Chemistry, 4(2), 90-98.
- SA Score: Ertl, P., & Schuffenhauer, A. (2009). J. Cheminform., 1(1), 8.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MoleculeScorer:
    """
    分子スコアリングエンジン
    
    Features:
    - QED: 薬品適性スコア（0-1、高いほど良い）
    - SA Score: 合成容易性（1-10、低いほど良い）
    - Lipinski: 医薬品らしさの5つのルール
    - PAINS: 不適切な化合物フィルタ
    
    Example:
        >>> scorer = MoleculeScorer()
        >>> score = scorer.score_molecule('CCO')
        >>> print(score['qed'])
        0.45
    """
    
    def __init__(self):
        """初期化"""
        self.cache = {}
    
    def calculate_qed(self, smiles: str) -> Optional[float]:
        """
        QED（Quantitative Estimation of Drug-likeness）計算
        
        論文引用:
            Bickerton, G. R., et al. (2012).
            "Quantifying the chemical beauty of drugs"
            Nature Chemistry, 4(2), 90-98.
            
            "QED provides a measure of drug-likeness based on the concept 
            of desirability functions for eight molecular properties."
            
            日本語訳：
            「QEDは8つの分子特性に対する望ましさ関数の概念に基づいて
            薬品適性の尺度を提供する。」
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            QEDスコア（0-1）、高いほど薬品らしい
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import QED as rdkit_qed
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            qed = rdkit_qed.qed(mol)
            return float(qed)
            
        except ImportError:
            logger.warning("RDKit not available for QED calculation")
            return None
        except Exception as e:
            logger.error(f"QED calculation failed: {e}")
            return None
    
    def calculate_sa_score(self, smiles: str) -> Optional[float]:
        """
        SA Score（Synthetic Accessibility Score）計算
        
        論文引用:
            Ertl, P., & Schuffenhauer, A. (2009).
            "Estimation of synthetic accessibility score of drug-like molecules"
            J. Cheminform., 1(1), 8.
            
            "SA score estimates the ease of synthesis, ranging from 1 (easy)
            to 10 (difficult), based on molecular complexity and fragment contributions."
            
            日本語訳：
            「SAスコアは分子の複雑さとフラグメント寄与に基づいて、
            合成の容易さを1（容易）から10（困難）の範囲で推定する。」
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            SAスコア（1-10）、低いほど合成が容易
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import rdMolDescriptors
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # 簡易版SA Score（RDKitに組み込みがない場合の近似）
            # 実際のSAスコアは複雑なので、ここでは分子の複雑さから推定
            num_atoms = mol.GetNumAtoms()
            num_bonds = mol.GetNumBonds()
            num_rings = rdMolDescriptors.CalcNumRings(mol)
            
            # 簡易計算（実際のSAスコアとは異なる）
            complexity = (num_atoms * 0.1 + num_bonds * 0.05 + num_rings * 0.5)
            sa_score = min(10.0, max(1.0, complexity / 2))
            
            return float(sa_score)
            
        except ImportError:
            logger.warning("RDKit not available for SA Score calculation")
            return None
        except Exception as e:
            logger.error(f"SA Score calculation failed: {e}")
            return None
    
    def check_lipinski(self, smiles: str) -> Optional[Dict[str, bool]]:
        """
        Lipinski's Rule of Five適合性チェック
        
        ルール:
        - MW ≤ 500 Da
        - logP ≤ 5
        - HBD ≤ 5（水素結合ドナー）
        - HBA ≤ 10（水素結合アクセプター）
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            各ルールの適合状況
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Lipinski
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            
            return {
                'mw_pass': mw <= 500,
                'logp_pass': logp <= 5,
                'hbd_pass': hbd <= 5,
                'hba_pass': hba <= 10,
                'all_pass': all([mw <= 500, logp <= 5, hbd <= 5, hba <= 10]),
                'values': {
                    'mw': float(mw),
                    'logp': float(logp),
                    'hbd': int(hbd),
                    'hba': int(hba),
                }
            }
            
        except ImportError:
            logger.warning("RDKit not available for Lipinski check")
            return None
        except Exception as e:
            logger.error(f"Lipinski check failed: {e}")
            return None
    
    def check_pains(self, smiles: str) -> Optional[bool]:
        """
        PAINS（Pan Assay INterference compoundS）フィルタ
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            True=クリーン、False=PAINS検出
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import FilterCatalog
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # PAINSフィルタ
            params = FilterCatalog.FilterCatalogParams()
            params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
            catalog = FilterCatalog.FilterCatalog(params)
            
            entry = catalog.GetFirstMatch(mol)
            
            # エントリーがない = クリーン
            return entry is None
            
        except (ImportError, AttributeError):
            logger.warning("RDKit PAINS filter not available")
            return None
        except Exception as e:
            logger.error(f"PAINS check failed: {e}")
            return None
    
    def score_molecule(self, smiles: str) -> Dict[str, Any]:
        """
        総合スコアリング
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            全スコア辞書
        """
        # QED
        qed = self.calculate_qed(smiles)
        
        # SA Score
        sa_score = self.calculate_sa_score(smiles)
        
        # Lipinski
        lipinski = self.check_lipinski(smiles)
        
        # PAINS
        pains_free = self.check_pains(smiles)
        
        # 総合スコア計算（0-1）
        overall_score = 0.0
        if qed is not None and sa_score is not None and lipinski is not None:
            # QED: 0-1（高い方が良い）
            # SA Score: 1-10（低い方が良い） → 正規化: (10 - sa) / 9
            # Lipinski: all_pass → 1.0 or 0.0
            
            qed_norm = qed
            sa_norm = (10.0 - sa_score) / 9.0
            lipinski_norm = 1.0 if lipinski['all_pass'] else 0.5
            
            overall_score = (qed_norm * 0.4 + sa_norm * 0.3 + lipinski_norm * 0.3)
        
        return {
            'smiles': smiles,
            'qed': qed,
            'sa_score': sa_score,
            'lipinski': lipinski,
            'pains_free': pains_free,
            'overall_score': overall_score,
            'valid': qed is not None,
        }
    
    def rank_molecules(
        self,
        smiles_list: list[str],
        criterion: str = 'overall'
    ) -> list[tuple[str, float]]:
        """
        分子リストをランキング
        
        Args:
            smiles_list: SMILESリスト
            criterion: 'overall', 'qed', 'sa_score'
        
        Returns:
            [(smiles, score), ...]（降順）
        """
        scored = []
        
        for smiles in smiles_list:
            result = self.score_molecule(smiles)
            
            if criterion == 'overall':
                score = result['overall_score']
            elif criterion == 'qed':
                score = result.get('qed', 0.0) or 0.0
            elif criterion == 'sa_score':
                # SA Scoreは低い方が良いので反転
                sa = result.get('sa_score', 10.0) or 10.0
                score = (10.0 - sa) / 9.0
            else:
                score = result['overall_score']
            
            scored.append((smiles, score))
        
        # 降順ソート
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored
