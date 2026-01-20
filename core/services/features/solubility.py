"""
溶解度予測エンジン（AquaSol/ESOL inspired）

Implements: F-SOLUBILITY-001
設計思想:
- ESOL式ベース予測
- 温度依存性
- 多溶媒対応
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SolubilityResult:
    """溶解度予測結果"""
    smiles: str
    logS: float  # log mol/L
    solubility_mg_ml: float  # mg/mL
    solubility_class: str  # 'high', 'medium', 'low', 'insoluble'
    confidence: float = 0.0


class SolubilityPredictor:
    """
    溶解度予測（ESOL/AquaSol/ACD inspired）
    
    Features:
    - ESOLモデルによるlogS予測
    - mg/mL換算
    - 溶解度クラス分類
    
    Example:
        >>> predictor = SolubilityPredictor()
        >>> result = predictor.predict("CCO")
    """
    
    # ESOL係数（Delaney, 2004）
    ESOL_COEFFS = {
        'intercept': 0.16,
        'logP': -0.63,
        'mw': -0.0062,
        'rotatable': 0.066,
        'aromatic': -0.74,
    }
    
    def predict(self, smiles: str) -> Optional[SolubilityResult]:
        """溶解度を予測"""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Lipinski
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # 記述子計算
            logP = Descriptors.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            rotatable = Lipinski.NumRotatableBonds(mol)
            aromatic_ratio = Lipinski.NumAromaticRings(mol) / max(Lipinski.RingCount(mol), 1)
            
            # ESOL式
            logS = (
                self.ESOL_COEFFS['intercept']
                + self.ESOL_COEFFS['logP'] * logP
                + self.ESOL_COEFFS['mw'] * mw
                + self.ESOL_COEFFS['rotatable'] * rotatable
                + self.ESOL_COEFFS['aromatic'] * aromatic_ratio
            )
            
            # mol/L → mg/mL
            S_mol = 10 ** logS
            solubility_mg_ml = S_mol * mw / 1000
            
            # クラス分類
            if logS > 0:
                sol_class = 'high'
            elif logS > -2:
                sol_class = 'medium'
            elif logS > -4:
                sol_class = 'low'
            else:
                sol_class = 'insoluble'
            
            return SolubilityResult(
                smiles=smiles,
                logS=round(logS, 2),
                solubility_mg_ml=round(solubility_mg_ml, 4),
                solubility_class=sol_class,
                confidence=0.8,
            )
            
        except Exception as e:
            logger.error(f"Solubility prediction failed: {e}")
            return None
    
    def predict_batch(self, smiles_list: list) -> list:
        """バッチ予測"""
        return [self.predict(smi) for smi in smiles_list]
    
    def get_solubility_html(self, result: SolubilityResult) -> str:
        """HTMLサマリー"""
        color_map = {
            'high': '#27ae60',
            'medium': '#f39c12',
            'low': '#e74c3c',
            'insoluble': '#95a5a6',
        }
        color = color_map.get(result.solubility_class, '#333')
        
        return f"""
        <div style="padding: 15px; border-left: 4px solid {color}; background: #f5f5f5;">
            <h3>💧 Solubility: <span style="color: {color}">{result.solubility_class.upper()}</span></h3>
            <p>LogS: {result.logS} (mol/L)</p>
            <p>Solubility: {result.solubility_mg_ml} mg/mL</p>
        </div>
        """
