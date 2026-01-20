"""
ADMET予測エンジン（Optibrium StarDrop inspired）

Implements: F-ADMET-001
設計思想:
- 吸収、分布、代謝、排泄、毒性予測
- マルチターゲット予測
- 警告フラグシステム
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ADMETProfile:
    """ADMET プロファイル"""
    smiles: str
    
    # 吸収 (Absorption)
    caco2_permeability: Optional[float] = None  # cm/s
    hia: Optional[float] = None  # Human Intestinal Absorption %
    bioavailability: Optional[float] = None  # F%
    pgp_substrate: Optional[bool] = None
    
    # 分布 (Distribution)
    vdss: Optional[float] = None  # Volume of Distribution L/kg
    ppb: Optional[float] = None  # Plasma Protein Binding %
    bbb_penetration: Optional[float] = None  # Blood-Brain Barrier
    
    # 代謝 (Metabolism)
    cyp2d6_inhibitor: Optional[bool] = None
    cyp3a4_inhibitor: Optional[bool] = None
    cyp2c9_inhibitor: Optional[bool] = None
    
    # 排泄 (Excretion)
    clearance: Optional[float] = None  # mL/min/kg
    half_life: Optional[float] = None  # hours
    
    # 毒性 (Toxicity)
    ames_mutagenicity: Optional[bool] = None
    herg_inhibition: Optional[bool] = None
    hepatotoxicity: Optional[bool] = None
    ld50: Optional[float] = None  # mg/kg
    
    # 警告フラグ
    alerts: List[str] = None
    
    def __post_init__(self):
        if self.alerts is None:
            self.alerts = []
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ADMETPredictor:
    """
    ADMET予測エンジン（StarDrop/pkCSM inspired）
    
    Features:
    - 記述子ベースの簡易ADMET予測
    - 構造アラート検出
    - リスクスコア算出
    
    Example:
        >>> predictor = ADMETPredictor()
        >>> profile = predictor.predict("CCO")
    """
    
    def __init__(self):
        # PAINS/Toxicophore パターン（実際の実装では完全版を使用）
        self.toxic_patterns = [
            ("[OH]c1ccc([N+](=O)[O-])cc1", "Nitrophenol"),
            ("[#6]1[#6][#6][#6]([F,Cl,Br,I])[#6][#6]1[F,Cl,Br,I]", "Polyhalogen"),
            ("C(=O)Cl", "Acyl halide"),
            ("[N;R0]=[N;R0]", "Azo compound"),
        ]
    
    def predict(self, smiles: str) -> Optional[ADMETProfile]:
        """ADMET予測"""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Lipinski
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            profile = ADMETProfile(smiles=smiles)
            
            # 記述子計算
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            rotatable = Lipinski.NumRotatableBonds(mol)
            
            # 吸収予測（経験的モデル）
            profile.hia = self._predict_hia(logp, tpsa, mw)
            profile.caco2_permeability = self._predict_caco2(tpsa, mw)
            profile.bioavailability = self._predict_bioavailability(mw, logp, hbd)
            profile.pgp_substrate = mw > 400 and tpsa > 75
            
            # 分布予測
            profile.bbb_penetration = self._predict_bbb(logp, tpsa, hbd)
            profile.ppb = self._predict_ppb(logp, mw)
            profile.vdss = self._predict_vdss(logp, mw)
            
            # 代謝予測（簡易ルールベース）
            profile.cyp3a4_inhibitor = logp > 3 and mw > 300
            profile.cyp2d6_inhibitor = self._has_basic_nitrogen(mol)
            
            # 毒性予測
            profile.herg_inhibition = logp > 3.5 and self._has_basic_nitrogen(mol)
            profile.alerts = self._detect_alerts(mol)
            profile.ames_mutagenicity = len(profile.alerts) > 2
            
            return profile
            
        except Exception as e:
            logger.error(f"ADMET prediction failed: {e}")
            return None
    
    def _predict_hia(self, logp: float, tpsa: float, mw: float) -> float:
        """Human Intestinal Absorption %"""
        # 簡易モデル
        score = 100 - (tpsa / 2)
        score = min(100, max(0, score))
        if mw > 500:
            score *= 0.8
        return round(score, 1)
    
    def _predict_caco2(self, tpsa: float, mw: float) -> float:
        """Caco-2 透過性 (log cm/s)"""
        return round(-5 - (tpsa / 100) - (mw / 1000), 2)
    
    def _predict_bioavailability(self, mw: float, logp: float, hbd: int) -> float:
        """経口バイオアベイラビリティ %"""
        score = 100
        if mw > 500:
            score -= 20
        if logp > 5 or logp < 0:
            score -= 15
        if hbd > 5:
            score -= 20
        return max(0, min(100, score))
    
    def _predict_bbb(self, logp: float, tpsa: float, hbd: int) -> float:
        """BBB透過性 (log ratio)"""
        return round(0.3 * logp - 0.02 * tpsa - 0.2 * hbd, 2)
    
    def _predict_ppb(self, logp: float, mw: float) -> float:
        """血漿蛋白結合率 %"""
        ppb = 50 + 10 * logp
        return round(min(99, max(10, ppb)), 1)
    
    def _predict_vdss(self, logp: float, mw: float) -> float:
        """定常状態分布容積 L/kg"""
        return round(0.5 + 0.3 * logp, 2)
    
    def _has_basic_nitrogen(self, mol) -> bool:
        """塩基性窒素の存在チェック"""
        from rdkit import Chem
        pattern = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O)]")
        return mol.HasSubstructMatch(pattern) if pattern else False
    
    def _detect_alerts(self, mol) -> List[str]:
        """構造アラート検出"""
        from rdkit import Chem
        
        alerts = []
        for smarts, name in self.toxic_patterns:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                alerts.append(name)
        
        return alerts
    
    def get_risk_score(self, profile: ADMETProfile) -> Dict[str, float]:
        """リスクスコア算出"""
        scores = {}
        
        # 吸収リスク
        if profile.hia:
            scores['absorption'] = profile.hia / 100
        
        # BBBリスク（CNS薬の場合は高い方がよい）
        if profile.bbb_penetration:
            scores['cns_penetration'] = min(1, max(0, (profile.bbb_penetration + 1) / 2))
        
        # 毒性リスク
        tox_score = 1.0
        if profile.alerts:
            tox_score -= 0.2 * len(profile.alerts)
        if profile.herg_inhibition:
            tox_score -= 0.3
        scores['safety'] = max(0, tox_score)
        
        scores['overall'] = np.mean(list(scores.values()))
        
        return scores
