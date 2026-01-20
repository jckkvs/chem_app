"""
pKa/LogD予測（ChemAxon inspired）

Implements: F-PKA-001
設計思想:
- イオン化状態予測
- pH依存LogD計算
- プロトン化状態
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IonizationState:
    """イオン化状態"""
    smiles: str
    pka_values: List[float] = field(default_factory=list)
    pkb_values: List[float] = field(default_factory=list)
    logp: float = 0.0
    logd_at_pH7: float = 0.0
    ionizable_groups: List[Dict[str, Any]] = field(default_factory=list)


class pKaPredictor:
    """
    pKa/LogD予測（ChemAxon/ACD Labs inspired）
    
    Features:
    - 酸性/塩基性官能基検出
    - 経験的pKa推定
    - pH依存LogD計算
    
    Example:
        >>> predictor = pKaPredictor()
        >>> state = predictor.predict("CC(=O)O")  # 酢酸
    """
    
    # 官能基とその典型的なpKa値
    ACIDIC_GROUPS = {
        'Carboxylic acid': ('[CX3](=O)[OX1H1]', 4.5),
        'Sulfonic acid': ('S(=O)(=O)O', 1.0),
        'Phenol': ('c[OX2H1]', 10.0),
        'Sulfonamide': ('S(=O)(=O)N', 10.5),
        'Tetrazole': ('c1nnn[nH]1', 4.9),
    }
    
    BASIC_GROUPS = {
        'Primary amine': ('[NX3H2;!$(NC=O)]', 10.5),
        'Secondary amine': ('[NX3H1;!$(NC=O)]', 10.0),
        'Tertiary amine': ('[NX3H0;!$(NC=O)]', 9.5),
        'Pyridine': ('n1ccccc1', 5.2),
        'Imidazole': ('c1cnc[nH]1', 6.0),
        'Guanidine': ('NC(=N)N', 13.0),
    }
    
    def predict(self, smiles: str) -> Optional[IonizationState]:
        """pKa/LogDを予測"""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            state = IonizationState(smiles=smiles)
            
            # LogP
            state.logp = Descriptors.MolLogP(mol)
            
            # 酸性基検出
            for name, (smarts, pka) in self.ACIDIC_GROUPS.items():
                pattern = Chem.MolFromSmarts(smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    matches = mol.GetSubstructMatches(pattern)
                    for match in matches:
                        state.pka_values.append(pka)
                        state.ionizable_groups.append({
                            'type': 'acidic',
                            'name': name,
                            'pKa': pka,
                            'atoms': list(match),
                        })
            
            # 塩基性基検出
            for name, (smarts, pkb) in self.BASIC_GROUPS.items():
                pattern = Chem.MolFromSmarts(smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    matches = mol.GetSubstructMatches(pattern)
                    for match in matches:
                        state.pkb_values.append(pkb)
                        state.ionizable_groups.append({
                            'type': 'basic',
                            'name': name,
                            'pKb': pkb,
                            'atoms': list(match),
                        })
            
            # LogD at pH 7.4
            state.logd_at_pH7 = self._calculate_logd(state, 7.4)
            
            return state
            
        except Exception as e:
            logger.error(f"pKa prediction failed: {e}")
            return None
    
    def _calculate_logd(self, state: IonizationState, pH: float) -> float:
        """pH依存LogDを計算"""
        logp = state.logp
        
        # Henderson-Hasselbalch式の近似
        ionization_correction = 0.0
        
        for pka in state.pka_values:
            # 酸の場合
            fraction_ionized = 1 / (1 + 10 ** (pka - pH))
            ionization_correction -= fraction_ionized * 0.5
        
        for pkb in state.pkb_values:
            # 塩基の場合
            fraction_ionized = 1 / (1 + 10 ** (pH - pkb))
            ionization_correction -= fraction_ionized * 0.5
        
        return logp + ionization_correction
    
    def get_logd_curve(
        self,
        smiles: str,
        pH_range: tuple = (0, 14),
        n_points: int = 50,
    ) -> Dict[str, List[float]]:
        """pH-LogD曲線を取得"""
        state = self.predict(smiles)
        if state is None:
            return {'pH': [], 'logD': []}
        
        pH_values = np.linspace(pH_range[0], pH_range[1], n_points)
        logD_values = [self._calculate_logd(state, pH) for pH in pH_values]
        
        return {
            'pH': pH_values.tolist(),
            'logD': logD_values,
        }
    
    def predict_dominant_species(self, smiles: str, pH: float = 7.4) -> str:
        """支配的化学種を予測"""
        state = self.predict(smiles)
        if state is None:
            return "Unknown"
        
        # 酸性/塩基性の状態判定
        charges = []
        
        for group in state.ionizable_groups:
            if group['type'] == 'acidic':
                pka = group['pKa']
                if pH > pka + 1:
                    charges.append(-1)  # 脱プロトン
            else:  # basic
                pkb = group['pKb']
                if pH < pkb - 1:
                    charges.append(+1)  # プロトン化
        
        net_charge = sum(charges)
        
        if net_charge > 0:
            return "Cationic"
        elif net_charge < 0:
            return "Anionic"
        elif len(state.ionizable_groups) > 0:
            return "Zwitterionic"
        else:
            return "Neutral"
