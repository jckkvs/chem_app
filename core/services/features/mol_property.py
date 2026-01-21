"""
分子物性計算機

Implements: F-MOLPROP-001
設計思想:
- RDKitベースの即座物性計算
- 一般的な物性値を一括取得
- Lipinski's Rule of 5など
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MolecularProperties:
    """分子物性"""
    smiles: str
    molecular_weight: float
    logp: float
    hbd: int  # H-bond donors
    hba: int  # H-bond acceptors
    tpsa: float  # Topological polar surface area
    rotatable_bonds: int
    num_rings: int
    num_aromatic_rings: int
    num_atoms: int
    num_heavy_atoms: int
    fraction_sp3: float
    
    # Lipinski's Rule of 5
    lipinski_violations: int
    is_druglike: bool
    
    # 追加物性
    molecular_formula: str = ""
    exact_mass: float = 0.0
    qed: float = 0.0  # Quantitative Estimate of Drug-likeness
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MolecularPropertyCalculator:
    """
    分子物性計算機
    
    Features:
    - 即座の物性計算
    - Lipinski's Rule of 5チェック
    - QED計算
    - バッチ計算
    
    Example:
        >>> calc = MolecularPropertyCalculator()
        >>> props = calc.calculate("CCO")
        >>> print(f"LogP: {props.logp}, MW: {props.molecular_weight}")
    """
    
    def calculate(self, smiles: str) -> Optional[MolecularProperties]:
        """
        分子物性を計算
        
        Args:
            smiles: SMILES文字列
            
        Returns:
            MolecularProperties or None
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
            from rdkit.Chem.QED import qed
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return None
            
            # 基本物性
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            rotatable = Lipinski.NumRotatableBonds(mol)
            num_rings = Lipinski.RingCount(mol)
            num_aromatic = Lipinski.NumAromaticRings(mol)
            num_atoms = mol.GetNumAtoms()
            num_heavy = Lipinski.HeavyAtomCount(mol)
            frac_sp3 = Lipinski.FractionCSP3(mol)
            
            # Lipinski's Rule of 5
            violations = 0
            if mw > 500:
                violations += 1
            if logp > 5:
                violations += 1
            if hbd > 5:
                violations += 1
            if hba > 10:
                violations += 1
            
            is_druglike = violations <= 1
            
            # 分子式
            formula = rdMolDescriptors.CalcMolFormula(mol)
            exact_mass = Descriptors.ExactMolWt(mol)
            
            # QED
            qed_score = qed(mol)
            
            return MolecularProperties(
                smiles=smiles,
                molecular_weight=mw,
                logp=logp,
                hbd=hbd,
                hba=hba,
                tpsa=tpsa,
                rotatable_bonds=rotatable,
                num_rings=num_rings,
                num_aromatic_rings=num_aromatic,
                num_atoms=num_atoms,
                num_heavy_atoms=num_heavy,
                fraction_sp3=frac_sp3,
                lipinski_violations=violations,
                is_druglike=is_druglike,
                molecular_formula=formula,
                exact_mass=exact_mass,
                qed=qed_score,
            )
            
        except Exception as e:
            logger.error(f"Property calculation failed: {e}")
            return None
    
    def calculate_batch(self, smiles_list: List[str]) -> List[Optional[MolecularProperties]]:
        """バッチ計算"""
        return [self.calculate(smi) for smi in smiles_list]
    
    def to_dataframe(self, smiles_list: List[str]):
        """物性をDataFrameで返す"""
        import pandas as pd
        
        results = self.calculate_batch(smiles_list)
        data = [r.to_dict() if r else {'smiles': s} for s, r in zip(smiles_list, results)]
        return pd.DataFrame(data)
    
    def check_lipinski(self, smiles: str) -> Dict[str, Any]:
        """Lipinski's Rule of 5チェック"""
        props = self.calculate(smiles)
        if props is None:
            return {'valid': False}
        
        return {
            'valid': True,
            'mw_pass': props.molecular_weight <= 500,
            'logp_pass': props.logp <= 5,
            'hbd_pass': props.hbd <= 5,
            'hba_pass': props.hba <= 10,
            'violations': props.lipinski_violations,
            'is_druglike': props.is_druglike,
        }
    
    def get_summary_html(self, props: MolecularProperties) -> str:
        """物性サマリーHTML"""
        druglike_icon = "✅" if props.is_druglike else "❌"
        
        return f"""
        <div style="font-family: Arial; padding: 15px; background: #f5f5f5; border-radius: 10px;">
            <h3>📊 Molecular Properties</h3>
            <table style="width: 100%;">
                <tr><td>SMILES</td><td><code>{props.smiles}</code></td></tr>
                <tr><td>Formula</td><td>{props.molecular_formula}</td></tr>
                <tr><td>MW</td><td>{props.molecular_weight:.2f} g/mol</td></tr>
                <tr><td>LogP</td><td>{props.logp:.2f}</td></tr>
                <tr><td>TPSA</td><td>{props.tpsa:.1f} Å²</td></tr>
                <tr><td>H-Bond Donors</td><td>{props.hbd}</td></tr>
                <tr><td>H-Bond Acceptors</td><td>{props.hba}</td></tr>
                <tr><td>Rotatable Bonds</td><td>{props.rotatable_bonds}</td></tr>
                <tr><td>Rings</td><td>{props.num_rings} ({props.num_aromatic_rings} aromatic)</td></tr>
                <tr><td>QED</td><td>{props.qed:.3f}</td></tr>
                <tr><td>Drug-like</td><td>{druglike_icon} ({props.lipinski_violations} violations)</td></tr>
            </table>
        </div>
        """
