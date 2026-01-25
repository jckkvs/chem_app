"""
反応予測エンジン（RXNMapper/IBM RXN inspired）

Implements: F-REACTION-001
設計思想:
- 反応生成物予測
- 反応分類
- 原子マッピング
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ReactionResult:
    """反応予測結果"""
    reactants: List[str]
    products: List[str]
    reaction_smiles: str
    reaction_type: Optional[str] = None
    confidence: float = 0.0
    atom_mapping: Optional[Dict[int, int]] = None


class ReactionPredictor:
    """
    反応予測エンジン（RXN/Reaxys inspired）
    
    Features:
    - ルールベース反応予測
    - 反応タイプ分類
    - 逆合成分析
    
    Example:
        >>> predictor = ReactionPredictor()
        >>> result = predictor.predict_product("CC(=O)Cl.CCO")
    """
    
    # 基本的な反応ルール (SMARTS)
    REACTION_RULES = [
        {
            'name': 'Ester formation (Acyl chloride + Alcohol)',
            'type': 'condensation',
            'smarts': '[C:1](=[O:2])[Cl:3].[O:4][C:5]>>[C:1](=[O:2])[O:4][C:5]',
            'reactant_patterns': ['C(=O)Cl', 'O'],  # 簡易チェック用
        },
        {
            'name': 'Amide formation (Acyl chloride + Amine)',
            'type': 'condensation',
            'smarts': '[C:1](=[O:2])[Cl:3].[N:4][C:5]>>[C:1](=[O:2])[N:4][C:5]',
            'reactant_patterns': ['C(=O)Cl', 'N'],
        },
        {
            'name': 'Alkylation (Alkyl halide + Amine)',
            'type': 'substitution',
            'smarts': '[C:1][Cl,Br,I:2].[N:3]>>[C:1][N:3]',
            'reactant_patterns': ['[Cl,Br,I]', 'N'],
        },
    ]
    
    def predict_product(
        self,
        reactants_smiles: str,
    ) -> Optional[ReactionResult]:
        """反応生成物を予測"""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            # 反応物を分割
            reactants = reactants_smiles.split('.')
            
            if len(reactants) < 2:
                return None
            
            # 各反応ルールをチェック
            for rule in self.REACTION_RULES:
                result = self._apply_rule(reactants, rule)
                if result:
                    return result
            
            # マッチしない場合は入力をそのまま返す
            return ReactionResult(
                reactants=reactants,
                products=reactants,
                reaction_smiles=f"{reactants_smiles}>>",
                reaction_type="unknown",
                confidence=0.0,
            )
            
        except Exception as e:
            logger.error(f"Reaction prediction failed: {e}")
            return None
    
    def _apply_rule(
        self,
        reactants: List[str],
        rule: Dict[str, Any],
    ) -> Optional[ReactionResult]:
        """反応ルールを適用"""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            # SMARTSから反応オブジェクトを作成
            rxn = AllChem.ReactionFromSmarts(rule['smarts'])
            
            # 各反応物をMoleculeオブジェクトに変換
            mol_reactants = []
            for smi in reactants:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    mol_reactants.append(mol)
            
            if len(mol_reactants) != len(reactants):
                return None

            try:
                # 反応を実行 (RunReactantsは考えられる組み合わせをすべて返す)
                # 今回は単純化のため、入力順序がSMARTSと一致すると仮定するか、
                # あるいはPermutationを試すが、まずは単純にリストを渡す
                # RDKitのRunReactantsは tuple of tuples を返す ((product1,), (product1', product2'), ...)
                
                # 反応物の数が合わない場合はスキップ（簡易実装）
                if rxn.GetNumReactantTemplates() != len(mol_reactants):
                    # 順序や数が重要。ここでは単純に渡す
                    pass

                products_set = rxn.RunReactants(mol_reactants)
                
                if not products_set:
                    # 順序逆転して再トライ（2成分系のみ）
                    if len(mol_reactants) == 2:
                        products_set = rxn.RunReactants(mol_reactants[::-1])

                if products_set:
                    # 最初の生成物セットを採用
                    products = products_set[0]
                    product_smiles = [Chem.MolToSmiles(m) for m in products]
                    
                    return ReactionResult(
                        reactants=reactants,
                        products=product_smiles,
                        reaction_smiles=f"{'.'.join(reactants)}>>{'.'.join(product_smiles)}",
                        reaction_type=rule['type'],
                        confidence=0.9,
                    )
            
            except Exception as e:
                # logger.warning(f"Rule application failed: {e}")
                pass
            
            return None
            
        except Exception:
            return None
    
    def classify_reaction(self, reaction_smiles: str) -> Optional[str]:
        """反応タイプを分類"""
        try:
            from rdkit import Chem
            
            if '>>' not in reaction_smiles:
                return None
            
            reactants, products = reaction_smiles.split('>>')
            
            reactant_mols = [Chem.MolFromSmiles(s) for s in reactants.split('.')]
            product_mols = [Chem.MolFromSmiles(s) for s in products.split('.')]
            
            # 原子数変化
            r_atoms = sum(m.GetNumAtoms() for m in reactant_mols if m)
            p_atoms = sum(m.GetNumAtoms() for m in product_mols if m)
            
            if p_atoms < r_atoms:
                return "elimination"
            elif p_atoms > r_atoms:
                return "addition"
            else:
                return "substitution"
                
        except Exception:
            return None
    
    def get_retrosynthesis(
        self,
        target_smiles: str,
        max_steps: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        逆合成分析（簡易版）
        
        Returns:
            逆合成ルートリスト
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            routes = []
            mol = Chem.MolFromSmiles(target_smiles)
            
            if mol is None:
                return routes
            
            # エステル結合を切断
            ester_pattern = Chem.MolFromSmarts('C(=O)O')
            if mol.HasSubstructMatch(ester_pattern):
                routes.append({
                    'step': 1,
                    'reaction_type': 'Ester hydrolysis',
                    'precursors': ['Carboxylic acid', 'Alcohol'],
                })
            
            # アミド結合を切断
            amide_pattern = Chem.MolFromSmarts('C(=O)N')
            if mol.HasSubstructMatch(amide_pattern):
                routes.append({
                    'step': 1,
                    'reaction_type': 'Amide hydrolysis',
                    'precursors': ['Carboxylic acid', 'Amine'],
                })
            
            return routes
            
        except Exception as e:
            logger.error(f"Retrosynthesis failed: {e}")
            return []
