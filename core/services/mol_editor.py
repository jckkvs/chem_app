"""
インタラクティブ分子エディタ

Implements: F-MOLEDITOR-001
設計思想:
- 分子編集操作
- 履歴管理
- 検証
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class EditOperation:
    """編集操作"""
    operation: str
    params: Dict[str, Any]
    before_smiles: str
    after_smiles: str


class MolecularEditor:
    """
    分子エディタ
    
    Features:
    - 原子/結合の追加・削除
    - 操作履歴
    - Undo/Redo
    
    Example:
        >>> editor = MolecularEditor("CCO")
        >>> editor.add_functional_group("OH")
        >>> editor.undo()
    """
    
    def __init__(self, smiles: str = ""):
        self.current_smiles = smiles
        self.history: List[EditOperation] = []
        self.redo_stack: List[EditOperation] = []
    
    def set_molecule(self, smiles: str) -> bool:
        """分子を設定"""
        if self._validate(smiles):
            self.current_smiles = smiles
            self.history.clear()
            self.redo_stack.clear()
            return True
        return False
    
    def add_atom(self, atom_symbol: str, position: int = -1) -> bool:
        """原子を追加"""
        before = self.current_smiles
        
        if position == -1:
            new_smiles = f"{self.current_smiles}{atom_symbol}"
        else:
            new_smiles = (
                self.current_smiles[:position] +
                atom_symbol +
                self.current_smiles[position:]
            )
        
        if self._validate(new_smiles):
            self._record_operation('add_atom', {'atom': atom_symbol}, before, new_smiles)
            self.current_smiles = new_smiles
            return True
        return False
    
    def add_functional_group(self, group: str) -> bool:
        """官能基を追加"""
        before = self.current_smiles
        
        # 簡易的な追加
        new_smiles = f"{self.current_smiles}{group}"
        
        if self._validate(new_smiles):
            self._record_operation('add_group', {'group': group}, before, new_smiles)
            self.current_smiles = new_smiles
            return True
        return False
    
    def replace_atom(self, old_atom: str, new_atom: str) -> bool:
        """原子を置換"""
        if old_atom not in self.current_smiles:
            return False
        
        before = self.current_smiles
        new_smiles = self.current_smiles.replace(old_atom, new_atom, 1)
        
        if self._validate(new_smiles):
            self._record_operation(
                'replace_atom',
                {'old': old_atom, 'new': new_atom},
                before,
                new_smiles,
            )
            self.current_smiles = new_smiles
            return True
        return False
    
    def undo(self) -> bool:
        """元に戻す"""
        if not self.history:
            return False
        
        operation = self.history.pop()
        self.redo_stack.append(operation)
        self.current_smiles = operation.before_smiles
        return True
    
    def redo(self) -> bool:
        """やり直し"""
        if not self.redo_stack:
            return False
        
        operation = self.redo_stack.pop()
        self.history.append(operation)
        self.current_smiles = operation.after_smiles
        return True
    
    def _validate(self, smiles: str) -> bool:
        """SMILES検証"""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except Exception:
            # RDKitなしでも基本的な検証
            return len(smiles) > 0
    
    def _record_operation(
        self,
        operation: str,
        params: Dict[str, Any],
        before: str,
        after: str,
    ) -> None:
        """操作を記録"""
        self.history.append(EditOperation(
            operation=operation,
            params=params,
            before_smiles=before,
            after_smiles=after,
        ))
        self.redo_stack.clear()
    
    def get_history(self) -> List[EditOperation]:
        """履歴を取得"""
        return self.history.copy()
    
    def canonicalize(self) -> str:
        """標準化"""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(self.current_smiles)
            if mol:
                self.current_smiles = Chem.MolToSmiles(mol)
        except Exception:
            pass
        return self.current_smiles
