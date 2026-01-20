"""
SELFIES (SELF-referencIng Embedded Strings) サポート

Implements: F-SELFIES-001
設計思想:
- SMILESの問題点を解決する堅牢な分子表現
- 100%有効な分子を保証
- 生成モデル、強化学習に最適

参考文献:
- Krenn et al., "Self-referencing embedded strings (SELFIES)", 
  Machine Learning: Science and Technology, 2020

依存関係:
- selfies>=2.1 (pip install selfies)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_SELFIES_AVAILABLE: Optional[bool] = None


def _check_selfies() -> bool:
    global _SELFIES_AVAILABLE
    if _SELFIES_AVAILABLE is None:
        try:
            import selfies
            _SELFIES_AVAILABLE = True
            logger.info(f"selfies v{selfies.__version__} available")
        except ImportError:
            _SELFIES_AVAILABLE = False
    return _SELFIES_AVAILABLE


def is_selfies_available() -> bool:
    """SELFIESが利用可能か"""
    return _check_selfies()


def smiles_to_selfies(smiles: str) -> Optional[str]:
    """SMILESからSELFIESに変換"""
    if not _check_selfies():
        logger.warning("selfies not installed")
        return None
    
    import selfies as sf
    
    try:
        return sf.encoder(smiles)
    except Exception as e:
        logger.debug(f"SMILES to SELFIES failed: {e}")
        return None


def selfies_to_smiles(selfies_str: str) -> Optional[str]:
    """SELFIESからSMILESに変換"""
    if not _check_selfies():
        return None
    
    import selfies as sf
    
    try:
        return sf.decoder(selfies_str)
    except Exception as e:
        logger.debug(f"SELFIES to SMILES failed: {e}")
        return None


def batch_smiles_to_selfies(smiles_list: List[str]) -> List[Optional[str]]:
    """バッチ変換 SMILES→SELFIES"""
    return [smiles_to_selfies(smi) for smi in smiles_list]


def batch_selfies_to_smiles(selfies_list: List[str]) -> List[Optional[str]]:
    """バッチ変換 SELFIES→SMILES"""
    return [selfies_to_smiles(sf) for sf in selfies_list]


def get_selfies_alphabet() -> List[str]:
    """SELFIESアルファベット（トークン）を取得"""
    if not _check_selfies():
        return []
    
    import selfies as sf
    
    return list(sf.get_semantic_robust_alphabet())


def tokenize_selfies(selfies_str: str) -> List[str]:
    """SELFIESをトークンに分割"""
    if not _check_selfies():
        return []
    
    import selfies as sf
    
    try:
        return list(sf.split_selfies(selfies_str))
    except Exception:
        return []


def is_valid_selfies(selfies_str: str) -> bool:
    """有効なSELFIESか確認"""
    if not _check_selfies():
        return False
    
    # SELFIESは常に有効な分子を生成するが、
    # 念のためデコード可能か確認
    smiles = selfies_to_smiles(selfies_str)
    return smiles is not None


class SELFIESEncoder:
    """
    SMILES⇔SELFIES変換器
    
    Usage:
        encoder = SELFIESEncoder()
        selfies = encoder.encode("CCO")  # "[C][C][O]"
        smiles = encoder.decode("[C][C][O]")  # "CCO"
    """
    
    def __init__(self):
        if not _check_selfies():
            logger.warning("selfies not installed, operations will fail")
    
    def encode(self, smiles: str) -> Optional[str]:
        """SMILES → SELFIES"""
        return smiles_to_selfies(smiles)
    
    def decode(self, selfies_str: str) -> Optional[str]:
        """SELFIES → SMILES"""
        return selfies_to_smiles(selfies_str)
    
    def encode_batch(self, smiles_list: List[str]) -> List[Optional[str]]:
        """バッチ SMILES → SELFIES"""
        return batch_smiles_to_selfies(smiles_list)
    
    def decode_batch(self, selfies_list: List[str]) -> List[Optional[str]]:
        """バッチ SELFIES → SMILES"""
        return batch_selfies_to_smiles(selfies_list)
    
    def get_alphabet(self) -> List[str]:
        """トークンアルファベット"""
        return get_selfies_alphabet()
    
    def tokenize(self, selfies_str: str) -> List[str]:
        """トークン化"""
        return tokenize_selfies(selfies_str)
