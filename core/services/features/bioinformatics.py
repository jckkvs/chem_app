"""
バイオインフォマティクス統合

Implements: F-BIO-001
設計思想:
- タンパク質配列処理
- 配列類似性
- モチーフ検索
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SequenceInfo:
    """配列情報"""
    sequence: str
    length: int
    composition: Dict[str, float]
    molecular_weight: float


@dataclass
class AlignmentResult:
    """アラインメント結果"""
    seq1: str
    seq2: str
    score: float
    identity: float
    aligned_seq1: str
    aligned_seq2: str


class BioSequenceAnalyzer:
    """
    バイオインフォマティクス分析
    
    Features:
    - 配列組成分析
    - ペアワイズアラインメント
    - モチーフ検索
    
    Example:
        >>> bio = BioSequenceAnalyzer()
        >>> info = bio.analyze_sequence("MKTAYIAKQRQISFVKSHFSRQLE")
    """
    
    # アミノ酸の分子量
    AA_MW = {
        'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
        'E': 147.1, 'Q': 146.1, 'G': 75.1, 'H': 155.2, 'I': 131.2,
        'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
        'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1,
    }
    
    def analyze_sequence(self, sequence: str) -> SequenceInfo:
        """配列を分析"""
        seq_upper = sequence.upper()
        length = len(seq_upper)
        
        # 組成
        composition = {}
        for aa in set(seq_upper):
            composition[aa] = seq_upper.count(aa) / length
        
        # 分子量
        mw = sum(self.AA_MW.get(aa, 0) for aa in seq_upper) - (length - 1) * 18
        
        return SequenceInfo(
            sequence=sequence,
            length=length,
            composition=composition,
            molecular_weight=mw,
        )
    
    def align(
        self,
        seq1: str,
        seq2: str,
        match: int = 1,
        mismatch: int = -1,
        gap: int = -2,
    ) -> AlignmentResult:
        """ペアワイズアラインメント（Needleman-Wunsch）"""
        m, n = len(seq1), len(seq2)
        
        # スコア行列
        score = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 初期化
        for i in range(m + 1):
            score[i][0] = i * gap
        for j in range(n + 1):
            score[0][j] = j * gap
        
        # DP
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                match_score = match if seq1[i-1] == seq2[j-1] else mismatch
                score[i][j] = max(
                    score[i-1][j-1] + match_score,
                    score[i-1][j] + gap,
                    score[i][j-1] + gap,
                )
        
        # バックトレース
        aligned1, aligned2 = [], []
        i, j = m, n
        
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                match_score = match if seq1[i-1] == seq2[j-1] else mismatch
                if score[i][j] == score[i-1][j-1] + match_score:
                    aligned1.append(seq1[i-1])
                    aligned2.append(seq2[j-1])
                    i -= 1
                    j -= 1
                    continue
            
            if i > 0 and score[i][j] == score[i-1][j] + gap:
                aligned1.append(seq1[i-1])
                aligned2.append('-')
                i -= 1
            else:
                aligned1.append('-')
                aligned2.append(seq2[j-1])
                j -= 1
        
        aligned1 = ''.join(reversed(aligned1))
        aligned2 = ''.join(reversed(aligned2))
        
        # 一致率
        matches = sum(1 for a, b in zip(aligned1, aligned2) if a == b and a != '-')
        identity = matches / max(len(aligned1), 1)
        
        return AlignmentResult(
            seq1=seq1,
            seq2=seq2,
            score=score[m][n],
            identity=identity,
            aligned_seq1=aligned1,
            aligned_seq2=aligned2,
        )
    
    def find_motif(self, sequence: str, motif: str) -> List[int]:
        """モチーフ検索"""
        positions = []
        start = 0
        
        while True:
            pos = sequence.find(motif, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        
        return positions
