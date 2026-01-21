"""
毒性予測エンジン（Derek/Toxtree inspired）

Implements: F-TOX-001
設計思想:
- 構造アラート検出
- 毒性エンドポイント予測
- リスク評価
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToxAlert:
    """毒性アラート"""
    name: str
    category: str  # 'mutagenicity', 'carcinogenicity', 'hepatotoxicity', etc.
    severity: str  # 'high', 'medium', 'low'
    description: str
    matched_atoms: List[int] = field(default_factory=list)


@dataclass
class ToxicityProfile:
    """毒性プロファイル"""
    smiles: str
    alerts: List[ToxAlert] = field(default_factory=list)
    risk_score: float = 0.0
    is_toxic: bool = False
    endpoints: Dict[str, bool] = field(default_factory=dict)


class ToxicityPredictor:
    """
    毒性予測（Derek/Toxtree/ToxTree inspired）
    
    Features:
    - 構造アラート検出（Benigni/Bossa rules）
    - 変異原性/発がん性予測
    - リスクスコア計算
    
    Example:
        >>> predictor = ToxicityPredictor()
        >>> profile = predictor.predict("c1ccc(N)cc1")  # アニリン
    """
    
    # 毒性アラートSMARTS（Benigni/Bossaルール等から）
    ALERTS = [
        # 変異原性
        ("[N;R0]=[N;R0]", "Azo compound", "mutagenicity", "medium", "Azo結合は変異原性の可能性"),
        ("[OH]c1ccc([N+](=O)[O-])cc1", "Nitrophenol", "mutagenicity", "high", "ニトロフェノールは変異原性"),
        ("[N;!R]c1ccccc1", "Aromatic amine", "mutagenicity", "medium", "芳香族アミンはDNA付加体形成"),
        ("O=N-c", "Nitroso aromatic", "mutagenicity", "high", "ニトロソ化合物は強い変異原性"),
        
        # 発がん性
        ("C1OC1", "Epoxide", "carcinogenicity", "high", "エポキシドはDNAアルキル化"),
        ("[N+](=O)[O-]c1ccccc1", "Nitroaromatic", "carcinogenicity", "high", "ニトロ芳香族は発がん性"),
        
        # 肝毒性
        ("C(=O)Cl", "Acyl halide", "hepatotoxicity", "high", "アシルハライドは反応性"),
        ("[#6]S(=O)(=O)O", "Sulfate ester", "hepatotoxicity", "medium", "硫酸エステルはアルキル化"),
        
        # 皮膚感作性
        ("C=CC(=O)", "Michael acceptor", "skin_sensitization", "medium", "マイケルアクセプターは感作性"),
        ("[Cl,Br,I]C=O", "Halogenated carbonyl", "skin_sensitization", "high", "ハロゲン化カルボニル"),
    ]
    
    def predict(self, smiles: str) -> Optional[ToxicityProfile]:
        """毒性を予測"""
        try:
            from rdkit import Chem
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            profile = ToxicityProfile(smiles=smiles)
            
            # 各アラートをチェック
            for smarts, name, category, severity, description in self.ALERTS:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    matches = mol.GetSubstructMatches(pattern)
                    for match in matches:
                        profile.alerts.append(ToxAlert(
                            name=name,
                            category=category,
                            severity=severity,
                            description=description,
                            matched_atoms=list(match),
                        ))
            
            # エンドポイント集計
            categories = set(a.category for a in profile.alerts)
            profile.endpoints = {
                'mutagenicity': 'mutagenicity' in categories,
                'carcinogenicity': 'carcinogenicity' in categories,
                'hepatotoxicity': 'hepatotoxicity' in categories,
                'skin_sensitization': 'skin_sensitization' in categories,
            }
            
            # リスクスコア
            severity_weights = {'high': 3, 'medium': 2, 'low': 1}
            total_weight = sum(severity_weights.get(a.severity, 1) for a in profile.alerts)
            profile.risk_score = min(1.0, total_weight / 10)
            profile.is_toxic = profile.risk_score > 0.3
            
            return profile
            
        except Exception as e:
            logger.error(f"Toxicity prediction failed: {e}")
            return None
    
    def get_report_html(self, profile: ToxicityProfile) -> str:
        """HTML レポート"""
        risk_color = '#e74c3c' if profile.risk_score > 0.5 else '#f39c12' if profile.risk_score > 0.2 else '#27ae60'
        
        alerts_html = ""
        for alert in profile.alerts:
            alerts_html += f"<li><b>{alert.name}</b> ({alert.category}): {alert.description}</li>"
        
        return f"""
        <div style="padding: 15px; background: #fdf2f2; border-radius: 10px;">
            <h3>☠️ Toxicity Assessment</h3>
            <p>Risk Score: <span style="color: {risk_color}; font-size: 1.5em;">{profile.risk_score:.2f}</span></p>
            <h4>Detected Alerts ({len(profile.alerts)})</h4>
            <ul>{alerts_html if alerts_html else "<li>No alerts detected</li>"}</ul>
        </div>
        """
