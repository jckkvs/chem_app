"""
材料科学物性別記述子プリセット定義

Implements: F-PRESET-001
設計思想:
- 目的物性に応じた最適かつ最小限の記述子セット
- 材料科学文献・ドメイン知識に基づく選定
- ユーザーによる追加・削除を前提とした拡張性

参考文献:
- Molecular Descriptors for Chemoinformatics (Todeschini & Consonni, 2009)
- Polymer Property Prediction with ML (Kuenneth et al., 2021)
- Glass Transition Temperature Prediction (Pilania et al., 2019)
- Refractive Index Prediction in Polymers (Liu et al., 2020)

注意:
- これは「完成」ではなく継続的に改善されるべきリソース
- 新しい知見が得られたら追加・修正すべき
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum


class PropertyCategory(Enum):
    """物性カテゴリ"""
    OPTICAL = "optical"
    MECHANICAL = "mechanical"
    THERMAL = "thermal"
    ELECTRICAL = "electrical"
    CHEMICAL = "chemical"
    TRANSPORT = "transport"
    PHARMACOLOGICAL = "pharmacological"


@dataclass
class DescriptorPreset:
    """
    物性タイプ別の記述子プリセット定義
    
    Attributes:
        name: プリセット名
        name_ja: 日本語名
        category: 物性カテゴリ
        description: 説明
        rationale: 選定根拠（なぜこの記述子が有効か）
        rdkit_descriptors: RDKit記述子リスト
        morgan_fp: Morganフィンガープリントを使用するか
        morgan_radius: Morganフィンガープリントの半径
        morgan_bits: Morganフィンガープリントのビット数
        requires_3d: 3D記述子が必要か
        requires_xtb: XTB計算が必要か
        xtb_descriptors: XTB記述子リスト
        recommended_pretrained: 推奨事前学習モデル
        literature_refs: 文献参照
    """
    name: str
    name_ja: str
    category: PropertyCategory
    description: str
    rationale: str
    rdkit_descriptors: List[str]
    morgan_fp: bool = False
    morgan_radius: int = 2
    morgan_bits: int = 1024
    requires_3d: bool = False
    requires_xtb: bool = False
    xtb_descriptors: List[str] = field(default_factory=list)
    recommended_pretrained: List[str] = field(default_factory=list)
    literature_refs: List[str] = field(default_factory=list)
    
    @property
    def total_descriptor_count(self) -> int:
        """総記述子数の概算"""
        count = len(self.rdkit_descriptors)
        if self.morgan_fp:
            count += self.morgan_bits
        count += len(self.xtb_descriptors)
        return count


# =============================================================================
# 光学物性 (Optical Properties)
# =============================================================================

REFRACTIVE_INDEX_PRESET = DescriptorPreset(
    name="refractive_index",
    name_ja="屈折率",
    category=PropertyCategory.OPTICAL,
    description="屈折率・光学透過性の予測に最適",
    rationale="""
    屈折率は分子の電子分極率と密接に関連。
    - MolMR (分子屈折): 直接的な相関
    - 芳香環数: π電子による高分極率
    - 共役系長さ: 電子雲の広がり
    - ヘテロ原子: 電子密度変化
    """,
    rdkit_descriptors=[
        # 分極率関連
        'MolMR',           # 分子屈折（最も重要）
        'LabuteASA',       # 表面積（分極率と相関）
        
        # 芳香族性・共役
        'NumAromaticRings',
        'NumAromaticCarbocycles',
        'NumAromaticHeterocycles',
        'FractionCSP3',    # sp3炭素比率（逆相関）
        
        # 電子構造
        'NumHeteroatoms',
        'NumValenceElectrons',
        
        # 分子サイズ
        'HeavyAtomCount',
        'MolWt',
    ],
    requires_xtb=True,
    xtb_descriptors=['dipole_norm', 'polarizability'],
    recommended_pretrained=['unimol'],
    literature_refs=[
        "Liu et al., J. Chem. Inf. Model. 2020",
        "Bicerano, Prediction of Polymer Properties, 2002",
    ]
)

OPTICAL_GAP_PRESET = DescriptorPreset(
    name="optical_gap",
    name_ja="光学バンドギャップ",
    category=PropertyCategory.OPTICAL,
    description="HOMO-LUMOギャップ、吸収波長の予測",
    rationale="""
    光学ギャップは共役系の長さと電子状態に依存。
    - 芳香環数・共役長: バンドギャップと逆相関
    - ヘテロ原子: 電子状態への影響
    - XTB HOMO-LUMOギャップ: 直接的な計算値
    """,
    rdkit_descriptors=[
        'NumAromaticRings',
        'NumAromaticCarbocycles',
        'NumAromaticHeterocycles',
        'NumHeteroatoms',
        'NumValenceElectrons',
        'NumRadicalElectrons',
        'FractionCSP3',
        'BertzCT',         # 分子複雑度
    ],
    requires_xtb=True,
    xtb_descriptors=['homo_lumo_gap', 'energy', 'homo', 'lumo'],
    recommended_pretrained=['unimol', 'chemberta'],
    literature_refs=[
        "Kuenneth et al., Nat. Commun. 2021",
    ]
)


# =============================================================================
# 機械物性 (Mechanical Properties)
# =============================================================================

ELASTIC_MODULUS_PRESET = DescriptorPreset(
    name="elastic_modulus",
    name_ja="弾性率・ヤング率",
    category=PropertyCategory.MECHANICAL,
    description="弾性率、剛性、変形抵抗の予測",
    rationale="""
    弾性率は分子鎖の剛直性と分子間相互作用に依存。
    - 回転可能結合数: 柔軟性の指標（逆相関）
    - 芳香環数: 剛直性の寄与
    - 水素結合能: 分子間相互作用
    - 分子形状: 充填効率
    """,
    rdkit_descriptors=[
        # 剛直性
        'NumRotatableBonds',  # 柔軟性（逆相関）
        'NumAromaticRings',   # 剛直性
        'RingCount',
        'NumSpiroAtoms',
        'NumBridgeheadAtoms',
        
        # 分子間相互作用
        'NumHDonors',
        'NumHAcceptors',
        'TPSA',
        
        # 形状
        'FractionCSP3',
        'HeavyAtomCount',
        'MolWt',
        
        # 3D形状記述子（requires_3d=Trueの場合に利用）
        'Asphericity',
        'Eccentricity', 
        'InertialShapeFactor',
        'RadiusOfGyration',
    ],
    requires_3d=True,
    recommended_pretrained=['unimol'],
    literature_refs=[
        "Afzal & Huan, J. Phys. Chem. C 2019",
        "Wu et al., npj Comput. Mater. 2020",
    ]
)

TENSILE_STRENGTH_PRESET = DescriptorPreset(
    name="tensile_strength",
    name_ja="引張強度",
    category=PropertyCategory.MECHANICAL,
    description="引張強度、破断強度の予測",
    rationale="""
    引張強度は分子鎖の絡み合いと分子間結合に依存。
    弾性率と類似だが、分子量の寄与が大きい。
    """,
    rdkit_descriptors=[
        'MolWt',
        'HeavyAtomMolWt',
        'NumRotatableBonds',
        'NumAromaticRings',
        'RingCount',
        'NumHDonors',
        'NumHAcceptors',
        'Chi0', 'Chi1',    # 連結性指数
        'Kappa1', 'Kappa2', 'Kappa3',  # 形状指数
    ],
    recommended_pretrained=['unimol'],
)

HARDNESS_PRESET = DescriptorPreset(
    name="hardness",
    name_ja="硬度",
    category=PropertyCategory.MECHANICAL,
    description="硬度、耐摩耗性の予測",
    rationale="""
    硬度は架橋密度、結晶性、分子間相互作用に関連。
    """,
    rdkit_descriptors=[
        'NumAromaticRings',
        'NumHDonors',
        'NumHAcceptors',
        'RingCount',
        'NumSpiroAtoms',
        'NumBridgeheadAtoms',
        'BertzCT',
        'HallKierAlpha',
    ],
)


# =============================================================================
# 熱物性 (Thermal Properties)
# =============================================================================

GLASS_TRANSITION_PRESET = DescriptorPreset(
    name="glass_transition",
    name_ja="ガラス転移温度(Tg)",
    category=PropertyCategory.THERMAL,
    description="ガラス転移温度の予測（ポリマー重要物性）",
    rationale="""
    Tgは分子鎖の運動性に依存。
    - 回転可能結合: 運動性（Tg低下）
    - 剛直構造（芳香環等）: Tg上昇
    - 分子間相互作用（水素結合）: Tg上昇
    - 側鎖の嵩高さ: 自由体積→Tg低下
    
    PolyBERTが特に有効。
    """,
    rdkit_descriptors=[
        # 運動性
        'NumRotatableBonds',
        'FractionCSP3',
        
        # 剛直性
        'NumAromaticRings',
        'NumAromaticCarbocycles',
        'RingCount',
        
        # 分子間相互作用
        'NumHDonors',
        'NumHAcceptors',
        'TPSA',
        
        # サイズ・形状
        'MolWt',
        'HeavyAtomCount',
        
        # トポロジー
        'Chi0', 'Chi1', 'Chi2n', 'Chi3n',
        'Kappa1', 'Kappa2', 'Kappa3',
        'BertzCT',
    ],
    recommended_pretrained=['polybert', 'unimol'],
    literature_refs=[
        "Pilania et al., J. Chem. Inf. Model. 2019",
        "Kuenneth et al., Macromolecules 2021",
    ]
)

MELTING_POINT_PRESET = DescriptorPreset(
    name="melting_point",
    name_ja="融点",
    category=PropertyCategory.THERMAL,
    description="融点の予測",
    rationale="""
    融点は分子対称性、分子間相互作用、分子量に依存。
    - 対称性の高い分子: 高融点
    - 水素結合能: 融点上昇
    - 分子量: 一般に融点上昇
    """,
    rdkit_descriptors=[
        'MolWt',
        'HeavyAtomMolWt',
        'NumHDonors',
        'NumHAcceptors',
        'TPSA',
        'NumAromaticRings',
        'RingCount',
        'NumRotatableBonds',
        'FractionCSP3',
        'Chi0', 'Chi1',
    ],
    morgan_fp=True,
    morgan_radius=2,
    morgan_bits=512,
    recommended_pretrained=['chemberta'],
)

THERMAL_CONDUCTIVITY_PRESET = DescriptorPreset(
    name="thermal_conductivity",
    name_ja="熱伝導率",
    category=PropertyCategory.THERMAL,
    description="熱伝導率の予測",
    rationale="""
    熱伝導率はフォノン伝搬と分子振動モードに関連。
    - 剛直な骨格: 高い熱伝導率
    - 共役系: フォノン伝搬促進
    - 結晶性: 秩序構造で高熱伝導率
    
    MD/DFT由来記述子が最も有効だが、
    RDKitでは間接的な指標を使用。
    """,
    rdkit_descriptors=[
        'NumAromaticRings',
        'NumRotatableBonds',
        'RingCount',
        'FractionCSP3',
        'MolWt',
        'HeavyAtomCount',
        'BertzCT',
        'Chi0', 'Chi1',
    ],
    requires_xtb=True,
    xtb_descriptors=['energy', 'dipole_norm'],
    literature_refs=[
        "Ma et al., ACS Appl. Mater. Interfaces 2019",
    ]
)

THERMAL_STABILITY_PRESET = DescriptorPreset(
    name="thermal_stability",
    name_ja="熱安定性（分解温度）",
    category=PropertyCategory.THERMAL,
    description="熱分解開始温度(Td)の予測",
    rationale="""
    熱安定性は結合エネルギーと分子構造に依存。
    - 芳香環: 高い熱安定性
    - 弱い結合（例：エステル）: 低下
    - ヘテロ原子: 依存（N,Oは低下傾向）
    """,
    rdkit_descriptors=[
        'NumAromaticRings',
        'NumAromaticCarbocycles',
        'RingCount',
        'NumHeteroatoms',
        'FractionCSP3',
        'BertzCT',
        'MolWt',
    ],
    requires_xtb=True,
    xtb_descriptors=['energy'],
)


# =============================================================================
# 電気物性 (Electrical Properties)
# =============================================================================

DIELECTRIC_CONSTANT_PRESET = DescriptorPreset(
    name="dielectric_constant",
    name_ja="誘電率",
    category=PropertyCategory.ELECTRICAL,
    description="誘電率・誘電損失の予測",
    rationale="""
    誘電率は分子の極性と分極率に依存。
    - 極性基: 高誘電率
    - 分極率: 電子分極成分
    """,
    rdkit_descriptors=[
        'MolMR',
        'TPSA',
        'NumHDonors',
        'NumHAcceptors',
        'NumHeteroatoms',
        'LabuteASA',
    ],
    requires_xtb=True,
    xtb_descriptors=['dipole_norm', 'polarizability'],
)

CONDUCTIVITY_PRESET = DescriptorPreset(
    name="conductivity",
    name_ja="電気伝導度",
    category=PropertyCategory.ELECTRICAL,
    description="電気伝導度の予測（導電性ポリマー等）",
    rationale="""
    電気伝導度は共役系の長さとHOMO-LUMOギャップに依存。
    - 長い共役系: 高伝導度
    - 狭いバンドギャップ: 伝導促進
    """,
    rdkit_descriptors=[
        'NumAromaticRings',
        'NumAromaticCarbocycles',
        'NumValenceElectrons',
        'FractionCSP3',
        'NumHeteroatoms',
    ],
    requires_xtb=True,
    xtb_descriptors=['homo_lumo_gap', 'homo', 'lumo'],
)


# =============================================================================
# 化学物性 (Chemical Properties)
# =============================================================================

VISCOSITY_PRESET = DescriptorPreset(
    name="viscosity",
    name_ja="粘度",
    category=PropertyCategory.CHEMICAL,
    description="粘度の予測",
    rationale="""
    粘度は分子サイズ、形状、分子間相互作用に依存。
    - 分子量: 粘度上昇
    - 直鎖性: 絡み合い
    - 水素結合: 分子間相互作用
    """,
    rdkit_descriptors=[
        'MolWt',
        'HeavyAtomMolWt',
        'NumRotatableBonds',
        'NumHDonors',
        'NumHAcceptors',
        'TPSA',
        'FractionCSP3',
        'Chi0', 'Chi1',
    ],
)

DENSITY_PRESET = DescriptorPreset(
    name="density",
    name_ja="密度",
    category=PropertyCategory.CHEMICAL,
    description="密度の予測",
    rationale="""
    密度は分子体積と分子量の比。
    - 分子量と体積（MolMR）から間接推定
    - 芳香環: 高密度
    - 脂肪族: 低密度
    """,
    rdkit_descriptors=[
        'MolWt',
        'MolMR',
        'LabuteASA',
        'HeavyAtomCount',
        'NumAromaticRings',
        'FractionCSP3',
    ],
)

SURFACE_TENSION_PRESET = DescriptorPreset(
    name="surface_tension",
    name_ja="表面張力",
    category=PropertyCategory.CHEMICAL,
    description="表面張力の予測",
    rationale="""
    表面張力は分子間相互作用強度に依存。
    - 極性基: 高表面張力
    - 水素結合: 表面張力上昇
    """,
    rdkit_descriptors=[
        'TPSA',
        'NumHDonors',
        'NumHAcceptors',
        'MolLogP',
        'LabuteASA',
    ],
)


# =============================================================================
# 輸送物性 (Transport Properties)
# =============================================================================

GAS_PERMEABILITY_PRESET = DescriptorPreset(
    name="gas_permeability",
    name_ja="ガス透過性",
    category=PropertyCategory.TRANSPORT,
    description="ガス透過係数の予測（膜分離材料）",
    rationale="""
    ガス透過性は自由体積と分子運動性に依存。
    - FractionCSP3: 自由体積と相関
    - 回転可能結合: 鎖運動性
    - 芳香環: 剛直性→低透過性
    """,
    rdkit_descriptors=[
        'FractionCSP3',
        'NumRotatableBonds',
        'NumAromaticRings',
        'RingCount',
        'MolWt',
        'LabuteASA',
        'TPSA',
    ],
    literature_refs=[
        "Yuan et al., J. Membr. Sci. 2021",
    ]
)

SOLUBILITY_PRESET = DescriptorPreset(
    name="solubility",
    name_ja="溶解度・LogP",
    category=PropertyCategory.CHEMICAL,
    description="水溶性、脂溶性、分配係数の予測",
    rationale="""
    溶解度は極性と分子サイズのバランス。
    - MolLogP: 直接的な脂溶性指標
    - TPSA: 極性表面積（水溶性と相関）
    - 水素結合能: 水との相互作用
    """,
    rdkit_descriptors=[
        'MolLogP',
        'MolMR',
        'TPSA',
        'NumHDonors',
        'NumHAcceptors',
        'NumRotatableBonds',
        'NumHeteroatoms',
        'FractionCSP3',
        'LabuteASA',
        'NumAromaticRings',
        'NumSaturatedRings',
    ],
    recommended_pretrained=['chemberta'],
)


# =============================================================================
# 薬理学的物性 (Pharmacological Properties)
# =============================================================================

ADMET_PRESET = DescriptorPreset(
    name="admet",
    name_ja="ADMET・薬物動態",
    category=PropertyCategory.PHARMACOLOGICAL,
    description="吸収・分布・代謝・排泄・毒性の予測",
    rationale="""
    ADMETはLipinski's Rule of Fiveに基づく。
    - MW < 500, LogP < 5, HBD ≤ 5, HBA ≤ 10
    - QED: 薬剤らしさの定量指標
    """,
    rdkit_descriptors=[
        'MolLogP',
        'MolWt',
        'TPSA',
        'NumHDonors',
        'NumHAcceptors',
        'NumRotatableBonds',
        'NumAromaticRings',
        'FractionCSP3',
        'qed',  # Quantitative Estimate of Drug-likeness
    ],
    morgan_fp=True,
    morgan_radius=2,
    morgan_bits=1024,
    recommended_pretrained=['chemberta'],
    literature_refs=[
        "Lipinski et al., Adv. Drug Deliv. Rev. 2001",
    ]
)

PKA_PRESET = DescriptorPreset(
    name="pka",
    name_ja="酸解離定数(pKa)",
    category=PropertyCategory.PHARMACOLOGICAL,
    description="酸塩基平衡、pKaの予測",
    rationale="""
    pKaは酸性・塩基性官能基と電子効果に依存。
    UniPKa事前学習モデルが最適。
    """,
    rdkit_descriptors=[
        'NumHDonors',
        'NumHAcceptors',
        'NumHeteroatoms',
        'TPSA',
        'MaxPartialCharge',
        'MinPartialCharge',
    ],
    requires_xtb=True,
    xtb_descriptors=['dipole_norm'],
    recommended_pretrained=['unipka', 'unimol'],
    literature_refs=[
        "Liao et al., Nat. Commun. 2024 (UniPKa)",
    ]
)


# =============================================================================
# 汎用プリセット
# =============================================================================

GENERAL_PRESET = DescriptorPreset(
    name="general",
    name_ja="汎用（バランス型）",
    category=PropertyCategory.CHEMICAL,
    description="特定の物性を指定しない場合の推奨セット",
    rationale="""
    物性が不明な場合のバランスの取れた汎用セット。
    様々な物性との相関が報告されている基本記述子。
    """,
    rdkit_descriptors=[
        'MolWt',
        'MolLogP',
        'MolMR',
        'TPSA',
        'NumHDonors',
        'NumHAcceptors',
        'NumRotatableBonds',
        'NumAromaticRings',
        'FractionCSP3',
        'HeavyAtomCount',
        'RingCount',
        'LabuteASA',
        'Chi0', 'Chi1',
        'Kappa1', 'Kappa2',
        'BertzCT',
        'HallKierAlpha',
    ],
)


# =============================================================================
# プリセット辞書（公開API）
# =============================================================================

MATERIAL_PRESETS: Dict[str, DescriptorPreset] = {
    # 光学物性
    'refractive_index': REFRACTIVE_INDEX_PRESET,
    'optical_gap': OPTICAL_GAP_PRESET,
    
    # 機械物性
    'elastic_modulus': ELASTIC_MODULUS_PRESET,
    'tensile_strength': TENSILE_STRENGTH_PRESET,
    'hardness': HARDNESS_PRESET,
    
    # 熱物性
    'glass_transition': GLASS_TRANSITION_PRESET,
    'melting_point': MELTING_POINT_PRESET,
    'thermal_conductivity': THERMAL_CONDUCTIVITY_PRESET,
    'thermal_stability': THERMAL_STABILITY_PRESET,
    
    # 電気物性
    'dielectric_constant': DIELECTRIC_CONSTANT_PRESET,
    'conductivity': CONDUCTIVITY_PRESET,
    
    # 化学物性
    'viscosity': VISCOSITY_PRESET,
    'density': DENSITY_PRESET,
    'surface_tension': SURFACE_TENSION_PRESET,
    'solubility': SOLUBILITY_PRESET,
    
    # 輸送物性
    'gas_permeability': GAS_PERMEABILITY_PRESET,
    
    # 薬理学
    'admet': ADMET_PRESET,
    'pka': PKA_PRESET,
    
    # 汎用
    'general': GENERAL_PRESET,
}

# カテゴリ別インデックス
PRESETS_BY_CATEGORY: Dict[PropertyCategory, List[str]] = {}
for name, preset in MATERIAL_PRESETS.items():
    if preset.category not in PRESETS_BY_CATEGORY:
        PRESETS_BY_CATEGORY[preset.category] = []
    PRESETS_BY_CATEGORY[preset.category].append(name)


def list_presets() -> Dict[str, str]:
    """利用可能なプリセット一覧を取得"""
    return {name: preset.name_ja for name, preset in MATERIAL_PRESETS.items()}


def list_presets_by_category() -> Dict[str, Dict[str, str]]:
    """カテゴリ別のプリセット一覧"""
    result = {}
    for category, names in PRESETS_BY_CATEGORY.items():
        result[category.value] = {
            name: MATERIAL_PRESETS[name].name_ja for name in names
        }
    return result


def get_preset(name: str) -> Optional[DescriptorPreset]:
    """プリセットを取得"""
    return MATERIAL_PRESETS.get(name)


def get_all_unique_descriptors() -> Set[str]:
    """全プリセットで使用される記述子の和集合"""
    all_descs = set()
    for preset in MATERIAL_PRESETS.values():
        all_descs.update(preset.rdkit_descriptors)
        all_descs.update(preset.xtb_descriptors)
    return all_descs
