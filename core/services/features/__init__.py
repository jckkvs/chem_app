"""
特徴量抽出モジュール

=== Smart Feature Engineering ===
物性×データセット特性に基づく最適特徴量生成

### コア機能
- SmartFeatureEngine: 統合特徴量エンジン（推奨）
- DatasetAnalyzer: データセット分子構造分析
- MATERIAL_PRESETS: 19物性別の記述子プリセット

### 特徴量抽出器
- RDKitFeatureExtractor: 分子記述子
- XTBFeatureExtractor: 量子化学記述子（要xtb）
- UMAFeatureExtractor: UMAP埋め込み
- TarteFeatureExtractor: Transformer表形式特徴量（要tarte-ai）

### 深層学習埋め込み
- PretrainedEmbeddingEngine: Uni-Mol/ChemBERTa
- SelfSupervisedEmbeddingEngine: GROVER/MolCLR/GraphMVP
- EquivariantEmbeddingEngine: SchNet/PaiNN (3D)

### 特徴量選択
- BorutaSelector, MRMRSelector, EnsembleFeatureSelector

### 分析ツール
- ScaffoldAnalyzer: 骨格分析
- ApplicabilityDomainAnalyzer: 予測信頼性
- MolecularSimilaritySearch: 類似度検索

### 分子生成
- ConditionalMolecularGenerator: 逆設計

### ユーティリティ
- SELFIESEncoder: SMILES⇔SELFIES変換
"""

# 基底クラス
from .base import BaseFeatureExtractor

# 従来の抽出器
from .rdkit_eng import RDKitFeatureExtractor
from .xtb_eng import XTBFeatureExtractor
from .uma_eng import UMAFeatureExtractor
from .tarte_eng import TarteFeatureExtractor, is_tarte_available

# スマート特徴量エンジン
from .smart_feature_engine import (
    SmartFeatureEngine,
    FeatureConfig,
    FeatureGenerationResult,
    generate_smart_features,
)

# データセット分析
from .dataset_analyzer import (
    DatasetAnalyzer,
    DatasetProfile,
    analyze_dataset,
)

# 事前学習モデル
from .pretrained_embeddings import (
    PretrainedEmbeddingEngine,
    get_pretrained_embeddings,
)

# 物性別プリセット
from .descriptor_presets import (
    MATERIAL_PRESETS,
    DescriptorPreset,
    list_presets,
    get_preset,
)

# 高度な特徴量選択
from .advanced_selectors import (
    BorutaSelector,
    MRMRSelector,
    PermutationImportanceSelector,
    EnsembleFeatureSelector,
    select_features,
)

# 骨格・化学空間分析
from .scaffold_analysis import (
    ScaffoldAnalyzer,
    ScaffoldProfile,
    ChemicalSpaceAnalyzer,
    ScaffoldSplitter,
    analyze_scaffolds,
    scaffold_split,
)

# Applicability Domain分析
from .applicability_domain import (
    ApplicabilityDomainAnalyzer,
    ADResult,
    DistanceBasedAD,
    LOFBasedAD,
    check_applicability_domain,
)

# 類似度検索
from .similarity_search import (
    MolecularSimilaritySearch,
    SimilaritySearchResult,
    ActivityCliffDetector,
    search_similar,
    detect_activity_cliffs,
)

# 等変GNN (3D)
from .equivariant_gnn import (
    EquivariantEmbeddingEngine,
    MolecularStructure,
    get_equivariant_embeddings,
)

# 自己教師あり学習
from .ssl_embeddings import (
    SelfSupervisedEmbeddingEngine,
    GROVEREmbedding,
    MolCLREmbedding,
    GraphMVPEmbedding,
    get_ssl_embeddings,
)

# SELFIES
from .selfies_support import (
    SELFIESEncoder,
    is_selfies_available,
    smiles_to_selfies,
    selfies_to_smiles,
)

# 分子生成
from .molecular_generation import (
    ConditionalMolecularGenerator,
    GenerationResult,
    InverseDesignPipeline,
    generate_molecules,
)

__all__ = [
    # 基底
    "BaseFeatureExtractor",
    
    # 従来の抽出器
    "RDKitFeatureExtractor",
    "XTBFeatureExtractor",
    "UMAFeatureExtractor",
    "TarteFeatureExtractor",
    "is_tarte_available",
    
    # スマート特徴量（推奨）
    "SmartFeatureEngine",
    "FeatureConfig",
    "FeatureGenerationResult",
    "generate_smart_features",
    
    # データセット分析
    "DatasetAnalyzer",
    "DatasetProfile",
    "analyze_dataset",
    
    # 事前学習モデル
    "PretrainedEmbeddingEngine",
    "get_pretrained_embeddings",
    
    # プリセット
    "MATERIAL_PRESETS",
    "DescriptorPreset",
    "list_presets",
    "get_preset",
    
    # 高度な特徴量選択
    "BorutaSelector",
    "MRMRSelector",
    "PermutationImportanceSelector",
    "EnsembleFeatureSelector",
    "select_features",
    
    # 骨格・化学空間分析
    "ScaffoldAnalyzer",
    "ScaffoldProfile",
    "ChemicalSpaceAnalyzer",
    "ScaffoldSplitter",
    "analyze_scaffolds",
    "scaffold_split",
    
    # AD分析
    "ApplicabilityDomainAnalyzer",
    "ADResult",
    "DistanceBasedAD",
    "LOFBasedAD",
    "check_applicability_domain",
    
    # 類似度検索
    "MolecularSimilaritySearch",
    "SimilaritySearchResult",
    "ActivityCliffDetector",
    "search_similar",
    "detect_activity_cliffs",
    
    # 等変GNN
    "EquivariantEmbeddingEngine",
    "MolecularStructure",
    "get_equivariant_embeddings",
    
    # 自己教師あり学習
    "SelfSupervisedEmbeddingEngine",
    "GROVEREmbedding",
    "MolCLREmbedding",
    "GraphMVPEmbedding",
    "get_ssl_embeddings",
    
    # SELFIES
    "SELFIESEncoder",
    "is_selfies_available",
    "smiles_to_selfies",
    "selfies_to_smiles",
    
    # 分子生成
    "ConditionalMolecularGenerator",
    "GenerationResult",
    "InverseDesignPipeline",
    "generate_molecules",
]


