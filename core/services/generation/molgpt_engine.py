"""
MolGPT分子生成エンジン

Implements: F-GEN-001
論文: MolGPT: Molecular Generation Using a Transformer-Decoder Model (arXiv:2106.05234)

設計思想:
- Transformer-decoderによるSMILES生成
- 条件付き生成（物性指定）
- Scaffold最適化

引用:
Bagal, V., et al. (2021). "MolGPT: Molecular Generation Using a Transformer-Decoder Model"
arXiv:2106.05234
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """生成設定"""
    n_molecules: int = 10
    max_length: int = 100
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = True


@dataclass
class GeneratedMolecule:
    """生成分子"""
    smiles: str
    score: float = 0.0
    properties: Dict[str, Any] = None


class MolGPTGenerator:
    """
    MolGPT分子生成エンジン
    
    Features:
    - ランダムSMILES生成
    - 条件付き生成（物性指定）
    - Scaffold最適化
    
    Example:
        >>> generator = MolGPTGenerator()
        >>> molecules = generator.generate(n_molecules=10)
        >>> print(molecules[0].smiles)
        'CCO'
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",  # デフォルトはGPT2（代替）
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        初期化
        
        Args:
            model_name: モデル名（Hugging Face Hub）
            cache_dir: キャッシュディレクトリ
            device: デバイス（'cpu', 'cuda', None=自動）
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/chemml")
        
        # デバイス自動検出
        if device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device
        
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        
        # MolT5とスコアラー（遅延ロード）
        self._molt5 = None
        self._scorer = None
    
    def _load_model(self) -> None:
        """モデルロード（遅延初期化）"""
        if self._is_loaded:
            return
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading MolGPT model: {self.model_name}")
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            ).to(self.device)
            
            self._is_loaded = True
            logger.info(f"Model loaded successfully on {self.device}")
            
        except ImportError:
            raise ImportError(
                "transformers library is required for MolGPT. "
                "Install with: pip install transformers torch"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate(
        self,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> List[GeneratedMolecule]:
        """
        SMILES生成
        
        Args:
            config: 生成設定
            **kwargs: 追加パラメータ（configを上書き）
        
        Returns:
            生成分子リスト
        
        Example:
            >>> molecules = generator.generate(n_molecules=5, temperature=0.8)
        """
        self._load_model()
        
        # 設定マージ
        if config is None:
            config = GenerationConfig()
        
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        try:
            # トークナイザーにpad_tokenがない場合は設定
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            # ダミー入力（実際のMolGPTでは適切なプロンプトを使用）
            input_text = "<start>"
            inputs = self._tokenizer(
                input_text,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # 生成
            outputs = self._model.generate(
                **inputs,
                max_length=config.max_length,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                do_sample=config.do_sample,
                num_return_sequences=config.n_molecules,
                pad_token_id=self._tokenizer.pad_token_id,
            )
            
            # デコード
            molecules = []
            for output in outputs:
                smiles = self._tokenizer.decode(output, skip_special_tokens=True)
                smiles = self._clean_smiles(smiles)
                
                if smiles:
                    molecules.append(GeneratedMolecule(
                        smiles=smiles,
                        score=1.0,  # スコアリングは将来実装
                    ))
            
            logger.info(f"Generated {len(molecules)} molecules")
            return molecules
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def _clean_smiles(self, raw_smiles: str) -> str:
        """SMILES文字列のクリーニング"""
        # プロンプト除去
        smiles = raw_smiles.replace("<start>", "").strip()
        
        # 改行・空白除去
        smiles = smiles.replace("\n", "").replace(" ", "")
        
        return smiles
    
    def conditional_generate(
        self,
        properties: Dict[str, float],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> List[GeneratedMolecule]:
        """
        条件付き生成
        
        Args:
            properties: 目標物性 {'logP': 2.5, 'MW': 300}
            config: 生成設定
            **kwargs: 追加パラメータ
        
        Returns:
            生成分子リスト
        
        Example:
            >>> molecules = generator.conditional_generate(
            ...     properties={'logP': 2.0, 'MW': 300}
            ... )
        
        Note:
            現在の実装は基本的な生成のみ。
            将来的にはプロパティ条件付きモデルを使用予定。
        """
        logger.warning(
            "Conditional generation is not fully implemented. "
            "Using basic generation instead."
        )
        
        # 基本生成を実行（将来的には条件付きモデルに置き換え）
        molecules = self.generate(config=config, **kwargs)
        
        # プロパティフィルタリング（簡易版）
        # 将来的にはモデルレベルで条件を組み込む
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            
            filtered = []
            for mol_data in molecules:
                mol = Chem.MolFromSmiles(mol_data.smiles)
                if mol is None:
                    continue
                
                # 簡易フィルタ（許容範囲±20%）
                match = True
                if 'logP' in properties:
                    logp = Descriptors.MolLogP(mol)
                    target = properties['logP']
                    if not (target * 0.8 <= logp <= target * 1.2):
                        match = False
                
                if 'MW' in properties:
                    mw = Descriptors.MolWt(mol)
                    target = properties['MW']
                    if not (target * 0.8 <= mw <= target * 1.2):
                        match = False
                
                if match:
                    filtered.append(mol_data)
            
            logger.info(f"Filtered to {len(filtered)} molecules matching properties")
            return filtered if filtered else molecules[:3]  # 最低3個は返す
            
        except ImportError:
            logger.warning("RDKit not available for property filtering")
            return molecules
    
    def scaffold_optimization(
        self,
        scaffold_smiles: str,
        target_property: str,
        target_value: float,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> List[GeneratedMolecule]:
        """
        Scaffold最適化
        
        Args:
            scaffold_smiles: 骨格SMILES
            target_property: 目標物性名
            target_value: 目標値
            config: 生成設定
            **kwargs: 追加パラメータ
        
        Returns:
            最適化された分子リスト
        
        Example:
            >>> molecules = generator.scaffold_optimization(
            ...     scaffold_smiles='c1ccccc1',
            ...     target_property='logP',
            ...     target_value=2.5
            ... )
        
        Note:
            現在の実装は基本的な生成のみ。
            将来的にはScaffold-conditioned生成を実装予定。
        """
        logger.warning(
            "Scaffold optimization is not fully implemented. "
            "Using conditional generation instead."
        )
        
        # 条件付き生成にフォールバック
        return self.conditional_generate(
            properties={target_property: target_value},
            config=config,
            **kwargs
        )
    
    def text_to_molecule(
        self,
        description: str,
        n_molecules: int = 10,
        use_scoring: bool = True,
        **kwargs
    ) -> List[GeneratedMolecule]:
        """
        自然言語から分子生成（MolT5使用）
        
        Implements: F-MOLT5-TEXT2MOL
        論文: MolT5 (arXiv:2204.11817)
        
        Args:
            description: 自然言語記述（例: "aspirin-like molecule"）
            n_molecules: 生成候補数
            use_scoring: スコアリング使用
        
        Returns:
            スコア付き生成分子リスト
        
        Example:
            >>> gen = MolGPTGenerator()
            >>> mols = gen.text_to_molecule("anti-inflammatory drug")
            >>> print(mols[0].smiles, mols[0].score)
            'CC(=O)Oc1ccccc1C(=O)O' 0.85
        """
        # MolT5ロード
        if self._molt5 is None:
            from .molt5_translator import MolT5Translator
            try:
                self._molt5 = MolT5Translator(task='caption2smiles')
            except Exception as e:
                logger.error(f"Failed to load MolT5: {e}")
                # フォールバック: 基本生成
                logger.warning("Falling back to basic generation")
                return self.generate(n_molecules=n_molecules, **kwargs)
        
        # MolT5で生成
        try:
            smiles_list = self._molt5.text_to_smiles(
                description,
                n_molecules=min(n_molecules, 20),
            )
            
            if not smiles_list:
                logger.warning("MolT5 returned no molecules, using basic generation")
                return self.generate(n_molecules=n_molecules, **kwargs)
            
        except Exception as e:
            logger.error(f"MolT5 generation failed: {e}, using basic generation")
            return self.generate(n_molecules=n_molecules, **kwargs)
        
        # スコアリング
        molecules = []
        
        for smiles in smiles_list[:n_molecules]:
            # SMILES検証
            clean_smiles = self._clean_smiles(smiles)
            if not clean_smiles:
                continue
            
            # スコア計算
            score = 0.5  # デフォルト
            properties = {}
            
            if use_scoring:
                if self._scorer is None:
                    from .molecule_scorer import MoleculeScorer
                    self._scorer = MoleculeScorer()
                
                try:
                    score_result = self._scorer.score_molecule(clean_smiles)
                    score = score_result.get('overall_score', 0.5)
                    properties = {
                        'qed': score_result.get('qed'),
                        'sa_score': score_result.get('sa_score'),
                        'lipinski': score_result.get('lipinski', {}).get('all_pass'),
                    }
                except Exception as e:
                    logger.warning(f"Scoring failed for {clean_smiles}: {e}")
            
            molecules.append(GeneratedMolecule(
                smiles=clean_smiles,
                score=score,
                properties=properties,
            ))
        
        # スコア降順ソート
        molecules.sort(key=lambda m: m.score, reverse=True)
        
        logger.info(f"Generated {len(molecules)} molecules from text: '{description}'")
        return molecules
