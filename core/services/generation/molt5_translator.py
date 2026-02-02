"""
MolT5分子翻訳エンジン

Implements: F-MOLT5-001
論文: Translation between Molecules and Natural Language (arXiv:2204.11817)

設計思想:
- T5ベースのSeq2Seq翻訳
- Text→Molecule（caption2smiles）
- Molecule→Text（smiles2caption）
- バッチ処理対応

引用:
Edwards, C., et al. (2022). "Translation between Molecules and Natural Language"
arXiv:2204.11817
https://arxiv.org/abs/2204.11817
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    """翻訳結果"""
    input_text: str
    output: str
    score: float = 0.0
    model_used: str = ""


class MolT5Translator:
    """
    MolT5翻訳エンジン
    
    Features:
    - Text → SMILES（分子生成）
    - SMILES → Text（説明生成）
    - バッチ処理
    - ビーム探索
    
    Example:
        >>> translator = MolT5Translator(task='caption2smiles')
        >>> result = translator.translate("aspirin-like molecule")
        >>> print(result.output)
        'CC(=O)Oc1ccccc1C(=O)O'
    
    論文引用:
        Edwards, C., et al. (2022). "Translation between Molecules and Natural Language"
        
        "MolT5 leverages the T5 architecture for translating between molecular 
        representations (SMILES) and natural language descriptions. The model is
        pre-trained on large corpora and fine-tuned for specific translation tasks."
        
        日本語訳：
        「MolT5はT5アーキテクチャを活用し、分子表現（SMILES）と自然言語記述の
        間を翻訳する。モデルは大規模コーパスで事前学習され、特定の翻訳タスクに
        ファインチューニングされる。」
    """
    
    # HuggingFaceモデル名
    MODELS = {
        'caption2smiles': 'laituan245/molt5-small-caption2smiles',
        'smiles2caption': 'laituan245/molt5-small-smiles2caption',
    }
    
    def __init__(
        self,
        task: str = 'caption2smiles',
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        初期化
        
        Args:
            task: 'caption2smiles' or 'smiles2caption'
            model_name: カスタムモデル名（Noneの場合デフォルト）
            cache_dir: モデルキャッシュディレクトリ
            device: 'cpu', 'cuda', None（自動検出）
        """
        if task not in self.MODELS and model_name is None:
            raise ValueError(f"Unknown task: {task}. Must be one of {list(self.MODELS.keys())}")
        
        self.task = task
        self.model_name = model_name or self.MODELS.get(task)
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/chemml/molt5")
        
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
        
        logger.info(f"MolT5Translator initialized: task={task}, device={self.device}")
    
    def _load_model(self) -> None:
        """モデル遅延ロード"""
        if self._is_loaded:
            return
        
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            
            logger.info(f"Loading MolT5 model: {self.model_name}")
            
            self._tokenizer = T5Tokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                legacy=False,
            )
            
            self._model = T5ForConditionalGeneration.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            )
            
            # デバイスへ移動
            self._model.to(self.device)
            self._model.eval()
            
            self._is_loaded = True
            logger.info(f"MolT5 model loaded successfully on {self.device}")
            
        except ImportError as e:
            logger.error(f"transformers not installed: {e}")
            raise ImportError(
                "transformers library required for MolT5. "
                "Install with: pip install transformers torch"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load MolT5 model: {e}")
            raise
    
    def translate(
        self,
        text: str,
        max_length: int = 512,
        num_beams: int = 5,
        num_return_sequences: int = 1,
        **kwargs
    ) -> TranslationResult:
        """
        単一テキスト翻訳
        
        Args:
            text: 入力テキスト
            max_length: 最大生成長
            num_beams: ビーム探索のビーム数
            num_return_sequences: 返却する候補数
        
        Returns:
            TranslationResult
        """
        self._load_model()
        
        try:
            import torch
            
            # トークン化
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True,
            ).to(self.device)
            
            # 生成
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    early_stopping=True,
                    **kwargs
                )
            
            # デコード
            decoded = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return TranslationResult(
                input_text=text,
                output=decoded.strip(),
                model_used=self.model_name,
            )
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return TranslationResult(
                input_text=text,
                output="",
                score=0.0,
                model_used=self.model_name,
            )
    
    def batch_translate(
        self,
        texts: List[str],
        max_length: int = 512,
        num_beams: int = 5,
        batch_size: int = 8,
        **kwargs
    ) -> List[TranslationResult]:
        """
        バッチ翻訳
        
        Args:
            texts: 入力テキストリスト
            max_length: 最大生成長
            num_beams: ビーム探索のビーム数
            batch_size: バッチサイズ
        
        Returns:
            List[TranslationResult]
        """
        self._load_model()
        
        results = []
        
        # バッチ処理
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                import torch
                
                # トークン化
                inputs = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                    padding=True,
                ).to(self.device)
                
                # 生成
                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_length=max_length,
                        num_beams=num_beams,
                        early_stopping=True,
                        **kwargs
                    )
                
                # デコード
                for j, output in enumerate(outputs):
                    decoded = self._tokenizer.decode(output, skip_special_tokens=True)
                    results.append(TranslationResult(
                        input_text=batch[j],
                        output=decoded.strip(),
                        model_used=self.model_name,
                    ))
                    
            except Exception as e:
                logger.error(f"Batch translation failed: {e}")
                # エラー時は空結果を追加
                for text in batch:
                    results.append(TranslationResult(
                        input_text=text,
                        output="",
                        score=0.0,
                        model_used=self.model_name,
                    ))
        
        return results
    
    def text_to_smiles(
        self,
        description: str,
        n_molecules: int = 5,
        **kwargs
    ) -> List[str]:
        """
        自然言語→SMILES変換（便利メソッド）
        
        Args:
            description: 分子の自然言語記述
            n_molecules: 生成候補数
        
        Returns:
            SMILESリスト
        """
        if self.task != 'caption2smiles':
            logger.warning(f"Task is {self.task}, not caption2smiles")
        
        result = self.translate(
            description,
            num_return_sequences=min(n_molecules, 10),
            **kwargs
        )
        
        # 単一結果の場合
        if result.output:
            return [result.output]
        return []
    
    def smiles_to_text(self, smiles: str, **kwargs) -> str:
        """
        SMILES→自然言語変換（便利メソッド）
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            自然言語説明
        """
        if self.task != 'smiles2caption':
            logger.warning(f"Task is {self.task}, not smiles2caption")
        
        result = self.translate(smiles, **kwargs)
        return result.output
    
    def __repr__(self) -> str:
        return f"MolT5Translator(task={self.task}, device={self.device})"
