"""
デモデータセット管理

ユーザーがデータなしでも試せるサンプルデータセットを提供
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DemoDatasetInfo:
    """デモデータセット情報"""
    id: str
    name: str
    name_ja: str
    description: str
    n_molecules: int
    smiles_column: str
    target_column: str
    task_type: str  # 'regression' or 'classification'
    difficulty: str  # 'beginner', 'intermediate', 'advanced'
    file_path: str


class DemoDatasets:
    """
    デモデータセット管理
    
    Features:
    - 初心者向けサンプルデータ
    - ワンクリックロード
    - 自動列検出
    
    Example:
        >>> demos = DemoDatasets()
        >>> df = demos.load('solubility')
    """
    
    DATASETS: Dict[str, DemoDatasetInfo] = {
        'solubility': DemoDatasetInfo(
            id='solubility',
            name='Water Solubility',
            name_ja='水溶解度予測',
            description='ESOL dataset - 1128分子の水溶解度データ',
            n_molecules=1128,
            smiles_column='SMILES',
            target_column='measured log solubility in mols per litre',
            task_type='regression',
            difficulty='beginner',
            file_path='static/demo/esol.csv'
        ),
        'toxicity': DemoDatasetInfo(
            id='toxicity',
            name='Toxicity Prediction',
            name_ja='毒性予測',
            description='Tox21 dataset - 200分子の毒性データ（サンプル）',
            n_molecules=200,
            smiles_column='smiles',
            target_column='NR-AR',
            task_type='classification',
            difficulty='beginner',
            file_path='static/demo/tox21_sample.csv'
        ),
        'lipophilicity': DemoDatasetInfo(
            id='lipophilicity',
            name='Lipophilicity',
            name_ja='脂溶性予測',
            description='脂溶性（logD）予測 - 300分子',
            n_molecules=300,
            smiles_column='smiles',
            target_column='exp',
            task_type='regression',
            difficulty='beginner',
            file_path='static/demo/lipophilicity.csv'
        ),
        'qed': DemoDatasetInfo(
            id='qed',
            name='Drug-likeness',
            name_ja='医薬品らしさ',
            description='QED (Quantitative Estimate of Drug-likeness) - 500分子',
            n_molecules=500,
            smiles_column='SMILES',
            target_column='QED',
            task_type='regression',
            difficulty='beginner',
            file_path='static/demo/qed.csv'
        ),
        'bace': DemoDatasetInfo(
            id='bace',
            name='BACE Inhibition',
            name_ja='BACE阻害活性',
            description='BACE-1阻害活性予測（アルツハイマー病）- 250分子',
            n_molecules=250,
            smiles_column='mol',
            target_column='Class',
            task_type='classification',
            difficulty='intermediate',
            file_path='static/demo/bace_sample.csv'
        )
    }
    
    @classmethod
    def list_all(cls) -> List[DemoDatasetInfo]:
        """すべてのデモデータセットをリスト"""
        return list(cls.DATASETS.values())
    
    @classmethod
    def get_info(cls, dataset_id: str) -> Optional[DemoDatasetInfo]:
        """データセット情報を取得"""
        return cls.DATASETS.get(dataset_id)
    
    @classmethod
    def load(cls, dataset_id: str) -> pd.DataFrame:
        """
        デモデータセットをロード
        
        Args:
            dataset_id: データセットID
        
        Returns:
            DataFrame
        
        Raises:
            ValueError: 存在しないデータセットID
        """
        info = cls.get_info(dataset_id)
        if info is None:
            available = ', '.join(cls.DATASETS.keys())
            raise ValueError(
                f"Dataset '{dataset_id}' not found. "
                f"Available: {available}"
            )
        
        try:
            # パスを解決（プロジェクトルートからの相対パス）
            file_path = Path(info.file_path)
            
            if not file_path.exists():
                logger.warning(f"Demo file not found: {file_path}")
                # フォールバック: ダミーデータ生成
                return cls._generate_dummy_data(info)
            
            df = pd.read_csv(file_path)
            logger.info(f"Loaded demo dataset: {dataset_id} ({len(df)} rows)")
            return df
        
        except Exception as e:
            logger.error(f"Failed to load demo dataset: {e}")
            return cls._generate_dummy_data(info)
    
    @classmethod
    def _generate_dummy_data(cls, info: DemoDatasetInfo) -> pd.DataFrame:
        """ダミーデータ生成（ファイルがない場合のフォールバック）"""
        import random
        
        # 簡単な SMILES例
        sample_smiles = [
            'CCO', 'CC(C)O', 'CCCC', 'c1ccccc1', 'CC(=O)O',
            'CC(C)CO', 'CCCCO', 'c1ccc(O)cc1', 'CC(C)(C)O', 'CCCCCO'
        ]
        
        n = min(info.n_molecules, 50)  # 最大50個のダミー
        
        data = {
            info.smiles_column: [random.choice(sample_smiles) for _ in range(n)]
        }
        
        if info.task_type == 'regression':
            data[info.target_column] = [random.uniform(-2, 5) for _ in range(n)]
        else:  # classification
            data[info.target_column] = [random.choice([0, 1]) for _ in range(n)]
        
        logger.warning(f"Using dummy data for {info.id}")
        return pd.DataFrame(data)
    
    @classmethod
    def get_by_difficulty(cls, difficulty: str) -> List[DemoDatasetInfo]:
        """難易度別にフィルタ"""
        return [
            info for info in cls.DATASETS.values()
            if info.difficulty == difficulty
        ]
    
    @classmethod
    def get_by_task_type(cls, task_type: str) -> List[DemoDatasetInfo]:
        """タスクタイプ別にフィルタ"""
        return [
            info for info in cls.DATASETS.values()
            if info.task_type == task_type
        ]
