"""
XTB量子化学記述子抽出エンジン - 完全実装版

Implements: F-003
設計思想:
- GFN2-xTBによる半経験的量子計算
- dipole_normの完全パース実装（既存の問題を修正）
- 適切なエラーハンドリングとログ

参考文献:
- GFN2-xTB: An Accurate and Broadly Parametrized Self-Consistent Tight-Binding 
  Quantum Chemical Method (Bannwarth et al., 2019)
- DOI: 10.1021/acs.jctc.8b01176
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from shutil import which
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


class XTBFeatureExtractor(BaseFeatureExtractor):
    """
    XTB (Extended Tight-Binding) 量子化学記述子抽出器
    
    Workflow:
    SMILES → RDKit 3D Embed → MMFF Optimize → XYZ → XTB → Properties
    
    Features:
    - GFN2-xTB法による電子構造計算
    - Total Energy, HOMO-LUMO Gap, Dipole Moment の抽出
    - 3D構造最適化のオプション
    - 並列処理対応
    
    Example:
        >>> extractor = XTBFeatureExtractor()
        >>> features = extractor.transform(['c1ccccc1', 'CCO'])
    """
    
    def __init__(
        self,
        xtb_path: str = 'xtb',
        optimize: bool = True,
        parallel: int = 1,
        timeout: int = 60,
        include_smiles: bool = True,
        **kwargs
    ):
        """
        Args:
            xtb_path: XTBバイナリのパス
            optimize: ジオメトリ最適化を行うか
            parallel: XTBの並列スレッド数
            timeout: タイムアウト秒数
            include_smiles: 出力にSMILESカラムを含めるか
        """
        super().__init__(**kwargs)
        self.xtb_path = xtb_path
        self.optimize = optimize
        self.parallel = parallel
        self.timeout = timeout
        self.include_smiles = include_smiles
        
        # XTBの利用可能性をチェック
        self._xtb_available = self._check_xtb_available()
        if not self._xtb_available:
            logger.warning(f"XTBバイナリが見つかりません: {xtb_path}")
    
    def _check_xtb_available(self) -> bool:
        """XTBバイナリが利用可能かチェック"""
        return which(self.xtb_path) is not None
    
    def _embed_3d(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """3D構造を生成"""
        mol = Chem.AddHs(mol)
        
        # ETKDG法で3D埋め込み
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        
        result = AllChem.EmbedMolecule(mol, params)
        if result != 0:
            # フォールバック: ランダム座標
            params.useRandomCoords = True
            result = AllChem.EmbedMolecule(mol, params)
            if result != 0:
                return None
        
        # MMFF力場で予備最適化
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        except Exception as e:
            logger.debug(f"MMFF最適化失敗: {e}")
        
        return mol
    
    def _run_xtb(self, xyz_path: str, workdir: str) -> Optional[str]:
        """XTB計算を実行"""
        cmd = [
            self.xtb_path,
            xyz_path,
            '--gfn', '2',
            '--parallel', str(self.parallel),
        ]
        
        if self.optimize:
            cmd.append('--opt')
        else:
            cmd.append('--sp')  # Single point
        
        try:
            result = subprocess.run(
                cmd,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            
            if result.returncode != 0:
                logger.debug(f"XTB実行エラー: {result.stderr[:200]}")
                return None
            
            return result.stdout
            
        except subprocess.TimeoutExpired:
            logger.warning("XTB計算がタイムアウト")
            return None
        except Exception as e:
            logger.debug(f"XTB実行例外: {e}")
            return None
    
    def _parse_xtb_output(self, output: str) -> Dict[str, float]:
        """
        XTB出力をパースして物性値を抽出
        
        抽出する値:
        - energy: 全エネルギー (Hartree)
        - homo_lumo_gap: HOMO-LUMO ギャップ (eV)
        - dipole_norm: 双極子モーメントのノルム (Debye)
        """
        data = {
            'energy': np.nan,
            'homo_lumo_gap': np.nan,
            'dipole_norm': np.nan,
        }
        
        lines = output.splitlines()
        in_dipole_section = False
        
        for i, line in enumerate(lines):
            # Total Energy
            # Format: "| TOTAL ENERGY              -14.3456789 Eh   |"
            if 'TOTAL ENERGY' in line and 'Eh' in line:
                try:
                    parts = line.split()
                    for j, p in enumerate(parts):
                        if p == 'Eh':
                            data['energy'] = float(parts[j-1])
                            break
                except (ValueError, IndexError):
                    pass
            
            # HOMO-LUMO Gap
            # Format: "| HOMO-LUMO GAP              5.123456789 eV   |"
            if 'HOMO-LUMO GAP' in line and 'eV' in line:
                try:
                    parts = line.split()
                    for j, p in enumerate(parts):
                        if p == 'eV':
                            data['homo_lumo_gap'] = float(parts[j-1])
                            break
                except (ValueError, IndexError):
                    pass
            
            # Dipole Moment - セクション検出
            # Format:
            # molecular dipole:
            #                  x           y           z       tot (Debye)
            #    q only:       0.000       0.000       0.000
            #    full:         0.123       0.456       0.789       0.912
            if 'molecular dipole:' in line.lower():
                in_dipole_section = True
                continue
            
            if in_dipole_section and line.strip().startswith('full:'):
                try:
                    parts = line.split()
                    # full: x y z tot
                    if len(parts) >= 5:
                        data['dipole_norm'] = float(parts[4])
                    in_dipole_section = False
                except (ValueError, IndexError):
                    pass
            
            # 代替パース: "total dipole moment" 形式
            if 'total dipole moment' in line.lower():
                try:
                    # "   │   total dipole moment      :    1.2345 Debye  │"
                    parts = line.split()
                    for j, p in enumerate(parts):
                        if p.lower() == 'debye':
                            data['dipole_norm'] = float(parts[j-1])
                            break
                except (ValueError, IndexError):
                    pass
        
        return data
    
    def _process_single(self, smi: str) -> Dict[str, float]:
        """単一SMILESを処理"""
        # デフォルト値
        default = {
            'energy': np.nan,
            'homo_lumo_gap': np.nan,
            'dipole_norm': np.nan,
        }
        
        # 分子オブジェクト生成
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            logger.debug(f"無効なSMILES: {smi}")
            return default
        
        # 3D構造生成
        mol_3d = self._embed_3d(mol)
        if mol_3d is None:
            logger.debug(f"3D埋め込み失敗: {smi}")
            return default
        
        # 一時ディレクトリでXTB実行
        with tempfile.TemporaryDirectory() as tmpdir:
            xyz_path = os.path.join(tmpdir, 'mol.xyz')
            
            try:
                Chem.MolToXYZFile(mol_3d, xyz_path)
            except Exception as e:
                logger.debug(f"XYZ出力失敗: {e}")
                return default
            
            # XTB実行
            output = self._run_xtb(xyz_path, tmpdir)
            if output is None:
                return default
            
            # パース
            return self._parse_xtb_output(output)
    
    def transform(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        SMILESリストからXTB記述子を抽出
        
        Args:
            smiles_list: SMILESのリスト
            
        Returns:
            pd.DataFrame: XTB記述子DataFrame
        """
        if not self._xtb_available:
            logger.warning("XTBが利用不可のため、ゼロ値を返します")
            return self._return_empty(smiles_list)
        
        features = []
        success_count = 0
        
        for i, smi in enumerate(smiles_list):
            if (i + 1) % 10 == 0:
                logger.info(f"XTB処理中: {i+1}/{len(smiles_list)}")
            
            result = self._process_single(smi)
            features.append(result)
            
            if not any(np.isnan(v) for v in result.values()):
                success_count += 1
        
        logger.info(f"XTB計算完了: {success_count}/{len(smiles_list)} 成功")
        
        df = pd.DataFrame(features)
        
        if self.include_smiles:
            df.insert(0, 'SMILES', smiles_list)
        
        return df
    
    def _return_empty(self, smiles_list: List[str]) -> pd.DataFrame:
        """XTB利用不可時の空DataFrame"""
        cols = ['energy', 'homo_lumo_gap', 'dipole_norm']
        df = pd.DataFrame(np.nan, index=range(len(smiles_list)), columns=cols)
        
        if self.include_smiles:
            df.insert(0, 'SMILES', smiles_list)
        
        return df
    
    @property
    def descriptor_names(self) -> List[str]:
        """記述子名のリスト"""
        return ['energy', 'homo_lumo_gap', 'dipole_norm']
    
    @property
    def is_available(self) -> bool:
        """XTBが利用可能か"""
        return self._xtb_available
