"""
計算化学エンジン - 構造最適化 & 一点計算

Implements: F-COMPCHEM-001
設計思想:
- RDKit/XTBによる構造最適化
- 一点エネルギー計算
- 振動解析
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalculationResult:
    """計算結果"""
    smiles: str
    method: str
    energy: Optional[float] = None  # Hartree
    dipole_moment: Optional[float] = None  # Debye
    homo: Optional[float] = None  # eV
    lumo: Optional[float] = None  # eV
    gap: Optional[float] = None  # eV
    coordinates: Optional[np.ndarray] = None  # 3D座標
    atom_symbols: Optional[List[str]] = None
    success: bool = True
    error_message: Optional[str] = None
    additional_properties: Dict[str, Any] = field(default_factory=dict)


class ComputationalChemistry:
    """
    計算化学エンジン
    
    Features:
    - 構造最適化（RDKit MMFF/UFF、XTB GFN2）
    - 一点計算（XTB）
    - 物性計算（HOMO/LUMO、双極子モーメント）
    
    Example:
        >>> chem = ComputationalChemistry()
        >>> result = chem.optimize("CCO", method="mmff")
        >>> sp_result = chem.single_point_calculation("CCO", method="gfn2")
    """
    
    def __init__(
        self,
        xtb_path: str = "xtb",
        working_dir: Optional[str] = None,
    ):
        """
        Args:
            xtb_path: XTB実行ファイルパス
            working_dir: 作業ディレクトリ
        """
        self.xtb_path = xtb_path
        self.working_dir = working_dir or tempfile.gettempdir()
    
    def optimize(
        self,
        smiles: str,
        method: Literal['mmff', 'uff', 'gfn2', 'gfnff'] = 'mmff',
        max_iterations: int = 500,
    ) -> CalculationResult:
        """
        構造最適化
        
        Args:
            smiles: SMILES文字列
            method: 最適化手法
            max_iterations: 最大反復回数
        
        Returns:
            CalculationResult
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return CalculationResult(
                    smiles=smiles, method=method,
                    success=False, error_message="Invalid SMILES"
                )
            
            # 水素付加
            mol = Chem.AddHs(mol)
            
            # 初期3D座標生成
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            result = AllChem.EmbedMolecule(mol, params)
            
            if result != 0:
                params.useRandomCoords = True
                result = AllChem.EmbedMolecule(mol, params)
                if result != 0:
                    return CalculationResult(
                        smiles=smiles, method=method,
                        success=False, error_message="3D embedding failed"
                    )
            
            # 力場最適化
            if method == 'mmff':
                converged = AllChem.MMFFOptimizeMolecule(
                    mol, maxIters=max_iterations
                )
            elif method == 'uff':
                converged = AllChem.UFFOptimizeMolecule(
                    mol, maxIters=max_iterations
                )
            elif method in ['gfn2', 'gfnff']:
                return self._optimize_with_xtb(mol, smiles, method)
            else:
                return CalculationResult(
                    smiles=smiles, method=method,
                    success=False, error_message=f"Unknown method: {method}"
                )
            
            # 座標抽出
            conf = mol.GetConformer()
            coords = np.array([
                [conf.GetAtomPosition(i).x,
                 conf.GetAtomPosition(i).y,
                 conf.GetAtomPosition(i).z]
                for i in range(mol.GetNumAtoms())
            ])
            
            atom_symbols = [a.GetSymbol() for a in mol.GetAtoms()]
            
            # エネルギー計算
            if method == 'mmff':
                ff = AllChem.MMFFGetMoleculeForceField(
                    mol, AllChem.MMFFGetMoleculeProperties(mol)
                )
                if ff:
                    energy = ff.CalcEnergy() / 627.5  # kcal/mol → Hartree
                else:
                    energy = None
            else:  # UFF
                ff = AllChem.UFFGetMoleculeForceField(mol)
                if ff:
                    energy = ff.CalcEnergy() / 627.5
                else:
                    energy = None
            
            return CalculationResult(
                smiles=smiles,
                method=method,
                energy=energy,
                coordinates=coords,
                atom_symbols=atom_symbols,
                success=True,
            )
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return CalculationResult(
                smiles=smiles, method=method,
                success=False, error_message=str(e)
            )
    
    def _optimize_with_xtb(
        self,
        mol,
        smiles: str,
        method: str,
    ) -> CalculationResult:
        """XTBで構造最適化"""
        from rdkit import Chem

        # XYZ形式で出力
        xyz_content = self._mol_to_xyz(mol)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            xyz_path = os.path.join(tmpdir, "input.xyz")
            with open(xyz_path, "w") as f:
                f.write(xyz_content)
            
            # XTBコマンド
            gfn = "2" if method == "gfn2" else "ff"
            cmd = [self.xtb_path, xyz_path, f"--gfn{gfn}", "--opt"]
            
            try:
                result = subprocess.run(
                    cmd, cwd=tmpdir, capture_output=True,
                    text=True, timeout=120
                )
                
                # 結果解析
                return self._parse_xtb_output(
                    result.stdout, smiles, method, tmpdir
                )
                
            except subprocess.TimeoutExpired:
                return CalculationResult(
                    smiles=smiles, method=method,
                    success=False, error_message="XTB timeout"
                )
            except FileNotFoundError:
                return CalculationResult(
                    smiles=smiles, method=method,
                    success=False, error_message="XTB not found"
                )
    
    def single_point_calculation(
        self,
        smiles: str,
        method: Literal['gfn2', 'gfnff', 'gfn1'] = 'gfn2',
    ) -> CalculationResult:
        """
        一点計算（Single Point Calculation）
        
        Args:
            smiles: SMILES文字列
            method: 計算手法
            
        Returns:
            CalculationResult（エネルギー、HOMO/LUMO等）
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            # まず構造最適化
            opt_result = self.optimize(smiles, method='mmff')
            if not opt_result.success or opt_result.coordinates is None:
                return CalculationResult(
                    smiles=smiles, method=method,
                    success=False, error_message="Pre-optimization failed"
                )
            
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol)
            
            xyz_content = self._mol_to_xyz(mol)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                xyz_path = os.path.join(tmpdir, "input.xyz")
                with open(xyz_path, "w") as f:
                    f.write(xyz_content)
                
                gfn = method.replace("gfn", "")
                cmd = [self.xtb_path, xyz_path, f"--gfn{gfn}"]
                
                try:
                    result = subprocess.run(
                        cmd, cwd=tmpdir, capture_output=True,
                        text=True, timeout=60
                    )
                    
                    return self._parse_xtb_output(
                        result.stdout, smiles, method, tmpdir
                    )
                    
                except subprocess.TimeoutExpired:
                    return CalculationResult(
                        smiles=smiles, method=method,
                        success=False, error_message="XTB timeout"
                    )
                except FileNotFoundError:
                    return CalculationResult(
                        smiles=smiles, method=method,
                        success=False, error_message="XTB not installed"
                    )
                    
        except Exception as e:
            logger.error(f"Single point calculation failed: {e}")
            return CalculationResult(
                smiles=smiles, method=method,
                success=False, error_message=str(e)
            )
    
    def _mol_to_xyz(self, mol) -> str:
        """RDKit MolをXYZ形式に変換"""
        from rdkit import Chem
        
        conf = mol.GetConformer()
        n_atoms = mol.GetNumAtoms()
        
        lines = [str(n_atoms), "Generated by ChemML"]
        
        for i in range(n_atoms):
            atom = mol.GetAtomWithIdx(i)
            pos = conf.GetAtomPosition(i)
            lines.append(f"{atom.GetSymbol():2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")
        
        return "\n".join(lines)
    
    def _parse_xtb_output(
        self,
        output: str,
        smiles: str,
        method: str,
        tmpdir: str,
    ) -> CalculationResult:
        """XTB出力を解析"""
        result = CalculationResult(smiles=smiles, method=method)
        
        for line in output.split('\n'):
            line_lower = line.lower().strip()
            
            # Total energy
            if 'total energy' in line_lower and 'eh' in line_lower:
                try:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p.lower() == 'eh' and i > 0:
                            result.energy = float(parts[i - 1])
                            break
                except (ValueError, IndexError):
                    pass
            
            # HOMO
            if 'homo' in line_lower and 'ev' in line_lower:
                try:
                    for part in line.split():
                        try:
                            val = float(part)
                            if -20 < val < 0:  # 妥当な範囲
                                result.homo = val
                                break
                        except ValueError:
                            continue
                except Exception:
                    pass
            
            # LUMO
            if 'lumo' in line_lower and 'ev' in line_lower:
                try:
                    for part in line.split():
                        try:
                            val = float(part)
                            if -5 < val < 10:
                                result.lumo = val
                                break
                        except ValueError:
                            continue
                except Exception:
                    pass
            
            # Dipole
            if 'molecular dipole' in line_lower or 'dipole moment' in line_lower:
                try:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if 'debye' in p.lower() and i > 0:
                            result.dipole_moment = float(parts[i - 1])
                            break
                except Exception:
                    pass
        
        # HOMO-LUMO gap
        if result.homo is not None and result.lumo is not None:
            result.gap = result.lumo - result.homo
        
        result.success = result.energy is not None
        
        return result
    
    def get_3d_viewer_html(
        self,
        smiles: str,
        style: str = 'stick',
        width: int = 400,
        height: int = 400,
    ) -> str:
        """
        Py3Dmol風の3D分子ビューアHTMLを生成
        """
        # まず構造最適化
        result = self.optimize(smiles, method='mmff')
        
        if not result.success or result.coordinates is None:
            return f"<p>3D構造生成に失敗しました: {result.error_message}</p>"
        
        # Mol形式で出力
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol)
            
            mol_block = Chem.MolToMolBlock(mol)
            
            # Py3Dmol HTML template
            html = f"""
            <script src="https://3dmol.org/build/3Dmol-min.js"></script>
            <div id="viewer_{id(self)}" style="width: {width}px; height: {height}px; position: relative;"></div>
            <script>
                let viewer = $3Dmol.createViewer('viewer_{id(self)}', {{backgroundColor: 'white'}});
                viewer.addModel(`{mol_block}`, 'mol');
                viewer.setStyle({{}}, {{{style}: {{}}}});
                viewer.zoomTo();
                viewer.render();
            </script>
            """
            
            return html
            
        except Exception as e:
            return f"<p>3D表示エラー: {e}</p>"
