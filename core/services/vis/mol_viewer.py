"""
インタラクティブ分子可視化エンジン

Implements: F-VIS-MOL-001
設計思想:
- RDKitベースの2D/3D分子描画
- 記述子値のハイライト表示
- SVG/PNG出力対応
"""

from __future__ import annotations

import base64
import io
import logging
from typing import List, Dict, Optional, Union, Any

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D

logger = logging.getLogger(__name__)


class MoleculeViewer:
    """
    分子構造可視化エンジン
    
    Features:
    - 2D構造描画（SVG/PNG）
    - 3D座標生成
    - 原子ハイライト
    - グリッド表示
    
    Example:
        >>> viewer = MoleculeViewer()
        >>> svg = viewer.draw_2d("c1ccccc1")
        >>> grid_svg = viewer.draw_grid(["CCO", "c1ccccc1", "CC(=O)O"])
    """
    
    def __init__(
        self,
        width: int = 400,
        height: int = 300,
        bond_line_width: float = 2.0,
    ):
        self.width = width
        self.height = height
        self.bond_line_width = bond_line_width
    
    def smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """SMILESをMolオブジェクトに変換"""
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            AllChem.Compute2DCoords(mol)
        return mol
    
    def draw_2d(
        self,
        smiles: str,
        highlight_atoms: Optional[List[int]] = None,
        highlight_bonds: Optional[List[int]] = None,
        legend: str = "",
        output_format: str = "svg",
    ) -> Optional[str]:
        """
        2D分子構造を描画
        
        Args:
            smiles: SMILES文字列
            highlight_atoms: ハイライトする原子インデックス
            highlight_bonds: ハイライトする結合インデックス
            legend: キャプション
            output_format: 'svg' or 'png'
            
        Returns:
            SVG文字列またはbase64エンコードPNG
        """
        mol = self.smiles_to_mol(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
            return None
        
        if output_format == "svg":
            drawer = rdMolDraw2D.MolDraw2DSVG(self.width, self.height)
        else:
            drawer = rdMolDraw2D.MolDraw2DCairo(self.width, self.height)
        
        # 描画オプション
        opts = drawer.drawOptions()
        opts.bondLineWidth = self.bond_line_width
        opts.addStereoAnnotation = True
        
        # ハイライト設定
        highlight_atom_map = {}
        highlight_bond_map = {}
        
        if highlight_atoms:
            for idx in highlight_atoms:
                highlight_atom_map[idx] = (1.0, 0.5, 0.5)  # 赤系
        
        if highlight_bonds:
            for idx in highlight_bonds:
                highlight_bond_map[idx] = (0.5, 0.5, 1.0)  # 青系
        
        drawer.DrawMolecule(
            mol,
            legend=legend,
            highlightAtoms=highlight_atoms or [],
            highlightAtomColors=highlight_atom_map if highlight_atoms else {},
            highlightBonds=highlight_bonds or [],
            highlightBondColors=highlight_bond_map if highlight_bonds else {},
        )
        drawer.FinishDrawing()
        
        if output_format == "svg":
            return drawer.GetDrawingText()
        else:
            png_data = drawer.GetDrawingText()
            return base64.b64encode(png_data).decode("utf-8")
    
    def draw_grid(
        self,
        smiles_list: List[str],
        legends: Optional[List[str]] = None,
        mols_per_row: int = 4,
        sub_img_size: tuple = (200, 200),
    ) -> Optional[str]:
        """
        複数分子をグリッド表示
        
        Args:
            smiles_list: SMILESリスト
            legends: 各分子のラベル
            mols_per_row: 1行あたりの分子数
            sub_img_size: 各分子の画像サイズ
            
        Returns:
            SVG文字列
        """
        mols = []
        valid_legends = []
        
        for i, smi in enumerate(smiles_list):
            mol = self.smiles_to_mol(smi)
            if mol:
                mols.append(mol)
                if legends and i < len(legends):
                    valid_legends.append(legends[i])
                else:
                    valid_legends.append(smi[:20])
        
        if not mols:
            return None
        
        try:
            img = Draw.MolsToGridImage(
                mols,
                molsPerRow=mols_per_row,
                subImgSize=sub_img_size,
                legends=valid_legends,
                returnPNG=False,  # PIL Image
            )
            
            # PIL ImageをSVGに変換
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            
            return f'<img src="data:image/png;base64,{png_b64}" />'
            
        except Exception as e:
            logger.error(f"Grid drawing failed: {e}")
            return None
    
    def draw_with_descriptors(
        self,
        smiles: str,
        descriptors: Dict[str, float],
        top_n: int = 5,
    ) -> str:
        """
        記述子情報付きで分子を描画
        
        Args:
            smiles: SMILES文字列
            descriptors: 記述子名と値の辞書
            top_n: 表示する記述子数
            
        Returns:
            HTML文字列（SVG + 記述子テーブル）
        """
        svg = self.draw_2d(smiles, output_format="svg") or ""
        
        # 記述子テーブル
        sorted_desc = sorted(descriptors.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        
        table_rows = "".join([
            f"<tr><td>{name}</td><td>{value:.4f}</td></tr>"
            for name, value in sorted_desc
        ])
        
        html = f"""
        <div style="display: flex; align-items: flex-start; gap: 20px;">
            <div>{svg}</div>
            <div>
                <table style="border-collapse: collapse; font-size: 12px;">
                    <thead>
                        <tr><th style="text-align: left;">Descriptor</th><th>Value</th></tr>
                    </thead>
                    <tbody>{table_rows}</tbody>
                </table>
            </div>
        </div>
        """
        
        return html
    
    def generate_3d_coords(self, smiles: str) -> Optional[str]:
        """
        3D座標を生成してMolブロックで返す
        
        Returns:
            MOL block文字列（3D座標付き）
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        mol = Chem.AddHs(mol)
        
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        
        result = AllChem.EmbedMolecule(mol, params)
        if result != 0:
            params.useRandomCoords = True
            result = AllChem.EmbedMolecule(mol, params)
            if result != 0:
                return None
        
        AllChem.MMFFOptimizeMolecule(mol)
        
        return Chem.MolToMolBlock(mol)
