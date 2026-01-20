"""
分子知識グラフ（Neo4j/RDKit inspired）

Implements: F-KG-001
設計思想:
- 分子関係グラフ
- 類似性エッジ
- パスファインディング
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MoleculeNode:
    """分子ノード"""
    id: str
    smiles: str
    properties: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)


@dataclass
class Edge:
    """エッジ"""
    source: str
    target: str
    relation: str  # 'similar', 'substructure', 'reaction_product', etc.
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)


class MolecularKnowledgeGraph:
    """
    分子知識グラフ
    
    Features:
    - 分子ノード管理
    - 類似性エッジ自動生成
    - パス検索
    
    Example:
        >>> kg = MolecularKnowledgeGraph()
        >>> kg.add_molecule("mol1", "CCO")
        >>> kg.add_similarity_edges(threshold=0.7)
        >>> path = kg.find_path("mol1", "mol2")
    """
    
    def __init__(self):
        self.nodes: Dict[str, MoleculeNode] = {}
        self.edges: List[Edge] = []
        self._adjacency: Dict[str, List[str]] = {}
    
    def add_molecule(
        self,
        id: str,
        smiles: str,
        properties: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None,
    ) -> MoleculeNode:
        """分子を追加"""
        node = MoleculeNode(
            id=id,
            smiles=smiles,
            properties=properties or {},
            tags=tags or set(),
        )
        self.nodes[id] = node
        self._adjacency[id] = []
        return node
    
    def add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        weight: float = 1.0,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Edge:
        """エッジを追加"""
        edge = Edge(
            source=source,
            target=target,
            relation=relation,
            weight=weight,
            properties=properties or {},
        )
        self.edges.append(edge)
        self._adjacency[source].append(target)
        self._adjacency[target].append(source)  # 無向グラフ
        return edge
    
    def add_similarity_edges(
        self,
        threshold: float = 0.7,
        fp_type: str = 'morgan',
    ) -> int:
        """類似性エッジを自動生成"""
        try:
            from core.services.features.fingerprint import FingerprintCalculator
            
            calc = FingerprintCalculator(fp_type=fp_type)
            node_list = list(self.nodes.values())
            added = 0
            
            for i, node1 in enumerate(node_list):
                for node2 in node_list[i+1:]:
                    sim = calc.tanimoto_similarity(node1.smiles, node2.smiles)
                    if sim >= threshold:
                        self.add_edge(
                            node1.id, node2.id,
                            relation='similar',
                            weight=sim,
                        )
                        added += 1
            
            return added
            
        except Exception as e:
            logger.warning(f"Failed to add similarity edges: {e}")
            return 0
    
    def find_path(
        self,
        source: str,
        target: str,
    ) -> Optional[List[str]]:
        """パス検索（BFS）"""
        if source not in self.nodes or target not in self.nodes:
            return None
        
        visited = {source}
        queue = [(source, [source])]
        
        while queue:
            current, path = queue.pop(0)
            
            if current == target:
                return path
            
            for neighbor in self._adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def get_neighbors(
        self,
        node_id: str,
        relation: Optional[str] = None,
    ) -> List[MoleculeNode]:
        """隣接ノードを取得"""
        if node_id not in self._adjacency:
            return []
        
        neighbor_ids = self._adjacency[node_id]
        
        if relation:
            neighbor_ids = [
                e.target if e.source == node_id else e.source
                for e in self.edges
                if (e.source == node_id or e.target == node_id) and e.relation == relation
            ]
        
        return [self.nodes[nid] for nid in neighbor_ids if nid in self.nodes]
    
    def get_subgraph(self, node_ids: List[str]) -> 'MolecularKnowledgeGraph':
        """サブグラフ抽出"""
        subgraph = MolecularKnowledgeGraph()
        
        for nid in node_ids:
            if nid in self.nodes:
                node = self.nodes[nid]
                subgraph.add_molecule(node.id, node.smiles, node.properties, node.tags)
        
        for edge in self.edges:
            if edge.source in node_ids and edge.target in node_ids:
                subgraph.add_edge(edge.source, edge.target, edge.relation, edge.weight)
        
        return subgraph
    
    def stats(self) -> Dict[str, Any]:
        """統計情報"""
        return {
            'n_nodes': len(self.nodes),
            'n_edges': len(self.edges),
            'avg_degree': len(self.edges) * 2 / max(len(self.nodes), 1),
        }
