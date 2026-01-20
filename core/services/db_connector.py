"""
データベースコネクタ（ChEMBL/PubChem inspired）

Implements: F-DB-001
設計思想:
- 外部DB接続
- クエリビルダー
- キャッシング
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json

logger = logging.getLogger(__name__)


@dataclass
class CompoundRecord:
    """化合物レコード"""
    id: str
    smiles: str
    name: Optional[str] = None
    properties: Dict[str, Any] = None
    source: str = ""


class ChEMBLConnector:
    """
    ChEMBL データベースコネクタ
    
    Features:
    - REST API接続
    - 化合物検索
    - 活性データ取得
    
    Example:
        >>> db = ChEMBLConnector()
        >>> compounds = db.search_by_smiles("CCO", similarity=0.8)
    """
    
    BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
    
    def __init__(self, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, Any] = {}
    
    def search_by_smiles(
        self,
        smiles: str,
        similarity: float = 0.7,
        limit: int = 10,
    ) -> List[CompoundRecord]:
        """SMILES類似検索"""
        try:
            import urllib.request
            import urllib.parse
            
            encoded_smiles = urllib.parse.quote(smiles)
            url = f"{self.BASE_URL}/similarity/{encoded_smiles}/{int(similarity*100)}.json?limit={limit}"
            
            cache_key = f"sim_{smiles}_{similarity}"
            if self.cache_enabled and cache_key in self._cache:
                return self._cache[cache_key]
            
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            records = []
            for mol in data.get('molecules', []):
                records.append(CompoundRecord(
                    id=mol.get('molecule_chembl_id', ''),
                    smiles=mol.get('molecule_structures', {}).get('canonical_smiles', ''),
                    name=mol.get('pref_name'),
                    source='ChEMBL',
                ))
            
            if self.cache_enabled:
                self._cache[cache_key] = records
            
            return records
            
        except Exception as e:
            logger.warning(f"ChEMBL search failed: {e}")
            return []
    
    def get_compound(self, chembl_id: str) -> Optional[CompoundRecord]:
        """ChEMBL IDで化合物取得"""
        try:
            import urllib.request
            
            url = f"{self.BASE_URL}/molecule/{chembl_id}.json"
            
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            return CompoundRecord(
                id=data.get('molecule_chembl_id', ''),
                smiles=data.get('molecule_structures', {}).get('canonical_smiles', ''),
                name=data.get('pref_name'),
                properties={
                    'mw': data.get('molecule_properties', {}).get('full_mwt'),
                    'logp': data.get('molecule_properties', {}).get('alogp'),
                },
                source='ChEMBL',
            )
            
        except Exception as e:
            logger.warning(f"ChEMBL get failed: {e}")
            return None


class PubChemConnector:
    """
    PubChem データベースコネクタ
    """
    
    BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    
    def search_by_name(self, name: str, limit: int = 5) -> List[CompoundRecord]:
        """名前で検索"""
        try:
            import urllib.request
            import urllib.parse
            
            encoded_name = urllib.parse.quote(name)
            url = f"{self.BASE_URL}/compound/name/{encoded_name}/property/CanonicalSMILES,MolecularWeight/JSON"
            
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            records = []
            for prop in data.get('PropertyTable', {}).get('Properties', [])[:limit]:
                records.append(CompoundRecord(
                    id=str(prop.get('CID', '')),
                    smiles=prop.get('CanonicalSMILES', ''),
                    name=name,
                    properties={'mw': prop.get('MolecularWeight')},
                    source='PubChem',
                ))
            
            return records
            
        except Exception as e:
            logger.warning(f"PubChem search failed: {e}")
            return []
    
    def get_by_cid(self, cid: int) -> Optional[CompoundRecord]:
        """CIDで取得"""
        try:
            import urllib.request
            
            url = f"{self.BASE_URL}/compound/cid/{cid}/property/CanonicalSMILES,IUPACName,MolecularWeight/JSON"
            
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            props = data.get('PropertyTable', {}).get('Properties', [{}])[0]
            
            return CompoundRecord(
                id=str(cid),
                smiles=props.get('CanonicalSMILES', ''),
                name=props.get('IUPACName'),
                properties={'mw': props.get('MolecularWeight')},
                source='PubChem',
            )
            
        except Exception as e:
            logger.warning(f"PubChem get failed: {e}")
            return None
