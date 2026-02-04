"""
モデル管理・ダウンロードマネージャー

Implements: F-MODEL-MANAGER-001
設計思想:
- プロキシ対応（Zscaler等）
- 手動ダウンロード対応
- ローカルキャッシュ管理
- 企業環境対応
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ModelManager:
    """
    外部モデル・Weightダウンロード管理
    
    Features:
    - プロキシ設定対応
    - 手動ダウンロード検出
    - ローカルキャッシュ管理
    - ダウンロード状態追跡
    
    Example:
        >>> manager = ModelManager()
        >>> manager.set_proxy('http://proxy.company.com:8080')
        >>> manager.download_model('unimol', version='v1.0')
    """
    
    # デフォルトのモデルキャッシュディレクトリ
    DEFAULT_CACHE_DIR = Path.home() / '.chemml' / 'models'
    
    # サポートするモデル定義
    MODELS = {
        'unimol': {
            'name': 'Uni-Mol',
            'description': '3D事前学習分子モデル',
            'url': 'https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/mol_pre_all_h_220816.pt',
            'filename': 'unimol_v0.1.pt',
            'size_mb': 850,
            'manual_url': 'https://github.com/dptech-corp/Uni-Mol/releases',
            'type': 'direct',
        },
        'chemberta': {
            'name': 'ChemBERTa',
            'description': 'BERT for Chemistry',
            'url': 'https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1',
            'filename': 'chemberta',  # ディレクトリ
            'size_mb': 450,
            'manual_url': 'https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1/tree/main',
            'type': 'huggingface',
        },
        'grover': {
            'name': 'GROVER',
            'description': 'Graph Representation Learning',
            'url': 'https://github.com/tencent-ailab/grover/releases/download/v1.0/grover_large.pt',
            'filename': 'grover_large.pt',
            'size_mb': 320,
            'manual_url': 'https://github.com/tencent-ailab/grover/releases',
            'type': 'direct',
        },
        'molclr': {
            'name': 'MolCLR',
            'description': 'Self-Supervised Molecular Learning',
            'url': 'https://github.com/yuyangw/MolCLR/releases/download/v1.0/model.pth',
            'filename': 'molclr_v1.0.pth',
            'size_mb': 180,
            'manual_url': 'https://github.com/yuyangw/MolCLR/releases',
            'type': 'direct',
        },
        'tarte': {
            'name': 'TARTE',
            'description': '分子Transformer',
            'url': 'https://huggingface.co/mizuno-group/tarte-base',
            'filename': 'tarte',  # ディレクトリ
            'size_mb': 250,
            'manual_url': 'https://huggingface.co/mizuno-group/tarte-base/tree/main',
            'type': 'huggingface',
        },
        'tabpfn': {
            'name': 'TabPFN',
            'description': 'Tabular Prior-Data Fitted Networks',
            'url': 'https://huggingface.co/TabPFN/TabPFN',
            'filename': 'tabpfn',  # ディレクトリ
            'size_mb': 150,
            'manual_url': 'https://huggingface.co/TabPFN/TabPFN/tree/main',
            'type': 'huggingface',
        },
    }
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Args:
            cache_dir: キャッシュディレクトリ（Noneならデフォルト）
        """
        self.cache_dir = cache_dir or self.DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # プロキシ設定
        self.proxy_http = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
        self.proxy_https = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
        
        # HuggingFace transformersのキャッシュディレクトリも設定
        # これにより、transformersが自動ダウンロードする際もこのディレクトリを使う
        os.environ['TRANSFORMERS_CACHE'] = str(self.cache_dir / 'transformers')
        os.environ['HF_HOME'] = str(self.cache_dir / 'huggingface')
        
        logger.info(f"ModelManager初期化: cache_dir={self.cache_dir}")
        logger.info(f"HuggingFaceキャッシュ: {os.environ['HF_HOME']}")
        if self.proxy_http:
            logger.info(f"HTTPプロキシ検出: {self.proxy_http}")
        if self.proxy_https:
            logger.info(f"HTTPSプロキシ検出: {self.proxy_https}")
    
    def set_proxy(
        self, 
        http_proxy: Optional[str] = None, 
        https_proxy: Optional[str] = None
    ):
        """
        プロキシを設定
        
        Args:
            http_proxy: HTTPプロキシURL (例: http://proxy.company.com:8080)
            https_proxy: HTTPSプロキシURL
        """
        if http_proxy:
            self.proxy_http = http_proxy
            os.environ['HTTP_PROXY'] = http_proxy
            os.environ['http_proxy'] = http_proxy
            logger.info(f"HTTPプロキシ設定: {http_proxy}")
        
        if https_proxy:
            self.proxy_https = https_proxy
            os.environ['HTTPS_PROXY'] = https_proxy
            os.environ['https_proxy'] = https_proxy
            logger.info(f"HTTPSプロキシ設定: {https_proxy}")
    
    def set_ssl_cert(self, cert_path: Optional[str] = None):
        """
        SSL証明書パスを設定（Zscaler等）
        
        Args:
            cert_path: 証明書ファイルパス（.pem, .crt等）
        """
        if cert_path:
            if not Path(cert_path).exists():
                logger.warning(f"証明書ファイルが見つかりません: {cert_path}")
                return False
            
            # requestsやurllib用
            os.environ['REQUESTS_CA_BUNDLE'] = cert_path
            os.environ['CURL_CA_BUNDLE'] = cert_path
            os.environ['SSL_CERT_FILE'] = cert_path
            
            logger.info(f"SSL証明書パス設定: {cert_path}")
            return True
        return False
    
    def disable_ssl_verify(self):
        """SSL検証を無効化（非推奨：開発/テスト用）"""
        os.environ['CURL_CA_BUNDLE'] = ''
        os.environ['REQUESTS_CA_BUNDLE'] = ''
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        logger.warning("SSL検証を無効化しました（セキュリティリスクあり）")
    
    def get_proxy_config(self) -> Dict[str, Optional[str]]:
        """現在のプロキシ設定を取得"""
        return {
            'http_proxy': self.proxy_http,
            'https_proxy': self.proxy_https,
            'ssl_cert_file': os.environ.get('SSL_CERT_FILE'),
            'ssl_verify': os.environ.get('REQUESTS_CA_BUNDLE', '') != '',
        }
    
    def configure_conda_proxy(
        self,
        http_proxy: Optional[str] = None,
        https_proxy: Optional[str] = None,
        ssl_verify: bool = True,
    ) -> Tuple[bool, str]:
        """
        Conda用プロキシ設定（.condarcに書き込み）
        
        Args:
            http_proxy: HTTPプロキシ
            https_proxy: HTTPSプロキシ
            ssl_verify: SSL検証を有効化
            
        Returns:
            (成功フラグ, メッセージ)
        """
        try:
            import yaml
            
            condarc_path = Path.home() / '.condarc'
            
            # 既存設定読み込み
            if condarc_path.exists():
                with open(condarc_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
            else:
                config = {}
            
            # プロキシ設定
            if http_proxy or https_proxy:
                config['proxy_servers'] = {
                    'http': http_proxy or self.proxy_http,
                    'https': https_proxy or self.proxy_https,
                }
            
            # SSL設定
            config['ssl_verify'] = ssl_verify
            
            # 書き込み
            with open(condarc_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.info(f"Conda設定を更新: {condarc_path}")
            return True, f".condarcを更新しました: {condarc_path}"
        
        except ImportError:
            msg = "PyYAMLが必要です: pip install pyyaml"
            logger.error(msg)
            return False, msg
        
        except Exception as e:
            logger.error(f"Conda設定失敗: {e}")
            return False, f"設定エラー: {str(e)}"
    
    def configure_pip_proxy(
        self,
        http_proxy: Optional[str] = None,
        https_proxy: Optional[str] = None,
        trusted_host: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        pip用プロキシ設定（pip.confに書き込み）
        
        Args:
            http_proxy: HTTPプロキシ
            https_proxy: HTTPSプロキシ
            trusted_host: 信頼するホスト（SSL検証スキップ）
            
        Returns:
            (成功フラグ, メッセージ)
        """
        try:
            # pip設定ディレクトリ
            if os.name == 'nt':  # Windows
                pip_config_dir = Path.home() / 'pip'
            else:  # Linux/Mac
                pip_config_dir = Path.home() / '.pip'
            
            pip_config_dir.mkdir(exist_ok=True)
            pip_config_path = pip_config_dir / 'pip.conf' if os.name != 'nt' else pip_config_dir / 'pip.ini'
            
            # 設定内容
            config_lines = ['[global]']
            
            if http_proxy or self.proxy_http:
                proxy = http_proxy or self.proxy_http
                config_lines.append(f'proxy = {proxy}')
            
            if trusted_host:
                config_lines.append(f'trusted-host = {trusted_host}')
            
            # 書き込み
            with open(pip_config_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(config_lines))
            
            logger.info(f"pip設定を更新: {pip_config_path}")
            return True, f"pip設定を更新しました: {pip_config_path}"
        
        except Exception as e:
            logger.error(f"pip設定失敗: {e}")
            return False, f"設定エラー: {str(e)}"
    
    def export_env_script(
        self,
        output_path: Optional[Path] = None
    ) -> Tuple[bool, str]:
        """
        環境変数設定スクリプトをエクスポート
        
        Args:
            output_path: 出力先パス（Noneなら自動生成）
            
        Returns:
            (成功フラグ, ファイルパス)
        """
        if output_path is None:
            output_path = Path.home() / 'chemml_proxy_env.bat' if os.name == 'nt' else Path.home() / 'chemml_proxy_env.sh'
        
        try:
            if os.name == 'nt':  # Windows
                lines = [
                    '@echo off',
                    'REM ChemML Proxy Environment Settings',
                    'REM Generated by ChemML Platform',
                    '',
                ]
                if self.proxy_http:
                    lines.append(f'set HTTP_PROXY={self.proxy_http}')
                    lines.append(f'set http_proxy={self.proxy_http}')
                if self.proxy_https:
                    lines.append(f'set HTTPS_PROXY={self.proxy_https}')
                    lines.append(f'set https_proxy={self.proxy_https}')
                
                ssl_cert = os.environ.get('SSL_CERT_FILE')
                if ssl_cert:
                    lines.append(f'set SSL_CERT_FILE={ssl_cert}')
                    lines.append(f'set REQUESTS_CA_BUNDLE={ssl_cert}')
                    lines.append(f'set CURL_CA_BUNDLE={ssl_cert}')
                
                lines.append('')
                lines.append('echo Proxy environment variables set!')
            
            else:  # Linux/Mac
                lines = [
                    '#!/bin/bash',
                    '# ChemML Proxy Environment Settings',
                    '# Generated by ChemML Platform',
                    '',
                ]
                if self.proxy_http:
                    lines.append(f'export HTTP_PROXY="{self.proxy_http}"')
                    lines.append(f'export http_proxy="{self.proxy_http}"')
                if self.proxy_https:
                    lines.append(f'export HTTPS_PROXY="{self.proxy_https}"')
                    lines.append(f'export https_proxy="{self.proxy_https}"')
                
                ssl_cert = os.environ.get('SSL_CERT_FILE')
                if ssl_cert:
                    lines.append(f'export SSL_CERT_FILE="{ssl_cert}"')
                    lines.append(f'export REQUESTS_CA_BUNDLE="{ssl_cert}"')
                    lines.append(f'export CURL_CA_BUNDLE="{ssl_cert}"')
                
                lines.append('')
                lines.append('echo "Proxy environment variables set!"')
            
            # 書き込み
            with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
                f.write('\n'.join(lines))
            
            # 実行権限付与（Unix系）
            if os.name != 'nt':
                os.chmod(output_path, 0o755)
            
            logger.info(f"環境変数スクリプトをエクスポート: {output_path}")
            return True, str(output_path)
        
        except Exception as e:
            logger.error(f"スクリプトエクスポート失敗: {e}")
            return False, str(e)
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        利用可能なモデル一覧を取得
        
        Returns:
            モデル情報のリスト（ダウンロード状態含む）
        """
        models = []
        for model_id, info in self.MODELS.items():
            model_path = self.cache_dir / info['filename']
            
            status = 'not_downloaded'
            if model_path.exists():
                if model_path.is_dir():
                    # ディレクトリ型（HuggingFace等）
                    if (model_path / 'config.json').exists():
                        status = 'downloaded'
                    else:
                        status = 'incomplete'
                else:
                    # ファイル型
                    status = 'downloaded'
            
            models.append({
                'id': model_id,
                'name': info['name'],
                'description': info['description'],
                'size_mb': info['size_mb'],
                'status': status,
                'path': str(model_path),
                'manual_url': info['manual_url'],
            })
        
        return models
    
    def get_model_path(self, model_id: str) -> Optional[Path]:
        """
        モデルのローカルパスを取得
        
        Args:
            model_id: モデルID
            
        Returns:
            モデルパス（存在しない場合None）
        """
        if model_id not in self.MODELS:
            logger.warning(f"不明なモデルID: {model_id}")
            return None
        
        model_path = self.cache_dir / self.MODELS[model_id]['filename']
        
        if not model_path.exists():
            return None
        
        return model_path
    
    def download_model(
        self, 
        model_id: str,
        force: bool = False,
    ) -> Tuple[bool, str]:
        """
        モデルをダウンロード
        
        Args:
            model_id: モデルID
            force: 強制再ダウンロード
            
        Returns:
            (成功フラグ, メッセージ)
        """
        if model_id not in self.MODELS:
            return False, f"不明なモデルID: {model_id}"
        
        info = self.MODELS[model_id]
        model_path = self.cache_dir / info['filename']
        
        # 既存チェック
        if model_path.exists() and not force:
            return True, f"{info['name']}は既にダウンロード済みです"
        
        logger.info(f"{info['name']}をダウンロード中...")
        
        try:
            # HuggingFaceモデルの場合
            if 'huggingface.co' in info['url']:
                return self._download_huggingface(model_id, info, model_path)
            
            # 直接ダウンロードの場合
            else:
                return self._download_direct(model_id, info, model_path)
        
        except Exception as e:
            logger.error(f"ダウンロード失敗: {e}")
            return False, f"ダウンロードエラー: {str(e)}"
    
    def _download_huggingface(
        self, 
        model_id: str, 
        info: Dict[str, Any],
        model_path: Path
    ) -> Tuple[bool, str]:
        """HuggingFaceモデルのダウンロード"""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            # モデルIDを抽出（URLから）
            hf_model_id = info['url'].split('huggingface.co/')[-1]
            
            logger.info(f"HuggingFaceからダウンロード: {hf_model_id}")
            
            # プロキシ設定を適用してダウンロード
            proxies = {}
            if self.proxy_http:
                proxies['http'] = self.proxy_http
            if self.proxy_https:
                proxies['https'] = self.proxy_https
            
            # ダウンロード
            model = AutoModel.from_pretrained(
                hf_model_id,
                cache_dir=str(model_path),
                proxies=proxies if proxies else None,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                hf_model_id,
                cache_dir=str(model_path),
                proxies=proxies if proxies else None,
            )
            
            logger.info(f"ダウンロード完了: {model_path}")
            return True, f"{info['name']}のダウンロードが完了しました"
        
        except ImportError:
            msg = "transformersライブラリが必要です: pip install transformers"
            logger.error(msg)
            return False, msg
        
        except Exception as e:
            logger.error(f"HuggingFaceダウンロード失敗: {e}")
            return False, f"ダウンロード失敗: {str(e)}"
    
    def _download_direct(
        self, 
        model_id: str, 
        info: Dict[str, Any],
        model_path: Path
    ) -> Tuple[bool, str]:
        """直接ダウンロード（requests使用）"""
        try:
            import requests
            from tqdm import tqdm
            
            # プロキシ設定
            proxies = {}
            if self.proxy_http:
                proxies['http'] = self.proxy_http
            if self.proxy_https:
                proxies['https'] = self.proxy_https
            
            # ダウンロード
            logger.info(f"ダウンロード開始: {info['url']}")
            response = requests.get(
                info['url'],
                stream=True,
                proxies=proxies if proxies else None,
                timeout=30,
            )
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            # 一時ファイルにダウンロード
            temp_path = model_path.with_suffix('.tmp')
            
            with open(temp_path, 'wb') as f:
                if total_size:
                    progress = tqdm(
                        total=total_size, 
                        unit='B', 
                        unit_scale=True,
                        desc=info['name']
                    )
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        progress.update(len(chunk))
                    progress.close()
                else:
                    f.write(response.content)
            
            # 成功したら正式な名前にリネーム
            temp_path.rename(model_path)
            
            logger.info(f"ダウンロード完了: {model_path}")
            return True, f"{info['name']}のダウンロードが完了しました"
        
        except requests.exceptions.ProxyError:
            msg = "プロキシ接続エラー。プロキシ設定を確認してください。"
            logger.error(msg)
            return False, msg
        
        except requests.exceptions.ConnectionError:
            msg = "ネットワーク接続エラー。手動ダウンロードを検討してください。"
            logger.error(msg)
            return False, msg
        
        except Exception as e:
            logger.error(f"直接ダウンロード失敗: {e}")
            return False, f"ダウンロード失敗: {str(e)}"
    
    def delete_model(self, model_id: str) -> Tuple[bool, str]:
        """
        モデルを削除
        
        Args:
            model_id: モデルID
            
        Returns:
            (成功フラグ, メッセージ)
        """
        if model_id not in self.MODELS:
            return False, f"不明なモデルID: {model_id}"
        
        info = self.MODELS[model_id]
        model_path = self.cache_dir / info['filename']
        
        if not model_path.exists():
            return False, f"{info['name']}は存在しません"
        
        try:
            if model_path.is_dir():
                shutil.rmtree(model_path)
            else:
                model_path.unlink()
            
            logger.info(f"モデル削除: {model_path}")
            return True, f"{info['name']}を削除しました"
        
        except Exception as e:
            logger.error(f"モデル削除失敗: {e}")
            return False, f"削除エラー: {str(e)}"
    
    def get_manual_instructions(self, model_id: str) -> Optional[Dict[str, str]]:
        """
        手動ダウンロード手順を取得
        
        Args:
            model_id: モデルID
            
        Returns:
            手順情報（URL、配置先パス等）
        """
        if model_id not in self.MODELS:
            return None
        
        info = self.MODELS[model_id]
        model_path = self.cache_dir / info['filename']
        
        return {
            'model_name': info['name'],
            'download_url': info['manual_url'],
            'target_path': str(model_path),
            'size_mb': info['size_mb'],
            'instructions': f"""
手動ダウンロード手順:

1. ブラウザで以下のURLにアクセス:
   {info['manual_url']}

2. ファイルをダウンロード（{info['size_mb']} MB程度）

3. ダウンロードしたファイルを以下に配置:
   {model_path}

4. アプリを再起動して認識を確認

注意: ディレクトリ型の場合は、フォルダ全体を配置してください。
""".strip()
        }
    
    def get_cache_info(self) -> Dict[str, Any]:
        """キャッシュディレクトリ情報を取得"""
        total_size = 0
        file_count = 0
        
        if self.cache_dir.exists():
            for item in self.cache_dir.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
                    file_count += 1
        
        return {
            'cache_dir': str(self.cache_dir),
            'total_size_mb': total_size / (1024 * 1024),
            'file_count': file_count,
            'exists': self.cache_dir.exists(),
        }
