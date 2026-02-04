"""
Streamlit プロキシ設定UI

Implements: F-STREAMLIT-PROXY-UI-001
設計思想:
- 企業環境（Zscaler等）の完全対応
- HTTP/HTTPSプロキシ設定
- SSL証明書設定
- Conda/pip設定の自動生成
- 接続テスト機能
"""

import streamlit as st
import requests
import os
from pathlib import Path
from typing import Optional, Dict

def render_proxy_settings():
    """
    プロキシ設定UIをレンダリング
    
    企業環境でモデルダウンロードが必要な開発者向け
    """
    st.title("🌐 ネットワーク・プロキシ設定")
    
    st.info("""
    **👤 この設定が必要な方:**
    - アプリ開発者・実装者
    - モデルの重みをダウンロードする方
    - 企業プロキシ環境（Zscaler等）の方
    
    **エンドユーザーの方**: この設定は不要です
    """)
    
    # タブ構成
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌐 プロキシ設定",
        "🔒 SSL証明書",
        "🛠️ ツール設定 (Conda/pip)",
        "📥 モデル管理"
    ])
    
    # === タブ1: プロキシ設定 ===
    with tab1:
        st.subheader("HTTP/HTTPS プロキシ設定")
        
        col1, col2 = st.columns(2)
        
        with col1:
            http_proxy = st.text_input(
                "HTTPプロキシ",
                value=os.environ.get('HTTP_PROXY', ''),
                help="例: http://proxy.company.com:8080",
                placeholder="http://proxy.company.com:8080"
            )
        
        with col2:
            https_proxy = st.text_input(
                "HTTPSプロキシ",
                value=os.environ.get('HTTPS_PROXY', ''),
                help="例: http://proxy.company.com:8080",
                placeholder="http://proxy.company.com:8080"
            )
        
        # No Proxy設定
        no_proxy = st.text_input(
            "No Proxy（除外リスト）",
            value=os.environ.get('NO_PROXY', 'localhost,127.0.0.1'),
            help="カンマ区切りで指定。例: localhost,127.0.0.1,.local"
        )
        
        # 保存ボタン
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 保存", type="primary", use_container_width=True):
                save_proxy_settings(http_proxy, https_proxy, no_proxy)
                st.success("✅ プロキシ設定を保存しました")
                st.rerun()
        
        with col2:
            if st.button("🔍 接続テスト", use_container_width=True):
                test_proxy_connection(http_proxy, https_proxy)
        
        with col3:
            if st.button("📄 スクリプトエクスポート", use_container_width=True):
                export_proxy_script(http_proxy, https_proxy, no_proxy)
        
        # 現在の設定表示
        st.divider()
        st.subheader("現在の環境変数")
        
        proxy_vars = {
            'HTTP_PROXY': os.environ.get('HTTP_PROXY', '未設定'),
            'HTTPS_PROXY': os.environ.get('HTTPS_PROXY', '未設定'),
            'NO_PROXY': os.environ.get('NO_PROXY', '未設定'),
        }
        
        for key, value in proxy_vars.items():
            st.text(f"{key}: {value}")
    
    # === タブ2: SSL証明書 ===
    with tab2:
        st.subheader("🔒 SSL証明書設定（Zscaler等）")
        
        st.info("""
        **企業環境でSSL検査がある場合に設定してください**
        
        Zscaler等のSSL検査プロキシを使用している場合、
        証明書パスを指定する必要があります。
        """)
        
        ssl_cert_path = st.text_input(
            "SSL証明書パス",
            value=os.environ.get('SSL_CERT_FILE', ''),
            help="例: C:\\Program Files\\Zscaler\\ZscalerRootCA.pem",
            placeholder="C:\\Program Files\\Zscaler\\ZscalerRootCA.pem"
        )
        
        # 証明書ファイルの存在確認
        if ssl_cert_path:
            if Path(ssl_cert_path).exists():
                st.success(f"✅ 証明書ファイルが見つかりました: {ssl_cert_path}")
            else:
                st.error(f"❌ 証明書ファイルが見つかりません: {ssl_cert_path}")
        
        # 保存ボタン
        if st.button("💾 SSL設定を保存", type="primary"):
            save_ssl_settings(ssl_cert_path)
            st.success("✅ SSL証明書設定を保存しました")
            st.rerun()
        
        # SSL証明書の取得方法
        with st.expander("📖 SSL証明書の取得方法（Zscaler）"):
            st.markdown("""
            ### Zscaler証明書の取得手順
            
            1. **ブラウザから取得（Chrome）**:
               - HTTPSサイトにアクセス
               - アドレスバーの🔒アイコンをクリック
               - 「証明書」→「詳細」→「エクスポート」
               - PEM形式で保存
            
            2. **Zscalerアプリから取得**:
               - Zscalerアプリを開く
               - 設定 → 証明書
               - 「証明書をエクスポート」
            
            3. **デフォルトパス（Windows）**:
               ```
               C:\\Program Files\\Zscaler\\ZscalerRootCA.pem
               C:\\ProgramData\\Zscaler\\cert\\ZscalerRootCA.cer
               ```
            """)
    
    # === タブ3: ツール設定 ===
    with tab3:
        st.subheader("🛠️ Conda / pip 設定")
        
        st.info("Conda や pip でパッケージをインストールする際のプロキシ設定を自動生成します")
        
        # Conda設定
        st.markdown("### Conda設定 (.condarc)")
        
        if st.button("📝 .condarc を生成・更新"):
            generate_condarc(http_proxy, https_proxy, ssl_cert_path)
            st.success("✅ .condarc を更新しました")
        
        # pip設定
        st.markdown("### pip設定 (pip.ini / pip.conf)")
        
        if st.button("📝 pip設定を生成・更新"):
            generate_pip_config(http_proxy, https_proxy)
            st.success("✅ pip設定を更新しました")
        
        # 設定ファイルの場所を表示
        with st.expander("📂 設定ファイルの場所"):
            st.markdown(f"""
            **Conda設定**:
            - Windows: `{Path.home() / '.condarc'}`
            - Linux/Mac: `~/.condarc`
            
            **pip設定**:
            - Windows: `{Path.home() / 'pip' / 'pip.ini'}`
            - Linux/Mac: `~/.pip/pip.conf`
            """)
    
    # === タブ4: モデル管理 ===
    with tab4:
        st.subheader("📥 モデル重みダウンロード")
        
        st.warning("""
        **⚠️ 開発者向け機能**
        
        以下のモデルは開発者が手動でダウンロードする必要があります。
        エンドユーザーは不要です。
        """)
        
        models = [
            {
                'name': 'Uni-Mol',
                'size': '850MB',
                'url': 'https://github.com/dptech-corp/Uni-Mol/releases',
                'path': '~/.chemml/models/unimol_v0.1.pt'
            },
            {
                'name': 'ChemBERTa',
                'size': '450MB',
                'url': 'https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1',
                'path': '~/.chemml/models/chemberta/'
            },
            {
                'name': 'TARTE',
                'size': '250MB',
                'url': 'https://huggingface.co/mizuno-group/tarte-base',
                'path': '~/.chemml/models/tarte/'
            },
            {
                'name': 'TabPFN',
                'size': '150MB',
                'url': 'https://huggingface.co/TabPFN/TabPFN',
                'path': '~/.chemml/models/tabpfn/'
            },
        ]
        
        for model in models:
            with st.expander(f"**{model['name']}** ({model['size']})"):
                st.markdown(f"""
                **ダウンロード元**: [{model['url']}]({model['url']})
                
                **配置先**: `{model['path']}`
                
                **手動ダウンロード手順**:
                1. 上記URLからブラウザでダウンロード
                2. ファイルを `{model['path']}` に配置
                3. アプリを再起動
                """)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button(f"🌐 {model['name']} URLを開く", key=f"open_{model['name']}"):
                        st.markdown(f"[{model['url']}]({model['url']})")
                
                with col2:
                    # 配置先フォルダを作成
                    if st.button(f"📁 配置先を作成", key=f"mkdir_{model['name']}"):
                        target_path = Path(model['path']).expanduser()
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        st.success(f"✅ フォルダを作成: {target_path.parent}")


def save_proxy_settings(http_proxy: str, https_proxy: str, no_proxy: str):
    """プロキシ設定を保存"""
    if http_proxy:
        os.environ['HTTP_PROXY'] = http_proxy
        os.environ['http_proxy'] = http_proxy
    
    if https_proxy:
        os.environ['HTTPS_PROXY'] = https_proxy
        os.environ['https_proxy'] = https_proxy
    
    if no_proxy:
        os.environ['NO_PROXY'] = no_proxy
        os.environ['no_proxy'] = no_proxy


def save_ssl_settings(ssl_cert_path: str):
    """SSL証明書設定を保存"""
    if ssl_cert_path:
        os.environ['SSL_CERT_FILE'] = ssl_cert_path
        os.environ['REQUESTS_CA_BUNDLE'] = ssl_cert_path
        os.environ['CURL_CA_BUNDLE'] = ssl_cert_path


def test_proxy_connection(http_proxy: str, https_proxy: str):
    """プロキシ接続テスト"""
    st.info("接続テスト中...")
    
    proxies = {}
    if http_proxy:
        proxies['http'] = http_proxy
    if https_proxy:
        proxies['https'] = https_proxy
    
    # テストURL
    test_urls = [
        ('HTTP', 'http://www.google.com'),
        ('HTTPS', 'https://www.google.com'),
        ('HuggingFace', 'https://huggingface.co'),
    ]
    
    results = []
    
    for name, url in test_urls:
        try:
            response = requests.get(url, proxies=proxies, timeout=5)
            if response.status_code == 200:
                results.append(f"✅ {name}: 成功 ({response.status_code})")
            else:
                results.append(f"⚠️ {name}: HTTP {response.status_code}")
        except Exception as e:
            results.append(f"❌ {name}: 失敗 - {str(e)[:50]}")
    
    # 結果表示
    for result in results:
        if "✅" in result:
            st.success(result)
        elif "⚠️" in result:
            st.warning(result)
        else:
            st.error(result)


def export_proxy_script(http_proxy: str, https_proxy: str, no_proxy: str):
    """プロキシ設定スクリプトをエクスポート"""
    
    # Windows PowerShell
    ps_script = f"""# プロキシ設定スクリプト (PowerShell)
$env:HTTP_PROXY="{http_proxy}"
$env:HTTPS_PROXY="{https_proxy}"
$env:NO_PROXY="{no_proxy}"

Write-Host "✅ プロキシ設定完了" -ForegroundColor Green
"""
    
    # Linux/Mac Bash
    bash_script = f"""# プロキシ設定スクリプト (Bash)
export HTTP_PROXY="{http_proxy}"
export HTTPS_PROXY="{https_proxy}"
export NO_PROXY="{no_proxy}"

echo "✅ プロキシ設定完了"
"""
    
    st.code(ps_script, language='powershell')
    st.download_button(
        "💾 PowerShellスクリプトをダウンロード",
        ps_script,
        file_name="set_proxy.ps1",
        mime="text/plain"
    )
    
    st.code(bash_script, language='bash')
    st.download_button(
        "💾 Bashスクリプトをダウンロード",
        bash_script,
        file_name="set_proxy.sh",
        mime="text/plain"
    )


def generate_condarc(http_proxy: str, https_proxy: str, ssl_cert_path: Optional[str] = None):
    """Conda設定ファイル (.condarc) を生成"""
    condarc_path = Path.home() / '.condarc'
    
    config = f"""# Conda プロキシ設定
proxy_servers:
  http: {http_proxy or ''}
  https: {https_proxy or ''}

ssl_verify: {'true' if not ssl_cert_path else ssl_cert_path}

channels:
  - defaults
  - conda-forge
"""
    
    condarc_path.write_text(config, encoding='utf-8')
    st.info(f"✅ .condarc を作成: {condarc_path}")


def generate_pip_config(http_proxy: str, https_proxy: str):
    """pip設定ファイルを生成"""
    import platform
    
    if platform.system() == 'Windows':
        pip_dir = Path.home() / 'pip'
        pip_config = pip_dir / 'pip.ini'
    else:
        pip_dir = Path.home() / '.pip'
        pip_config = pip_dir / 'pip.conf'
    
    pip_dir.mkdir(parents=True, exist_ok=True)
    
    config = f"""[global]
proxy = {http_proxy or https_proxy}
"""
    
    pip_config.write_text(config, encoding='utf-8')
    st.info(f"✅ pip設定を作成: {pip_config}")


if __name__ == "__main__":
    render_proxy_settings()
