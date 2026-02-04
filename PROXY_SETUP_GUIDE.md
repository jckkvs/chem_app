# プロキシ設定ガイド - 対象者別

このガイドは、**あなたの役割**に応じて必要な設定が異なります。

---

## 👤 あなたはどちらですか？

### 🛠️ **開発者 / アプリ実装者**

以下に該当する方：
- Chemical ML Platformのソースコードを入手してカスタマイズする
- 機械学習モデルの重み（Uni-Mol、ChemBERTa等）をダウンロードする
- 新しい機能を追加・開発する

👉 **[開発者向けガイド](#開発者向けガイド)** をご覧ください

---

### 👥 **エンドユーザー / アプリ利用者**

以下に該当する方：
- すでにセットアップ済みのアプリケーションを使用する
- Webブラウザ経由でアプリを利用する
- 機械学習は詳しくないが、分子物性予測を行いたい

👉 **[エンドユーザー向けガイド](#エンドユーザー向けガイド)** をご覧ください

---

## 🛠️ 開発者向けガイド

### 必要な設定

| 設定項目 | 必要度 | 理由 |
|---------|--------|------|
| **プロキシ（HTTP/HTTPS）** | ✅ 必須 | GitHub、HuggingFaceからの重みダウンロード |
| **SSL証明書** | △ 企業環境 | Zscaler等のSSL検査対応 |
| **.condarc** | △ Conda使用時 | `conda install` でのパッケージ取得 |
| **pip.ini** | ✅ ほぼ必須 | `pip install` でのパッケージ取得 |
| **環境変数** | ✅ 必須 | HuggingFace transformersのダウンロード |

### セットアップ手順

#### Step 1: プロキシ設定

```bash
# Windows (PowerShell)
$env:HTTP_PROXY="http://proxy.company.com:8080"
$env:HTTPS_PROXY="http://proxy.company.com:8080"

# Linux/Mac
export HTTP_PROXY="http://proxy.company.com:8080"
export HTTPS_PROXY="http://proxy.company.com:8080"
```

または、アプリの設定画面で：
1. `/proxy-settings` にアクセス
2. 「🌐 プロキシ設定」タブでプロキシURL入力
3. 「💾 保存」→「📥 スクリプトをエクスポート」
4. 生成されたスクリプトを実行

#### Step 2: pip設定（pip.ini / pip.conf）

**自動設定（推奨）**:
- アプリの「🛠️ ツール設定」→「pip設定」→「💾 pip設定を更新」

**手動設定**:
```ini
# Windows: C:\Users\YourName\pip\pip.ini
# Linux/Mac: ~/.pip/pip.conf

[global]
proxy = http://proxy.company.com:8080
```

#### Step 3: Conda設定（.condarc）- Conda使用者のみ

**自動設定（推奨）**:
- アプリの「🛠️ ツール設定」→「Conda設定」→「💾 .condarc を更新」

**手動設定**:
```yaml
# C:\Users\YourName\.condarc または ~/.condarc

proxy_servers:
  http: http://proxy.company.com:8080
  https: http://proxy.company.com:8080
ssl_verify: true
```

#### Step 4: SSL証明書（Zscaler等）

```bash
# SSL証明書パスを環境変数に設定
export SSL_CERT_FILE="/path/to/zscaler_cert.pem"
export REQUESTS_CA_BUNDLE="/path/to/zscaler_cert.pem"
```

または、アプリの「🔒 SSL証明書」タブで設定

#### Step 5: モデル重みダウンロード

以下のモデルは**開発者が手動でダウンロード**する必要があります：

| モデル | ダウンロード元 | サイズ | 必須？ |
|--------|--------------|--------|--------|
| **Uni-Mol** | [GitHub](https://github.com/dptech-corp/Uni-Mol/releases) | 850MB | △ |
| **ChemBERTa** | [HuggingFace](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1) | 450MB | △ |
| **TARTE** | [HuggingFace](https://huggingface.co/mizuno-group/tarte-base) | 250MB | △ |
| **TabPFN** | [HuggingFace](https://huggingface.co/TabPFN/TabPFN) | 150MB | △ |
| **GROVER** | [GitHub](https://github.com/tencent-ailab/grover/releases) | 320MB | △ |

**ダウンロード方法**:
1. アプリの「📦 モデル管理」タブで「ダウンロード」ボタン
2. または、「手動DL」ボタンで手順表示
3. ブラウザでダウンロード → `~/.chemml/models/` に配置

---

## 👥 エンドユーザー向けガイド

### 基本的に設定不要！

すでにセットアップ済みのアプリケーションを使用する場合、**プロキシ設定や重みダウンロードは不要**です。

開発者が既に以下を準備済み：
- ✅ 機械学習モデルの重み
- ✅ 必要なパッケージ
- ✅ プロキシ設定（サーバー側）

### 例外: 追加パッケージが必要な場合

アプリ管理者から「追加パッケージをインストールしてください」と指示があった場合のみ：

**企業プロキシ経由の場合**:

```bash
# プロキシ付きでpip install
pip install package_name --proxy http://proxy.company.com:8080
```

または、pip.ini を設定（上記の開発者向けガイド参照）

---

## ❓ よくある質問

### Q: 私は開発者？ユーザー？

**開発者**:
- GitHubからソースコードをcloneした
- `git clone https://github.com/...` を実行した
- 機能を追加・カスタマイズしたい

**ユーザー**:
- ブラウザでアクセスするだけ（例: `http://localhost:8000`）
- 管理者からURLを共有された
- コードは触らない

---

### Q: エンドユーザーだが、モデルダウンロードを求められた

**原因**: 管理者がモデルをセットアップしていない可能性

**対処法**:
1. アプリ管理者に連絡
2. または、一時的に開発者向けガイドに従ってダウンロード

---

### Q: pip install がエラーになる（エンドユーザー）

```
ProxyError: HTTPSConnectionPool...
```

**対処法**:

```bash
# 一時的にプロキシ指定
pip install package_name --proxy http://proxy.company.com:8080

# または、pip.ini設定（開発者向けガイド参照）
```

---

## 📋 チェックリスト

### 開発者向け

- [ ] プロキシURL設定（HTTP/HTTPS）
- [ ] pip.ini を設定
- [ ] .condarc を設定（Conda使用時）
- [ ] SSL証明書を設定（Zscaler等）
- [ ] 環境変数スクリプトを実行
- [ ] 必要なモデル重みをダウンロード
- [ ] モデル重みを `~/.chemml/models/` に配置
- [ ] アプリ起動確認

### エンドユーザー向け

- [ ] ブラウザでアプリにアクセス
- [ ] （追加パッケージ必要時のみ）pip.ini設定

---

**最終更新**: 2026-02-05  
**対応バージョン**: Chemical ML Platform v1.0+
