# セキュリティ設定ガイド

## 🔒 初回セットアップ（必須）

Chemical ML Platformを安全に使用するために、以下のセキュリティ設定を実施してください。

### 1. 環境変数ファイルの作成

```bash
# .env.exampleをコピー
cp .env.example .env
```

### 2. SECRET_KEYの生成と設定

```bash
# SECRET_KEYを生成
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

生成されたキーを`.env`ファイルに設定:

```bash
# .env
DJANGO_SECRET_KEY=<生成されたキーをここに貼り付け>
```

### 3. API認証トークンの生成

```bash
# API_SECRET_TOKENを生成
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

生成されたトークンを`.env`ファイルに設定:

```bash
# .env
API_SECRET_TOKEN=<生成されたトークンをここに貼り付け>
```

### 4. 環境変数の確認

`.env`ファイルが正し く設定されているか確認:

```bash
cat .env
```

以下の項目が設定されていることを確認:
- `DJANGO_SECRET_KEY`
- `DJANGO_DEBUG=True` (開発環境の場合)
- `ALLOWED_HOSTS=localhost,127.0.0.1`
- `API_SECRET_TOKEN`

---

## 🚀 本番環境設定

本番環境にデプロイする場合は、追加の設定が必要です。

### 1. DEBUG モードを無効化

```bash
# .env (本番環境)
DJANGO_DEBUG=False
```

### 2. ALLOWED_HOSTS を設定

```bash
# .env (本番環境)
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
```

### 3. HTTPS を有効化

本番環境では、以下のセキュリティ設定が自動的に有効になります:
- HTTPS強制リダイレクト
- Secure Cookies
- HSTS (HTTP Strict Transport Security)
- XSS Protection

Nginx/Apache等のリバースプロキシでSSL証明書を設定してください。

---

## 🔑 API認証の使用方法

### 開発環境（認証なし）

開発環境では`API_SECRET_TOKEN`が未設定の場合、警告が表示されますが認証なしでアクセス可能です。

```bash
# 認証なしでアクセス可能（開発環境のみ）
curl http://localhost:8000/api/datasets
```

### 本番環境（認証必須）

本番環境では、すべてのAPIリクエストに`Authorization`ヘッダーが必要です。

```bash
# 認証ヘッダー付きでアクセス
export API_TOKEN="your-api-token-here"
curl -H "Authorization: Bearer $API_TOKEN" https://yourdomain.com/api/datasets
```

### 公開エンドポイント（認証不要）

ヘルスチェック等の公開エンドポイントは認証不要です:

```bash
# 認証なしでアクセス可能
curl http://localhost:8000/api/public/health
curl http://localhost:8000/api/public/health/rdkit
```

---

## ⚠️ セキュリティ警告

### 絶対にやってはいけないこと

1. ❌ `.env`ファイルをGitにコミットしない（`.gitignore`に追加済み）
2. ❌ SECRET_KEYをコードにハードコードしない
3. ❌ 本番環境で`DEBUG=True`にしない
4. ❌ 本番環境で`ALLOWED_HOSTS=['*']`にしない
5. ❌ API_SECRET_TOKENを公開リポジトリに含めない

### 推奨事項

1. ✅ SECRET_KEYは定期的に変更する
2. ✅ API_SECRET_TOKENも定期的にローテーションする
3. ✅ 本番環境ではHTTPSを必ず使用する
4. ✅ 環境変数は環境変数管理サービス（AWS Secrets Manager等）を使用
5. ✅ アクセスログを定期的に確認する

---

## 🧪 セキュリティテストの実行

設定が正しく機能しているか確認:

```bash
# セキュリティテストを実行
pytest core/tests/test_security.py -v
```

すべてのテストがパスすることを確認してください。

---

## 🆘 トラブルシューティング

### エラー: "DJANGO_SECRET_KEY environment variable must be set"

**原因**: `.env`ファイルが読み込まれていない、またはSECRET_KEYが設定されていない

**解決策**:
1. `.env`ファイルが存在するか確認
2. `DJANGO_SECRET_KEY`が設定されているか確認
3. Djangoが`.env`を読み込むライブラリ（python-decouple等）をインストール

### エラー: API認証が失敗する

**原因**: トークンが正しく設定されていない

**解決策**:
1. `.env`に`API_SECRET_TOKEN`が設定されているか確認
2. リクエストの`Authorization`ヘッダーが正しいか確認
3. トークンの前後に余分なスペースがないか確認

---

## 📚 参考資料

- [Django Security Best Practices](https://docs.djangoproject.com/en/stable/topics/security/)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Django Ninja Authentication](https://django-ninja.rest-framework.com/guides/authentication/)
