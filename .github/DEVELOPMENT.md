# Development Workflow Guide

このドキュメントは開発者向けの内部ガイドです。

## リリースワークフロー

### 1. 機能開発
- Feature branchで開発を進める
- コミットメッセージは[Conventional Commits](https://www.conventionalcommits.org/)に従う
  - `feat:` 新機能
  - `fix:` バグ修正
  - `docs:` ドキュメント変更
  - `test:` テスト追加・修正
  - `refactor:` リファクタリング
  - `chore:` その他

### 2. マージ前のチェックリスト
- [ ] すべてのテストがパス (`pytest core/tests`)
- [ ] コードフォーマット適用 (`black .`)
- [ ] 型チェック (`mypy core/`)
- [ ] Linting (`flake8 core/`)

### 3. バージョンアップ時の作業

新しいバージョンをリリースする際は、以下の手順で進めます：

#### 3.1 CHANGELOG.mdの更新
`CHANGELOG.md`に以下の形式で記載：
```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- 新機能の説明

### Changed
- 変更内容

### Fixed
- 修正したバグ

### Removed
- 削除した機能
```

#### 3.2 バージョン番号の更新
以下のファイルを更新：
- `setup.py` または `pyproject.toml`
- `__init__.py` (バージョン定数がある場合)

#### 3.3 Git タグの作成
```bash
git tag -a v0.4.0 -m "Version 0.4.0: Test coverage improvements"
git push origin v0.4.0
```

### 4. 自動化ツール

プロジェクトルートに配置されている以下のスクリプトを活用：

#### `scripts/prepare_release.py` (作成推奨)
バージョンアップ作業を自動化するPythonスクリプト。
- CHANGELOG.mdの差分チェック
- バージョン番号の一括更新
- Gitタグの作成

使用例:
```bash
python scripts/prepare_release.py --version 0.5.0
```

## Git Hooks (推奨設定)

`.git/hooks/commit-msg` に以下を配置すると、コミットメッセージの形式をチェック：
```bash
#!/bin/sh
commit_msg=$(cat "$1")
if ! echo "$commit_msg" | grep -qE "^(feat|fix|docs|test|refactor|chore):"; then
    echo "Error: Commit message must follow Conventional Commits format"
    echo "Example: feat: Add new feature"
    exit 1
fi
```

## コードレビューのポイント

- 新機能には必ずテストを追加
- ドキュメント文字列（docstring）を記載
- 型ヒントを使用
- パフォーマンスへの影響を考慮

## リリース前の最終確認

```bash
# 全テスト実行
pytest core/tests --cov=core --cov-report=html

# カバレッジ確認 (目標: >80%)
open htmlcov/index.html

# セキュリティチェック
bandit -r core/

# 依存関係の脆弱性チェック
safety check
```

## トラブルシューティング

### テストが失敗する場合
1. 環境変数を確認 (`.env.example`参照)
2. 依存関係を再インストール (`pip install -r requirements.txt`)
3. データベースをリセット (`python manage.py migrate --run-syncdb`)

### マージコンフリクト
```bash
git fetch origin
git rebase origin/main
# コンフリクトを解決
git rebase --continue
```

---

**Note**: このドキュメントは内部用です。外部公開するドキュメントは`README.md`や`CONTRIBUTING.md`を参照してください。
