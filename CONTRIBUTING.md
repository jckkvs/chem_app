# 貢献ガイド（CONTRIBUTING）

Chemical ML Platformへの貢献にご興味をお持ちいただき、ありがとうございます！
このドキュメントでは、プロジェクトへの貢献方法を説明します。

## 🚀 クイックスタート

### 開発環境のセットアップ

```bash
# リポジトリをクローン
git clone https://github.com/jckkvs/chem_app.git
cd chem_app

# 仮想環境を作成
python -m venv .venv
.venv\Scripts\activate  # Windows

# 依存関係をインストール
pip install -r requirements.txt

# Django マイグレーション
python manage.py migrate

# 開発サーバー起動
python manage.py runserver
```

### テスト実行

```bash
# 全テスト実行
pytest core/tests/ -v

# カバレッジ付き
pytest core/tests/ -v --cov=core --cov-report=html
```

---

## 📝 コーディング規約

### Python スタイルガイド

- **PEP 8** に準拠
- **型ヒント** を使用（Python 3.9+）
- **Docstring** は Google スタイル

```python
def calculate_descriptors(smiles: str, descriptor_type: str = 'rdkit') -> pd.DataFrame:
    """
    SMILESから分子記述子を計算
    
    Args:
        smiles: SMILES文字列
        descriptor_type: 記述子タイプ（'rdkit', 'mordred', 'xtb'）
        
    Returns:
        pd.DataFrame: 計算された記述子
        
    Raises:
        ValueError: 無効なSMILES文字列の場合
    """
    pass
```

### 命名規則

- **クラス名**: `PascalCase` （例: `RDKitFeatureExtractor`）
- **関数名**: `snake_case` （例: `calculate_descriptors`）
- **定数**: `UPPER_SNAKE_CASE` （例: `MAX_MOLECULES`）
- **プライベート**: `_leading_underscore` （例: `_validate_smiles`）

### ファイル構成

```
core/services/
├── features/       # 特徴量抽出
│   ├── base.py     # 基底クラス
│   └── rdkit_eng.py
├── ml/             # 機械学習
│   ├── base.py     # 基底クラス
│   └── pipeline.py
└── vis/            # 可視化
    ├── base.py     # 基底クラス
    └── plots.py
```

---

## 🔧 新機能の追加方法

### 1. 特徴量抽出器を追加する場合

`core/services/features/` に新しいファイルを作成：

```python
# my_extractor.py
from .base import BaseFeatureExtractor
import pandas as pd
from typing import List

class MyFeatureExtractor(BaseFeatureExtractor):
    """
    カスタム特徴量抽出器
    
    Implements: F-MY-001
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初期化処理
    
    def transform(self, smiles_list: List[str]) -> pd.DataFrame:
        """SMILESを特徴量に変換"""
        # 実装
        pass
    
    @property
    def descriptor_names(self) -> List[str]:
        """記述子名のリスト"""
        return ['feature1', 'feature2', 'feature3']
```

テストを追加：

```python
# core/tests/test_my_extractor.py
import pytest
from core.services.features.my_extractor import MyFeatureExtractor

def test_transform():
    extractor = MyFeatureExtractor()
    smiles = ['CCO', 'c1ccccc1']
    result = extractor.transform(smiles)
    
    assert result.shape[0] == 2
    assert 'feature1' in result.columns
```

### 2. MLモデルを追加する場合

`core/services/ml/base.py` のベースクラスを継承：

```python
from .base import BaseMLModel

class MyModel(BaseMLModel):
    def fit(self, X, y):
        # 学習処理
        pass
    
    def predict(self, X):
        # 予測処理
        pass
```

---

## 🧪 テストガイドライン

### テストカバレッジ目標

- **行カバレッジ**: ≥85%
- **分岐カバレッジ**: ≥75%

### テストの種類

1. **ユニットテスト**: 個別関数・クラスのテスト
2. **統合テスト**: 複数モジュールの連携テスト
3. **APIテスト**: エンドポイントのテスト

### テスト命名規則

```python
def test_<機能>_<条件>_<期待結果>():
    pass

# 例
def test_rdkit_extractor_invalid_smiles_raises_error():
    pass
```

---

## 📦 プルリクエストの流れ

### 1. ブランチ作成

```bash
git checkout -b feature/my-new-feature
```

### 2. 変更をコミット

```bash
git add .
git commit -m "feat: Add new feature extractor for X"
```

#### コミットメッセージ規約

- `feat:` 新機能
- `fix:` バグ修正
- `docs:` ドキュメント
- `test:` テスト追加
- `refactor:` リファクタリング
- `style:` コードスタイル修正

### 3. プッシュ

```bash
git push origin feature/my-new-feature
```

### 4. プルリクエスト作成

GitHubでPRを作成し、以下を記載：

- **概要**: 何を変更したか
- **動機**: なぜ変更が必要か
- **テスト**: どのようにテストしたか
- **スクリーンショット**: UI変更の場合

---

## 🔍 コードレビュー基準

レビュワーは以下を確認します：

- [ ] コーディング規約に準拠しているか
- [ ] 型ヒントが適切に付与されているか
- [ ] テストが追加されているか
- [ ] テストが全てパスするか
- [ ] ドキュメントが更新されているか
- [ ] 既存機能に影響がないか

---

## 🐛 バグ報告

GitHubのIssueで以下を含めて報告：

1. **環境情報**: OS、Pythonバージョン、RDKitバージョン
2. **再現手順**: 最小限のコード例
3. **期待される動作**: 正しい動作
4. **実際の動作**: エラーメッセージ、トレースバック

---

## 💡 機能リクエスト

新機能の提案は歓迎です！Issueで以下を含めて提案：

1. **ユースケース**: どんな場面で必要か
2. **提案する実装**: 可能であれば、実装アイデア
3. **代替案**: 他の解決方法

---

## 📚 参考資料

- [アーキテクチャドキュメント](ARCHITECTURE.md)
- [プラグイン開発ガイド](docs/PLUGIN_DEVELOPMENT.md)
- [API設計ガイドライン](docs/API_GUIDELINES.md)

---

## ❓ 質問

質問がある場合は、以下の方法でお気軽に：

- GitHub Discussions
- Issue（質問タグ付き）

貢献をお待ちしています！🎉
