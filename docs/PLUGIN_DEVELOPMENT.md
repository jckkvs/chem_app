# プラグイン開発ガイド

Chemical ML Platformのプラグインシステムを使って、外部から機能を追加する方法を説明します。

## 📌 プラグインとは？

プラグインは、コアシステムを変更せずに新機能を追加できる仕組みです。

**利点**:
- ✅ コアコードを変更せずに拡張可能
- ✅ 機能のオン/オフが簡単
- ✅ 独立した開発・テストが可能
- ✅ 複数のプラグインを組み合わせ可能

---

## 🚀 クイックスタート

### 最小限のプラグイン例

```python
# plugins/my_plugin/plugin.py

from core.services.plugin import Plugin

def create_plugin():
    """プラグインのエントリーポイント"""
    return Plugin(
        name="my_plugin",
        version="1.0.0",
        description="カスタム機能を追加するプラグイン",
        hooks={
            "on_prediction": my_prediction_hook,
            "on_training": my_training_hook,
        }
    )

def my_prediction_hook(smiles, prediction):
    """予測後に実行されるフック"""
    print(f"Predicted {prediction} for {smiles}")
    return prediction

def my_training_hook(experiment):
    """学習後に実行されるフック"""
    print(f"Training completed for {experiment.name}")
```

### プラグインの配置

```
plugins/
└── my_plugin/
    ├── __init__.py
    ├── plugin.py       # メインファイル
    ├── utils.py        # ヘルパー関数
    └── README.md       # プラグインの説明
```

---

## 🔌 利用可能なフック

プラグインは以下のフックポイントで処理を追加できます。

### 1. `on_prediction` - 予測時

**タイミング**: モデルが予測を実行した直後

```python
def on_prediction(smiles: str, prediction: float) -> float:
    """
    予測結果を加工・検証
    
    Args:
        smiles: 入力SMILES
        prediction: 予測値
        
    Returns:
        加工後の予測値
    """
    # 例: 予測値の範囲制限
    return max(0, min(prediction, 100))
```

### 2. `on_training` - 学習完了時

**タイミング**: モデル学習が完了した直後

```python
def on_training(experiment: Experiment) -> None:
    """
    学習完了後の処理
    
    Args:
        experiment: 実験オブジェクト
    """
    # 例: Slackに通知
    send_slack_notification(f"Training completed: {experiment.name}")
```

### 3. `on_feature_extraction` - 特徴量抽出時

**タイミング**: 特徴量抽出の直前

```python
def on_feature_extraction(smiles_list: List[str]) -> List[str]:
    """
    SMILES前処理
    
    Args:
        smiles_list: SMILESリスト
        
    Returns:
        前処理済みSMILESリスト
    """
    # 例: SMILESの正規化
    return [standardize_smiles(s) for s in smiles_list]
```

### 4. `on_error` - エラー発生時

**タイミング**: 例外が発生した時

```python
def on_error(error: Exception, context: dict) -> None:
    """
    エラー処理
    
    Args:
        error: 発生した例外
        context: エラー発生時のコンテキスト情報
    """
    # 例: Sentryに送信
    sentry_sdk.capture_exception(error)
```

---

## 📦 プラグインの実装例

### 例1: カスタム分子検証プラグイン

```python
# plugins/mol_validator/plugin.py

from rdkit import Chem
from core.services.plugin import Plugin

def create_plugin():
    return Plugin(
        name="mol_validator",
        version="1.0.0",
        description="分子構造の妥当性を検証",
        hooks={
            "on_feature_extraction": validate_molecules,
        }
    )

def validate_molecules(smiles_list):
    """無効なSMILESを除外"""
    valid_smiles = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # 追加の検証ルール
            if mol.GetNumAtoms() > 0 and mol.GetNumAtoms() < 200:
                valid_smiles.append(smiles)
        else:
            print(f"Invalid SMILES: {smiles}")
    
    return valid_smiles
```

### 例2: 予測値の後処理プラグイン

```python
# plugins/prediction_postprocessor/plugin.py

import numpy as np
from core.services.plugin import Plugin

def create_plugin():
    return Plugin(
        name="prediction_postprocessor",
        version="1.0.0",
        description="予測値にドメイン知識を適用",
        hooks={
            "on_prediction": apply_domain_knowledge,
        }
    )

def apply_domain_knowledge(smiles, prediction):
    """
    化学的妥当性に基づく予測値補正
    """
    from rdkit import Chem, Descriptors
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return prediction
    
    # 例: 分子量が大きいと溶解度が低い傾向
    mw = Descriptors.MolWt(mol)
    if mw > 500:
        prediction *= 0.8  # 20%減少
    
    # 例: 負の予測値は物理的に無意味
    prediction = max(0, prediction)
    
    return prediction
```

### 例3: 実験結果通知プラグイン

```python
# plugins/slack_notifier/plugin.py

import requests
from core.services.plugin import Plugin

SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

def create_plugin():
    return Plugin(
        name="slack_notifier",
        version="1.0.0",
        description="実験完了時にSlack通知",
        hooks={
            "on_training": notify_training_complete,
        }
    )

def notify_training_complete(experiment):
    """Slackに通知"""
    message = {
        "text": f"🎉 Training Completed!",
        "attachments": [{
            "color": "good",
            "fields": [
                {"title": "Experiment", "value": experiment.name, "short": True},
                {"title": "Model", "value": experiment.model_type, "short": True},
                {"title": "R²", "value": f"{experiment.result.metrics.get('r2', 0):.3f}", "short": True},
            ]
        }]
    }
    
    requests.post(SLACK_WEBHOOK_URL, json=message)
```

---

## 🔧 プラグインの登録

### 自動登録（推奨）

`plugins/`ディレクトリにプラグインを配置すると、自動的に読み込まれます。

```python
# core/services/plugin.py（拡張後）
# 自動検出機能により、plugins/以下が自動読み込み
```

### 手動登録

```python
from core.services.plugin import plugin_manager
from plugins.my_plugin.plugin import create_plugin

# プラグイン登録
my_plugin = create_plugin()
plugin_manager.register(my_plugin)

# プラグイン無効化
plugin_manager.disable("my_plugin")

# プラグイン再有効化
plugin_manager.enable("my_plugin")
```

---

## 🧪 プラグインのテスト

### テンプレート

```python
# plugins/my_plugin/test_plugin.py

import pytest
from .plugin import create_plugin, my_prediction_hook

def test_plugin_creation():
    """プラグインが正しく作成されるか"""
    plugin = create_plugin()
    
    assert plugin.name == "my_plugin"
    assert plugin.version == "1.0.0"
    assert "on_prediction" in plugin.hooks

def test_prediction_hook():
    """予測フックが正しく動作するか"""
    result = my_prediction_hook("CCO", 42.0)
    
    assert isinstance(result, float)
    assert result >= 0  # 負の値にならないか確認
```

### テスト実行

```bash
pytest plugins/my_plugin/test_plugin.py -v
```

---

## 📊 プラグインのメタデータ

プラグインに追加情報を含めることができます。

```python
def create_plugin():
    return Plugin(
        name="advanced_plugin",
        version="2.1.0",
        description="高度な機能を提供",
        hooks={...},
        # メタデータ（オプション）
        author="Your Name",
        license="MIT",
        requires=["rdkit>=2023.09", "numpy>=1.24"],
        config={
            "threshold": 0.8,
            "debug_mode": False,
        }
    )
```

---

## 🚨 ベストプラクティス

### 1. エラーハンドリング

プラグインは例外を適切に処理すべきです。

```python
def my_hook(data):
    try:
        # 処理
        return process(data)
    except Exception as e:
        logger.error(f"Plugin error: {e}")
        return data  # フォールバック
```

### 2. 設定ファイル

プラグイン固有の設定は外部ファイルで管理。

```python
# plugins/my_plugin/config.yaml
threshold: 0.8
api_key: "YOUR_API_KEY"
```

```python
import yaml

with open("plugins/my_plugin/config.yaml") as f:
    config = yaml.safe_load(f)
```

### 3. ドキュメント

各プラグインにREADMEを含める。

```markdown
# My Plugin

## 概要
このプラグインは...

## インストール
```bash
pip install -r requirements.txt
```

## 設定
...
```

---

## 🔍 デバッグ

### プラグインリスト確認

```python
from core.services.plugin import plugin_manager

# 登録済みプラグイン一覧
plugins = plugin_manager.list_plugins()
for p in plugins:
    print(f"{p['name']} v{p['version']} - {p['enabled']}")
```

### ログ出力

```python
import logging

logger = logging.getLogger(__name__)

def my_hook(data):
    logger.info(f"Processing: {data}")
    # 処理
```

---

## 📚 参考資料

- [ARCHITECTURE.md](../ARCHITECTURE.md) - システムアーキテクチャ
- [CONTRIBUTING.md](../CONTRIBUTING.md) - 貢献ガイド
- [core/services/plugin.py](../core/services/plugin.py) - プラグインマネージャー実装

---

## ❓ FAQ

**Q: プラグインは複数登録できますか？**  
A: はい、複数のプラグインを同時に登録できます。

**Q: プラグインの実行順序は？**  
A: 登録順に実行されます。順序に依存する場合は、プラグイン内で依存関係を明示してください。

**Q: プラグインを配布できますか？**  
A: はい、PyPIパッケージとして配布することも可能です。

---

Happy Plugin Development! 🎉
