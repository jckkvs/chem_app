# Plugins Directory

このディレクトリはChemical ML Platformのプラグインを配置する場所です。

## 📁 プラグインの配置方法

プラグインは以下のディレクトリ構造で配置してください：

```
plugins/
└── my_plugin/
    ├── __init__.py
    ├── plugin.py       # create_plugin()関数を定義
    ├── utils.py        # （オプション）ヘルパー関数
    ├── requirements.txt # （オプション）依存パッケージ
    └── README.md       # （オプション）プラグインの説明
```

## 🚀 クイックスタート

1. 新しいプラグインディレクトリを作成：
```bash
mkdir plugins/my_plugin
```

2. `plugin.py`を作成し、`create_plugin()`関数を定義：
```python
# plugins/my_plugin/plugin.py
from core.services.plugin import Plugin

def create_plugin():
    return Plugin(
        name="my_plugin",
        version="1.0.0",
        description="My custom plugin",
        hooks={
            "on_prediction": my_hook_function,
        }
    )

def my_hook_function(smiles, prediction):
    print(f"Plugin called: {smiles} -> {prediction}")
    return prediction
```

3. プラグインは自動的に検出・ロードされます！

## 📖 詳細ガイド

プラグイン開発の詳細は [../docs/PLUGIN_DEVELOPMENT.md](../docs/PLUGIN_DEVELOPMENT.md) を参照してください。

## 📦 サンプルプラグイン

`example_plugin/` ディレクトリに実装例があります。
