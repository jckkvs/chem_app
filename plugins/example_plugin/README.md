# Example Plugin

予測結果のログ出力と範囲制限を行うサンプルプラグインです。

## 機能

- ✅ 予測値をログに出力
- ✅ 予測値を設定範囲内に制限（デフォルト: 0-100）
- ✅ 学習完了時にメトリクスを表示

## 使い方

プラグインは自動的に読み込まれます。

### 設定のカスタマイズ

```python
from core.services.plugin import plugin_manager

plugin = plugin_manager.plugins['example_plugin']
plugin.config['min_value'] = -10.0
plugin.config['max_value'] = 200.0
plugin.config['log_predictions'] = False
```

## 開発者向け

このプラグインをベースに、独自のプラグインを作成できます。
`plugin.py`をコピーして修正してください。

詳細は [../../docs/PLUGIN_DEVELOPMENT.md](../../docs/PLUGIN_DEVELOPMENT.md) を参照。
