# Examples

Chemical ML Platformの使い方を学ぶためのサンプルスクリプト集です。

## 📁 サンプル一覧

### 01_simple_descriptors.py
最も簡単な例。SMILESから分子記述子を計算します。

```bash
python examples/01_simple_descriptors.py
```

**学べること**:
- RDKitFeatureExtractorの基本的な使い方
- 特徴量の形状と記述子名の確認

---

### 02_basic_ml.py
基本的な機械学習パイプライン。溶解度を予測します。

```bash
python examples/02_basic_ml.py
```

**学べること**:
- MLPipelineの使い方
- モデル学習と予測
- モデル評価

---

### 03_api_usage.py
REST APIの使用例。プログラムからAPIを呼び出します。

**前提**: Djangoサーバーが起動していること
```bash
# 別ターミナルで
python manage.py runserver

# サンプル実行
python examples/03_api_usage.py
```

**学べること**:
- ヘルスチェック
- 分子物性取得
- SMILES検証
- バッチ予測

---

### 04_visualization.py
可視化の例。SHAP説明と化学空間マップを生成します。

```bash
python examples/04_visualization.py
```

**学べること**:
- SHAP説明の生成
- 化学空間マップの生成
- 結果の保存

---

## 📚 さらに詳しく

- **完全リファレンス**: [REFERENCE.md](../REFERENCE.md) - 全API、全メソッド、全引数を網羅
- **開発者向けガイド**: [CONTRIBUTING.md](../CONTRIBUTING.md)
- **アーキテクチャ**: [ARCHITECTURE.md](../ARCHITECTURE.md)

---

## 🚀 次のステップ

1. これらのサンプルを実行
2. REFERENCE.mdで詳細なAPIを確認
3. 独自のデータで試す
4. 新機能を追加（CONTRIBUTING.mdを参照）

Happy Coding! 🎉
