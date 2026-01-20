# 再現プロンプト (REPRODUCE_PROMPT)

このドキュメントは、Smart Feature Engineeringシステムを再現・拡張するためのプロンプト集です。

---

## 1. システム全体を再構築する場合

```
chem_ml_appの特徴量エンジニアリングモジュールを以下の設計で実装してください：

### 設計思想
- 物性別の記述子プリセット（19カテゴリ）
- データセット分子構造の自動分析
- 事前学習済みモデル統合（Uni-Mol, ChemBERTa, TARTE）
- 自己教師あり学習モデル（GROVER, MolCLR, GraphMVP）
- 等変GNN（SchNet, PaiNN）による3D構造考慮
- 高度な特徴量選択（Boruta, mRMR, Permutation Importance）

### 必須機能
1. SmartFeatureEngine: 統合エンジン
2. DatasetAnalyzer: 分子構造分析
3. 物性プリセット: 19カテゴリ（光学/機械/熱/電気/化学/薬理/汎用）
4. Applicability Domain分析
5. 骨格分析と骨格ベースデータ分割
6. 類似度検索と活性クリフ検出

### オプショナル依存
すべての深層学習モデルはオプショナルにし、
インストールされていない場合はフォールバック処理を行う。
```

---

## 2. 物性プリセットを追加する場合

```
以下の物性に対応する記述子プリセットを追加してください：

物性名: [例: 音速]
カテゴリ: [例: acoustic]

含めるべき情報:
- 物性に相関の高いRDKit記述子リスト
- 選定根拠（なぜその記述子が有効か）
- 文献参照
- 推奨事前学習モデル

参考: 既存のdescriptor_presets.pyのパターンに従う
```

---

## 3. 新しい事前学習モデルを統合する場合

```
以下の事前学習済み分子モデルを統合してください：

モデル名: [例: PolyBERT]
用途: [例: ポリマー物性予測]
インストール: [例: pip install polybert]

実装要件:
1. オプショナル依存として扱う（遅延インポート）
2. pretrained_embeddings.pyまたはssl_embeddings.pyに追加
3. 統一インターフェース（get_embeddings(smiles_list) → np.ndarray）
4. is_available()メソッドで利用可能性チェック
5. エラー時はゼロベクトルを返すフォールバック
```

---

## 4. 特徴量選択アルゴリズムを追加する場合

```
以下の特徴量選択アルゴリズムを実装してください：

アルゴリズム名: [例: SHAP重要度ベース選択]

実装要件:
1. advanced_selectors.pyに追加
2. fit(X, y) → self
3. transform(X) → X_selected
4. selected_features_属性
5. feature_scores_属性（オプション）
```

---

## 5. 分析ツールを追加する場合

```
以下の分子分析ツールを実装してください：

ツール名: [例: 化学反応性プロファイラー]
目的: [例: 官能基に基づく反応性予測]

参考: scaffold_analysis.py, applicability_domain.pyのパターン
```

---

## 6. 段階的なexampleを作成する場合

```
以下の難易度でexampleを作成してください：

ファイル名: examples/0X_[name].py

ヘッダーに含める情報:
- 難易度（⭐〜⭐⭐⭐⭐）
- 所要時間
- 必要パッケージ
- 学べること

各Partは "=" * 60 で区切り、
コメントで何をしているか説明する。
```

---

## 実装チェックリスト

新機能を追加する際は以下を確認：

- [ ] オプショナル依存の遅延インポート
- [ ] is_available()メソッド
- [ ] エラー時のフォールバック処理
- [ ] ロギング（logging.getLogger(__name__)）
- [ ] タイプヒント
- [ ] docstring（日本語可）
- [ ] __init__.pyへの追加
- [ ] exampleの追加
- [ ] READMEへの追加
