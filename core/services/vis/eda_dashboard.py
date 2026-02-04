"""
インタラクティブEDAダッシュボード - 人間中心のデータ観察支援

Implements: F-EDA-001
設計思想:
- 人間がデータを丁寧に見ることこそが最良のデータサイエンス
- モデル構築前の徹底的なデータ理解を支援
- pairplot、相関分析、分布確認を1クリックで

機能:
- 自動データプロファイリング
- インタラクティブpairplot
- 相関ヒートマップ
- 分布・外れ値可視化
- クラスタリング可視化
- PCA/t-SNE探索
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class EDADashboard:
    """
    人間中心のEDAダッシュボード
    
    「データを丁寧に見る」ための包括的なツールセット。
    
    Usage:
        dashboard = EDADashboard(df)
        dashboard.generate_full_report()
        dashboard.plot_pairplot()
        dashboard.plot_correlation_heatmap()
    """
    
    def __init__(self, df: pd.DataFrame, target_column: Optional[str] = None):
        """
        Args:
            df: 分析対象DataFrame
            target_column: ターゲット変数名（あれば）
        """
        self.df = df.copy()
        self.target_column = target_column
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"EDADashboard初期化: {len(df)}行 x {len(df.columns)}列")
        logger.info(f"数値列: {len(self.numeric_cols)}, カテゴリ列: {len(self.categorical_cols)}")
    
    # ========== データプロファイリング ==========
    
    def generate_data_profile(self) -> Dict[str, Any]:
        """
        包括的なデータプロファイル生成
        
        Returns:
            データ統計情報の辞書
        """
        profile = {
            'basic_info': {
                'n_rows': len(self.df),
                'n_columns': len(self.df.columns),
                'n_numeric': len(self.numeric_cols),
                'n_categorical': len(self.categorical_cols),
                'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            },
            'missing_values': self.df.isnull().sum().to_dict(),
            'missing_percent': (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
            'duplicates': {
                'n_duplicates': self.df.duplicated().sum(),
                'duplicate_percent': self.df.duplicated().sum() / len(self.df) * 100,
            },
        }
        
        # 数値列の統計量
        if self.numeric_cols:
            numeric_stats = self.df[self.numeric_cols].describe().to_dict()
            profile['numeric_stats'] = numeric_stats
            
            # 歪度・尖度
            profile['skewness'] = self.df[self.numeric_cols].skew().to_dict()
            profile['kurtosis'] = self.df[self.numeric_cols].kurt().to_dict()
        
        # カテゴリ列の統計
        if self.categorical_cols:
            cat_stats = {}
            for col in self.categorical_cols:
                cat_stats[col] = {
                    'n_unique': self.df[col].nunique(),
                    'top_values': self.df[col].value_counts().head(5).to_dict(),
                }
            profile['categorical_stats'] = cat_stats
        
        return profile
    
    def print_data_summary(self):
        """データサマリーを人間が読みやすい形式で出力"""
        profile = self.generate_data_profile()
        
        print("=" * 80)
        print("📊 データプロファイルサマリー")
        print("=" * 80)
        
        # 基本情報
        basic = profile['basic_info']
        print(f"\n【基本情報】")
        print(f"  行数: {basic['n_rows']:,}")
        print(f"  列数: {basic['n_columns']}")
        print(f"  数値列: {basic['n_numeric']}, カテゴリ列: {basic['n_categorical']}")
        print(f"  メモリ使用量: {basic['memory_usage_mb']:.2f} MB")
        
        # 欠損値
        missing_cols = {k: v for k, v in profile['missing_values'].items() if v > 0}
        if missing_cols:
            print(f"\n【欠損値】")
            for col, count in sorted(missing_cols.items(), key=lambda x: x[1], reverse=True)[:10]:
                pct = profile['missing_percent'][col]
                print(f"  {col}: {count:,} ({pct:.1f}%)")
        else:
            print(f"\n【欠損値】なし ✓")
        
        # 重複
        dup = profile['duplicates']
        print(f"\n【重複行】")
        print(f"  重複数: {dup['n_duplicates']:,} ({dup['duplicate_percent']:.2f}%)")
        
        # 数値列サマリー
        if 'numeric_stats' in profile:
            print(f"\n【数値列サマリー（トップ5）】")
            skew_sorted = sorted(profile['skewness'].items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            for col, skew in skew_sorted:
                kurt = profile['kurtosis'][col]
                print(f"  {col}: 歪度={skew:.2f}, 尖度={kurt:.2f}")
        
        print("=" * 80)
    
    # ========== 相関分析 ==========
    
    def plot_correlation_heatmap(
        self,
        method: str = 'pearson',
        figsize: Tuple[int, int] = (12, 10),
        top_n: Optional[int] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        相関ヒートマップ（人間が一目で相関を把握できる）
        
        Args:
            method: 'pearson', 'spearman', 'kendall'
            figsize: 図サイズ
            top_n: 上位N特徴量のみ表示（Noneで全表示）
            show: 表示するか
        
        Returns:
            matplotlib Figure
        """
        if not self.numeric_cols:
            logger.warning("数値列がありません")
            return None
        
        df_numeric = self.df[self.numeric_cols]
        
        # 上位N特徴量
        if top_n and len(self.numeric_cols) > top_n:
            # 分散が大きい列を選択
            variances = df_numeric.var().sort_values(ascending=False)
            top_cols = variances.head(top_n).index
            df_numeric = df_numeric[top_cols]
        
        # 相関計算
        corr = df_numeric.corr(method=method)
        
        # プロット
        fig, ax = plt.subplots(figsize=figsize)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(
            corr,
            mask=mask,
            annot=True if len(corr) <= 15 else False,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
        
        ax.set_title(f'相関ヒートマップ ({method.capitalize()})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig
    
    def find_high_correlations(self, threshold: float = 0.7) -> pd.DataFrame:
        """
        高相関ペアを発見（多重共線性の確認）
        
        Args:
            threshold: 相関係数の閾値
        
        Returns:
            高相関ペアのDataFrame
        """
        if not self.numeric_cols:
            return pd.DataFrame()
        
        corr = self.df[self.numeric_cols].corr()
        
        # 上三角のみ
        pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                if abs(corr.iloc[i, j]) >= threshold:
                    pairs.append({
                        'feature_1': corr.columns[i],
                        'feature_2': corr.columns[j],
                        'correlation': corr.iloc[i, j],
                    })
        
        result = pd.DataFrame(pairs)
        if not result.empty:
            result = result.sort_values('correlation', key=abs, ascending=False)
        
        return result
    
    # ========== Pairplot（散布図行列）==========
    
    def plot_pairplot(
        self,
        columns: Optional[List[str]] = None,
        hue: Optional[str] = None,
        max_features: int = 8,
        show: bool = True,
    ):
        """
        Pairplot - データの関係性を一目で把握
        
        Args:
            columns: プロット対象列（Noneで自動選択）
            hue: 色分け用カラム
            max_features: 最大特徴量数
            show: 表示するか
        
        Returns:
            seaborn PairGrid
        """
        if columns is None:
            # デフォルト: 分散が大きい上位N列
            if len(self.numeric_cols) > max_features:
                variances = self.df[self.numeric_cols].var().sort_values(ascending=False)
                columns = variances.head(max_features).index.tolist()
            else:
                columns = self.numeric_cols
        
        if not columns:
            logger.warning("プロット対象列がありません")
            return None
        
        # hue列を追加
        plot_cols = columns.copy()
        if hue and hue not in plot_cols:
            plot_cols.append(hue)
        
        # プロット
        try:
            grid = sns.pairplot(
                self.df[plot_cols],
                hue=hue,
                diag_kind='kde',
                plot_kws={'alpha': 0.6},
                height=2.5,
            )
            grid.fig.suptitle('Pairplot - データの全体像', y=1.01, fontsize=16, fontweight='bold')
            
            if show:
                plt.show()
            
            return grid
        except Exception as e:
            logger.error(f"Pairplot生成エラー: {e}")
            return None
    
    # ========== 分布可視化 ==========
    
    def plot_distributions(
        self,
        columns: Optional[List[str]] = None,
        n_cols: int = 3,
        figsize: Optional[Tuple[int, int]] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        全特徴量の分布を一覧表示（ヒストグラム + KDE）
        
        Args:
            columns: プロット対象列
            n_cols: 列数
            figsize: 図サイズ
            show: 表示するか
        
        Returns:
            matplotlib Figure
        """
        if columns is None:
            columns = self.numeric_cols
        
        if not columns:
            logger.warning("プロット対象列がありません")
            return None
        
        n_features = len(columns)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        if figsize is None:
            figsize = (n_cols * 5, n_rows * 4)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, col in enumerate(columns):
            ax = axes[idx]
            data = self.df[col].dropna()
            
            # ヒストグラム + KDE
            sns.histplot(data, kde=True, ax=ax, color='steelblue', alpha=0.6)
            
            # 統計量
            mean_val = data.mean()
            median_val = data.median()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
            
            ax.set_title(f'{col}\n歪度={data.skew():.2f}, 尖度={data.kurt():.2f}', fontweight='bold')
            ax.legend()
        
        # 空白セル非表示
        for idx in range(n_features, len(axes)):
            fig.delaxes(axes[idx])
        
        fig.suptitle('特徴量分布一覧', fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig
    
    # ========== 次元削減可視化 ==========
    
    def plot_pca(
        self,
        n_components: int = 2,
        color_by: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        show: bool = True,
    ) -> Tuple[plt.Figure, PCA]:
        """
        PCA可視化（データの主要構造を把握）
        
        Args:
            n_components: 主成分数
            color_by: 色分け用カラム
            figsize: 図サイズ
            show: 表示するか
        
        Returns:
            (Figure, PCAモデル)
        """
        if not self.numeric_cols:
            logger.warning("数値列がありません")
            return None, None
        
        # データ準備
        X = self.df[self.numeric_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # プロット
        fig, ax = plt.subplots(figsize=figsize)
        
        if color_by and color_by in self.df.columns:
            colors = self.df[color_by]
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, cmap='viridis', alpha=0.6, s=50)
            plt.colorbar(scatter, ax=ax, label=color_by)
        else:
            ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=50, color='steelblue')
        
        # 寄与率
        var_ratio = pca.explained_variance_ratio_
        ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}%)', fontsize=12)
        ax.set_title(f'PCA - 累積寄与率: {var_ratio.sum()*100:.1f}%', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        logger.info(f"PCA寄与率: {var_ratio}")
        
        return fig, pca
    
    def plot_tsne(
        self,
        perplexity: int = 30,
        color_by: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        show: bool = True,
    ) -> plt.Figure:
        """
        t-SNE可視化（非線形構造の把握）
        
        Args:
            perplexity: パープレキシティ
            color_by: 色分け用カラム
            figsize: 図サイズ
            show: 表示するか
        
        Returns:
            matplotlib Figure
        """
        if not self.numeric_cols:
            logger.warning("数値列がありません")
            return None
        
        # データ準備
        X = self.df[self.numeric_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # t-SNE
        logger.info("t-SNE実行中（時間がかかる場合があります）...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_tsne = tsne.fit_transform(X_scaled)
        
        # プロット
        fig, ax = plt.subplots(figsize=figsize)
        
        if color_by and color_by in self.df.columns:
            colors = self.df[color_by]
            scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, cmap='viridis', alpha=0.6, s=50)
            plt.colorbar(scatter, ax=ax, label=color_by)
        else:
            ax.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6, s=50, color='steelblue')
        
        ax.set_xlabel('t-SNE Component 1', fontsize=12)
        ax.set_ylabel('t-SNE Component 2', fontsize=12)
        ax.set_title(f't-SNE (perplexity={perplexity})', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig
    
    # ========== クラスタリング可視化 ==========
    
    def plot_kmeans_clusters(
        self,
        n_clusters: int = 3,
        use_pca: bool = True,
        figsize: Tuple[int, int] = (10, 8),
        show: bool = True,
    ) -> Tuple[plt.Figure, KMeans]:
        """
        K-meansクラスタリング可視化
        
        Args:
            n_clusters: クラスタ数
            use_pca: PCAで2次元化するか
            figsize: 図サイズ
            show: 表示するか
        
        Returns:
            (Figure, KMeansモデル)
        """
        if not self.numeric_cols:
            logger.warning("数値列がありません")
            return None, None
        
        # データ準備
        X = self.df[self.numeric_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # クラスタリング
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # 可視化用に2次元化
        if use_pca:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X_scaled)
            centers_2d = pca.transform(kmeans.cluster_centers_)
        else:
            # 最初の2列を使用
            X_2d = X_scaled[:, :2]
            centers_2d = kmeans.cluster_centers_[:, :2]
        
        # プロット
        fig, ax = plt.subplots(figsize=figsize)
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, cmap='viridis', alpha=0.6, s=50)
        ax.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X', s=300, edgecolor='black', linewidth=2, label='Centroids')
        
        ax.set_xlabel('Component 1', fontsize=12)
        ax.set_ylabel('Component 2', fontsize=12)
        ax.set_title(f'K-means クラスタリング (k={n_clusters})', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=ax, label='Cluster')
        plt.tight_layout()
        
        if show:
            plt.show()
        
        logger.info(f"K-means完了: Inertia={kmeans.inertia_:.2f}")
        
        return fig, kmeans
    
    # ========== フルレポート生成 ==========
    
    def generate_full_report(self, output_dir: str = './eda_report') -> Dict[str, Any]:
        """
        フルEDAレポート生成（全可視化を一括生成）
        
        Args:
            output_dir: 出力ディレクトリ
        
        Returns:
            レポートメタデータ
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("📊 フルEDAレポート生成開始...")
        
        report_meta = {
            'n_plots': 0,
            'files': [],
        }
        
        # 1. データプロファイル
        self.print_data_summary()
        
        # 2. 相関ヒートマップ
        try:
            fig = self.plot_correlation_heatmap(show=False)
            if fig:
                path = os.path.join(output_dir, '01_correlation_heatmap.png')
                fig.savefig(path, dpi=150, bbox_inches='tight')
                report_meta['files'].append(path)
                report_meta['n_plots'] += 1
                plt.close(fig)
        except Exception as e:
            logger.error(f"相関ヒートマップ生成エラー: {e}")
        
        # 3. Pairplot
        try:
            grid = self.plot_pairplot(show=False, max_features=6)
            if grid:
                path = os.path.join(output_dir, '02_pairplot.png')
                grid.savefig(path, dpi=150, bbox_inches='tight')
                report_meta['files'].append(path)
                report_meta['n_plots'] += 1
                plt.close()
        except Exception as e:
            logger.error(f"Pairplot生成エラー: {e}")
        
        # 4. 分布
        try:
            fig = self.plot_distributions(show=False)
            if fig:
                path = os.path.join(output_dir, '03_distributions.png')
                fig.savefig(path, dpi=150, bbox_inches='tight')
                report_meta['files'].append(path)
                report_meta['n_plots'] += 1
                plt.close(fig)
        except Exception as e:
            logger.error(f"分布プロット生成エラー: {e}")
        
        # 5. PCA
        try:
            fig, pca_model = self.plot_pca(show=False)
            if fig:
                path = os.path.join(output_dir, '04_pca.png')
                fig.savefig(path, dpi=150, bbox_inches='tight')
                report_meta['files'].append(path)
                report_meta['n_plots'] += 1
                plt.close(fig)
        except Exception as e:
            logger.error(f"PCAプロット生成エラー: {e}")
        
        # 6. K-means
        try:
            fig, kmeans_model = self.plot_kmeans_clusters(show=False)
            if fig:
                path = os.path.join(output_dir, '05_kmeans.png')
                fig.savefig(path, dpi=150, bbox_inches='tight')
                report_meta['files'].append(path)
                report_meta['n_plots'] += 1
                plt.close(fig)
        except Exception as e:
            logger.error(f"K-meansプロット生成エラー: {e}")
        
        logger.info(f"✅ フルEDAレポート生成完了: {report_meta['n_plots']}個のプロット生成")
        logger.info(f"📁 出力先: {output_dir}")
        
        return report_meta


# ========== ユーティリティ関数 ==========

def quick_eda(df: pd.DataFrame, target_column: Optional[str] = None) -> EDADashboard:
    """
    クイックEDA - 1行でデータを把握
    
    Usage:
        >>> dashboard = quick_eda(df, target_column='target')
        >>> # 自動的にサマリー表示 + 主要プロット生成
    
    Args:
        df: DataFrame
        target_column: ターゲット変数名
    
    Returns:
        EDADashboard
    """
    dashboard = EDADashboard(df, target_column)
    dashboard.print_data_summary()
    
    print("\n💡 主要な相関ペアを確認中...")
    high_corr = dashboard.find_high_correlations(threshold=0.7)
    if not high_corr.empty:
        print(high_corr.head(10).to_string(index=False))
    else:
        print("  高相関ペア（r>0.7）は見つかりませんでした。")
    
    return dashboard
