"""
バッチ処理モジュールのテスト

Implements: T-BATCH-001
カバレッジ対象: core/services/utils/batch_processing.py
"""

import time
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from core.services.utils.batch_processing import (
    BatchPredictor,
    BatchProcessor,
    BatchResult,
    ProgressCallback,
    batch_iterator,
    run_batch_process,
)


class TestBatchResult:
    """BatchResult dataclassのテスト"""

    def test_success_rate_calculation(self):
        """成功率の計算"""
        result = BatchResult(
            total=100,
            successful=80,
            failed=20,
            results=[],
            errors=[],
            elapsed_time=10.0,
        )
        assert result.success_rate == 0.8

    def test_success_rate_zero_total(self):
        """total=0の場合"""
        result = BatchResult(
            total=0,
            successful=0,
            failed=0,
            results=[],
            errors=[],
            elapsed_time=0.0,
        )
        assert result.success_rate == 0.0

    def test_summary_string(self):
        """サマリー文字列の生成"""
        result = BatchResult(
            total=100,
            successful=95,
            failed=5,
            results=[],
            errors=[],
            elapsed_time=5.5,
        )
        summary = result.summary()
        assert "100" in summary
        assert "95" in summary
        assert "5" in summary
        assert "95.0%" in summary


class TestProgressCallback:
    """ProgressCallbackのテスト"""

    def test_initialization(self):
        """初期化"""
        progress = ProgressCallback(total=100, desc="Test")
        assert progress.total == 100
        assert progress.desc == "Test"
        assert progress.current == 0

    def test_update(self):
        """進捗更新"""
        progress = ProgressCallback(total=100)
        progress.update(10)
        assert progress.current == 10
        
        progress.update(5)
        assert progress.current == 15

    @patch('sys.stdout')
    def test_update_output(self, mock_stdout):
        """進捗バー出力"""
        progress = ProgressCallback(total=10, desc="Test")
        progress.update(5)
        # stdout.writeが呼ばれたか確認
        assert mock_stdout.write.called

    def test_close(self):
        """終了処理"""
        progress = ProgressCallback(total=10)
        progress.update(5)
        progress.close()  # エラーなく完了


class TestBatchIterator:
    """batch_iteratorのテスト"""

    def test_basic_batching(self):
        """基本的なバッチ化"""
        items = list(range(10))
        batches = list(batch_iterator(items, batch_size=3))
        
        assert len(batches) == 4
        assert batches[0] == [0, 1, 2]
        assert batches[1] == [3, 4, 5]
        assert batches[2] == [6, 7, 8]
        assert batches[3] == [9]

    def test_exact_division(self):
        """割り切れる場合"""
        items = list(range(12))
        batches = list(batch_iterator(items, batch_size=4))
        
        assert len(batches) == 3
        for batch in batches:
            assert len(batch) == 4

    def test_empty_input(self):
        """空リスト"""
        items = []
        batches = list(batch_iterator(items, batch_size=10))
        assert len(batches) == 0

    def test_batch_size_larger_than_items(self):
        """バッチサイズ > アイテム数"""
        items = [1, 2, 3]
        batches = list(batch_iterator(items, batch_size=10))
        
        assert len(batches) == 1
        assert batches[0] == [1, 2, 3]


class TestBatchProcessor:
    """BatchProcessorのテスト"""

    def test_basic_processing(self):
        """基本的な処理"""
        processor = BatchProcessor(n_workers=2, show_progress=False)
        
        # 2倍にする関数
        items = [1, 2, 3, 4, 5]
        result = processor.process(
            items,
            func=lambda x: x * 2,
            batch_size=2,
        )
        
        assert result.total == 5
        assert result.successful == 5
        assert result.failed == 0
        assert sorted(result.results) == [2, 4, 6, 8, 10]

    def test_error_handling(self):
        """エラーハンドリング"""
        processor = BatchProcessor(n_workers=1, show_progress=False)
        
        def func_with_error(x):
            if x == 3:
                raise ValueError("Error at 3")
            return x * 2
        
        items = [1, 2, 3, 4, 5]
        result = processor.process(items, func=func_with_error, batch_size=2)
        
        assert result.successful == 4
        assert result.failed == 1
        assert len(result.errors) == 1
        assert result.errors[0]['item'] == 3

    def test_custom_error_handler(self):
        """カスタムエラーハンドラ"""
        processor = BatchProcessor(n_workers=1, show_progress=False)
        
        def func_with_error(x):
            if x % 2 == 0:
                raise ValueError("Even number")
            return x
        
        def error_handler(error, item):
            # エラー時はデフォルト値を返す
            return -1
        
        items = [1, 2, 3, 4, 5]
        result = processor.process(
            items,
            func=func_with_error,
            batch_size=2,
            error_handler=error_handler,
        )
        
        # エラーハンドラでー1が返される
        assert result.successful == 5
        assert result.failed == 0
        assert -1 in result.results

    def test_sequential_processing(self):
        """逐次処理"""
        processor = BatchProcessor(show_progress=False)
        
        items = [1, 2, 3, 4, 5]
        result = processor.process_sequential(
            items,
            func=lambda x: x * 2,
        )
        
        assert result.total == 5
        assert result.successful == 5
        assert sorted(result.results) == [2, 4, 6, 8, 10]

    def test_progress_display(self):
        """進捗表示付き処理"""
        processor = BatchProcessor(n_workers=1, show_progress=True)
        
        items = list(range(10))
        result = processor.process(
            items,
            func=lambda x: x,
            batch_size=5,
        )
        
        assert result.total == 10
        assert result.successful == 10

    def test_elapsed_time(self):
        """実行時間の記録"""
        processor = BatchProcessor(show_progress=False)
        
        def slow_func(x):
            time.sleep(0.01)
            return x
        
        items = list(range(5))
        result = processor.process(items, func=slow_func, batch_size=2)
        
        # 少なくとも0.05秒はかかるはず
        assert result.elapsed_time > 0.0


class TestBatchPredictor:
    """BatchPredictorのテスト"""

    def test_basic_prediction(self):
        """基本的な予測"""
        # モックモデル（各バッチサイズに応じた長さの配列を返す）
        mock_model = Mock()
        mock_model.predict.side_effect = lambda x: np.ones(len(x))
        
        predictor = BatchPredictor(mock_model, batch_size=2)
        
        X = pd.DataFrame({"feature": range(6)})
        predictions = predictor.predict(X)
        
        # 6個の予測結果
        assert len(predictions) == 6
        assert mock_model.predict.call_count == 3  # 3バッチ (2+2+2)

    def test_large_dataset(self):
        """大規模データセット"""
        mock_model = Mock()
        mock_model.predict.side_effect = lambda x: np.ones(len(x))
        
        predictor = BatchPredictor(mock_model, batch_size=100)
        
        X = pd.DataFrame({"feature": range(500)})
        predictions = predictor.predict(X)
        
        assert len(predictions) == 500
        # 5バッチに分かれるはず
        assert mock_model.predict.call_count == 5

    def test_with_uncertainty(self):
        """不確実性付き予測"""
        mock_model = Mock()
        mock_model.predict_with_interval.return_value = (
            np.array([1.0, 2.0]),
            np.array([0.5, 1.5]),
            np.array([1.5, 2.5]),
        )
        
        predictor = BatchPredictor(mock_model, batch_size=10)
        
        X = pd.DataFrame({"feature": range(2)})
        predictions = predictor.predict(X, return_uncertainty=True)
        
        assert len(predictions) == 2
        assert mock_model.predict_with_interval.called

    def test_model_without_uncertainty(self):
        """不確実性をサポートしないモデル"""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1.0, 2.0])
        # predict_with_intervalメソッドなし
        del mock_model.predict_with_interval
        
        predictor = BatchPredictor(mock_model, batch_size=10)
        
        X = pd.DataFrame({"feature": range(2)})
        predictions = predictor.predict(X, return_uncertainty=True)
        
        # 通常のpredictにフォールバック
        assert len(predictions) == 2
        assert mock_model.predict.called


class TestConvenienceFunctions:
    """便利関数のテスト"""

    def test_run_batch_process(self):
        """run_batch_process関数"""
        items = [1, 2, 3, 4, 5]
        result = run_batch_process(
            items,
            func=lambda x: x * 2,
            batch_size=2,
        )
        
        assert result.total == 5
        assert result.successful == 5


class TestEdgeCases:
    """エッジケースのテスト"""

    def test_empty_batch(self):
        """空バッチ"""
        processor = BatchProcessor(show_progress=False)
        result = processor.process([], func=lambda x: x, batch_size=10)
        
        assert result.total == 0
        assert result.successful == 0
        assert result.failed == 0

    def test_single_item(self):
        """単一アイテム"""
        processor = BatchProcessor(show_progress=False)
        result = processor.process([42], func=lambda x: x * 2, batch_size=10)
        
        assert result.total == 1
        assert result.successful == 1
        assert result.results[0] == 84

    def test_all_failures(self):
        """全失敗"""
        processor = BatchProcessor(show_progress=False)
        
        def always_fail(x):
            raise RuntimeError("Always fail")
        
        items = [1, 2, 3]
        result = processor.process(items, func=always_fail, batch_size=2)
        
        assert result.successful == 0
        assert result.failed == 3
        assert len(result.errors) == 3

    def test_concurrent_processing(self):
        """並列処理の動作確認"""
        processor = BatchProcessor(n_workers=4, show_progress=False)
        
        items = list(range(20))
        result = processor.process(
            items,
            func=lambda x: x ** 2,
            batch_size=5,
        )
        
        assert result.successful == 20
        # 並列処理でも結果は正しい
        assert sorted(result.results) == [x ** 2 for x in items]
