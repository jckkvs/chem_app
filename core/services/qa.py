"""
品質保証フレームワーク

Implements: F-QA-001
設計思想:
- テスト自動生成
- コードカバレッジ
- 回帰テスト
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """テスト結果"""
    name: str
    passed: bool
    duration_ms: float
    error: Optional[str] = None
    output: Any = None


@dataclass
class QAReport:
    """QAレポート"""
    total_tests: int
    passed: int
    failed: int
    pass_rate: float
    total_duration_ms: float
    results: List[TestResult] = field(default_factory=list)


class QualityAssurance:
    """
    品質保証フレームワーク
    
    Features:
    - テストケース管理
    - 自動実行
    - レポート生成
    
    Example:
        >>> qa = QualityAssurance()
        >>> qa.add_test("test_feature", test_func)
        >>> report = qa.run_all()
    """
    
    def __init__(self):
        self.tests: Dict[str, Callable] = {}
        self.fixtures: Dict[str, Any] = {}
        self.last_report: Optional[QAReport] = None
    
    def add_test(
        self,
        name: str,
        func: Callable,
        tags: Optional[List[str]] = None,
    ) -> None:
        """テストを追加"""
        self.tests[name] = func
    
    def add_fixture(self, name: str, value: Any) -> None:
        """フィクスチャを追加"""
        self.fixtures[name] = value
    
    def run_test(self, name: str) -> TestResult:
        """単一テスト実行"""
        if name not in self.tests:
            return TestResult(name=name, passed=False, duration_ms=0, error="Test not found")
        
        func = self.tests[name]
        
        start = time.perf_counter()
        try:
            output = func(**self.fixtures)
            duration = (time.perf_counter() - start) * 1000
            
            return TestResult(
                name=name,
                passed=True,
                duration_ms=duration,
                output=output,
            )
            
        except AssertionError as e:
            duration = (time.perf_counter() - start) * 1000
            return TestResult(
                name=name,
                passed=False,
                duration_ms=duration,
                error=f"Assertion failed: {e}",
            )
            
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return TestResult(
                name=name,
                passed=False,
                duration_ms=duration,
                error=str(e),
            )
    
    def run_all(self, tags: Optional[List[str]] = None) -> QAReport:
        """全テスト実行"""
        results = []
        
        for name in self.tests:
            result = self.run_test(name)
            results.append(result)
        
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        total_duration = sum(r.duration_ms for r in results)
        
        report = QAReport(
            total_tests=len(results),
            passed=passed,
            failed=failed,
            pass_rate=passed / len(results) if results else 0,
            total_duration_ms=total_duration,
            results=results,
        )
        
        self.last_report = report
        return report
    
    def generate_report_html(self, report: QAReport) -> str:
        """HTMLレポート生成"""
        status_color = '#27ae60' if report.pass_rate == 1.0 else '#e74c3c'
        
        test_rows = ""
        for r in report.results:
            status = "✅" if r.passed else "❌"
            error = r.error or ""
            test_rows += f"<tr><td>{status}</td><td>{r.name}</td><td>{r.duration_ms:.1f}ms</td><td>{error}</td></tr>"
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>QA Report</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; padding: 20px; background: #1a1a2e; color: #eee; }}
        .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat {{ background: #16213e; padding: 15px; border-radius: 10px; text-align: center; }}
        .stat-value {{ font-size: 2em; color: {status_color}; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background: #16213e; padding: 10px; }}
        td {{ padding: 8px; border-bottom: 1px solid #333; }}
    </style>
</head>
<body>
    <h1>🧪 QA Report</h1>
    
    <div class="stats">
        <div class="stat">
            <div class="stat-value">{report.pass_rate*100:.0f}%</div>
            <div>Pass Rate</div>
        </div>
        <div class="stat">
            <div class="stat-value">{report.passed}/{report.total_tests}</div>
            <div>Tests Passed</div>
        </div>
        <div class="stat">
            <div class="stat-value">{report.total_duration_ms:.0f}ms</div>
            <div>Total Duration</div>
        </div>
    </div>
    
    <table>
        <tr><th>Status</th><th>Test Name</th><th>Duration</th><th>Error</th></tr>
        {test_rows}
    </table>
    
    <p style="color: #666;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</body>
</html>
"""
