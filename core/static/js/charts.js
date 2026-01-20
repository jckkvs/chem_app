/**
 * Chart.js 設定とヘルパー関数
 */

// グローバルChart.js設定
if (typeof Chart !== 'undefined') {
    Chart.defaults.color = '#e0e0e0';
    Chart.defaults.borderColor = '#333';
    Chart.defaults.backgroundColor = 'rgba(102, 126, 234, 0.2)';
}

// メトリクス比較バーチャート
function createMetricsComparisonChart(canvasId, experiments) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const metricNames = ['r2', 'rmse', 'mae'];

    const datasets = metricNames.map((metric, i) => ({
        label: metric.toUpperCase(),
        data: experiments.map(e => e.metrics[metric] || 0),
        backgroundColor: [
            'rgba(102, 126, 234, 0.6)',
            'rgba(118, 75, 162, 0.6)',
            'rgba(237, 137, 54, 0.6)'
        ][i]
    }));

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: experiments.map(e => e.name),
            datasets
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'top' },
                title: { display: true, text: 'メトリクス比較' }
            }
        }
    });
}

// 学習曲線
function createLearningCurveChart(canvasId, trainScores, valScores) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: trainScores.map((_, i) => `Fold ${i + 1}`),
            datasets: [
                {
                    label: 'Training',
                    data: trainScores,
                    borderColor: 'rgb(102, 126, 234)',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                },
                {
                    label: 'Validation',
                    data: valScores,
                    borderColor: 'rgb(237, 137, 54)',
                    backgroundColor: 'rgba(237, 137, 54, 0.1)',
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'top' },
                title: { display: true, text: 'Learning Curve' }
            },
            scales: {
                y: { beginAtZero: false }
            }
        }
    });
}

// 特徴量重要度
function createFeatureImportanceChart(canvasId, features, importances) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    // Top 20のみ表示
    const topN = 20;
    const sorted = features
        .map((name, i) => ({ name, importance: importances[i] }))
        .sort((a, b) => b.importance - a.importance)
        .slice(0, topN);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: sorted.map(f => f.name),
            datasets: [{
                label: 'Importance',
                data: sorted.map(f => f.importance),
                backgroundColor: 'rgba(102, 126, 234, 0.6)',
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            plugins: {
                legend: { display: false },
                title: { display: true, text: '特徴量重要度 (Top 20)' }
            }
        }
    });
}

// 予測 vs 実測散布図
function createPredVsActualChart(canvasId, actual, predicted) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const data = actual.map((y, i) => ({ x: y, y: predicted[i] }));
    const min = Math.min(...actual, ...predicted);
    const max = Math.max(...actual, ...predicted);

    new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Predictions',
                data,
                backgroundColor: 'rgba(102, 126, 234, 0.6)',
            }, {
                label: 'Ideal',
                data: [{ x: min, y: min }, { x: max, y: max }],
                type: 'line',
                borderColor: 'rgba(245, 101, 101, 0.6)',
                borderDash: [5, 5],
                pointRadius: 0,
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'top' },
                title: { display: true, text: 'Predicted vs Actual' }
            },
            scales: {
                x: { title: { display: true, text: 'Actual' } },
                y: { title: { display: true, text: 'Predicted' } }
            }
        }
    });
}

// エクスポート
window.createMetricsComparisonChart = createMetricsComparisonChart;
window.createLearningCurveChart = createLearningCurveChart;
window.createFeatureImportanceChart = createFeatureImportanceChart;
window.createPredVsActualChart = createPredVsActualChart;
