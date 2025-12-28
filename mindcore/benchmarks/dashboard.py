"""Benchmark Dashboard Generator.

Generates an interactive HTML dashboard for benchmark results visualization.
Uses Chart.js for charts and a clean, modern design.

Usage:
    from mindcore.benchmarks.dashboard import generate_dashboard
    generate_dashboard("benchmark_results.json", "dashboard.html")

    # Or via CLI:
    python -m mindcore.benchmarks.dashboard benchmark_results.json
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any


def generate_dashboard(results_file: str, output_file: str = "dashboard.html") -> str:
    """Generate an HTML dashboard from benchmark results.

    Args:
        results_file: Path to benchmark_results.json
        output_file: Output HTML file path

    Returns:
        Path to generated dashboard
    """
    with open(results_file) as f:
        data = json.load(f)

    suite = data["results"][0]
    benchmarks = suite["benchmarks"]

    # Extract metrics for charts
    latency_data = []
    quality_data = []
    cost_data = []

    for b in benchmarks:
        m = b["metrics"]
        name = b["name"].replace("_", " ").title()

        if m["latency"]["count"] > 0:
            latency_data.append(
                {
                    "name": name,
                    "p50": m["latency"]["p50_ms"],
                    "p95": m["latency"]["p95_ms"],
                    "p99": m["latency"]["p99_ms"],
                }
            )

        if m["quality"]["true_positives"] > 0 or m["quality"]["false_negatives"] > 0:
            quality_data.append(
                {
                    "name": name,
                    "precision": m["quality"]["precision"],
                    "recall": m["quality"]["recall"],
                    "f1": m["quality"]["f1"],
                }
            )

        if m["cost"]["flr_queries"] > 0:
            cost_data.append(
                {
                    "name": name,
                    "flr_p50": m["cost"]["flr_latency"]["p50_ms"],
                    "clst_p50": m["cost"]["clst_latency"]["p50_ms"],
                }
            )

    html = _generate_html(suite, benchmarks, latency_data, quality_data, cost_data)

    with open(output_file, "w") as f:
        f.write(html)

    return output_file


def _generate_html(
    suite: dict[str, Any],
    benchmarks: list[dict[str, Any]],
    latency_data: list[dict[str, Any]],
    quality_data: list[dict[str, Any]],
    cost_data: list[dict[str, Any]],
) -> str:
    """Generate the HTML content."""
    passed = suite["passed"]
    total = suite["total"]
    pass_rate = (passed / total) * 100 if total > 0 else 0

    # Generate benchmark cards
    benchmark_cards = ""
    for b in benchmarks:
        status = "pass" if b["passed"] else "fail"
        status_icon = "check_circle" if b["passed"] else "cancel"
        status_color = "#10b981" if b["passed"] else "#ef4444"

        name = b["name"].replace("_", " ").title()
        m = b["metrics"]

        # Build metrics summary
        metrics_html = ""
        if m["latency"]["count"] > 0:
            metrics_html += f"""
                <div class="metric">
                    <span class="metric-label">p50 Latency</span>
                    <span class="metric-value">{m["latency"]["p50_ms"]:.2f}ms</span>
                </div>
                <div class="metric">
                    <span class="metric-label">p99 Latency</span>
                    <span class="metric-value">{m["latency"]["p99_ms"]:.2f}ms</span>
                </div>
            """

        if m["quality"]["true_positives"] > 0 or m["quality"]["false_negatives"] > 0:
            metrics_html += f"""
                <div class="metric">
                    <span class="metric-label">Recall</span>
                    <span class="metric-value">{m["quality"]["recall"]:.0%}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Precision</span>
                    <span class="metric-value">{m["quality"]["precision"]:.0%}</span>
                </div>
            """

        if m["determinism"]["replay_runs"] > 0:
            metrics_html += f"""
                <div class="metric">
                    <span class="metric-label">Consistency</span>
                    <span class="metric-value">{m["determinism"]["replay_consistency"]:.0%}</span>
                </div>
            """

        if m["robustness"]["noisy_inputs"] > 0:
            metrics_html += f"""
                <div class="metric">
                    <span class="metric-label">Noise Resistance</span>
                    <span class="metric-value">{m["robustness"]["noise_resistance"]:.0%}</span>
                </div>
            """

        if m["drift"]["total_preferences_tracked"] > 0:
            metrics_html += f"""
                <div class="metric">
                    <span class="metric-label">Drift Rate</span>
                    <span class="metric-value">{m["drift"]["drift_rate"]:.1%}</span>
                </div>
            """

        benchmark_cards += f"""
            <div class="benchmark-card {status}">
                <div class="card-header">
                    <span class="material-icons" style="color: {status_color}">{status_icon}</span>
                    <h3>{name}</h3>
                </div>
                <div class="card-metrics">
                    {metrics_html}
                </div>
            </div>
        """

    # Generate chart data
    latency_labels = json.dumps([d["name"] for d in latency_data])
    latency_p50 = json.dumps([d["p50"] for d in latency_data])
    latency_p95 = json.dumps([d["p95"] for d in latency_data])
    latency_p99 = json.dumps([d["p99"] for d in latency_data])

    cost_labels = json.dumps([d["name"] for d in cost_data])
    flr_data = json.dumps([d["flr_p50"] for d in cost_data])
    clst_data = json.dumps([d["clst_p50"] for d in cost_data])

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mindcore Benchmark Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e4e4e7;
            min-height: 100vh;
            padding: 2rem;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        header {{
            text-align: center;
            margin-bottom: 3rem;
        }}

        h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}

        .subtitle {{
            color: #a1a1aa;
            font-size: 1rem;
        }}

        .summary {{
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-bottom: 3rem;
        }}

        .summary-card {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 1.5rem 2.5rem;
            text-align: center;
            backdrop-filter: blur(10px);
        }}

        .summary-value {{
            font-size: 3rem;
            font-weight: 700;
            color: #10b981;
        }}

        .summary-label {{
            color: #a1a1aa;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }}

        .section-title {{
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .benchmarks-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
        }}

        .benchmark-card {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1.25rem;
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .benchmark-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
        }}

        .benchmark-card.pass {{
            border-left: 4px solid #10b981;
        }}

        .benchmark-card.fail {{
            border-left: 4px solid #ef4444;
        }}

        .card-header {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1rem;
        }}

        .card-header h3 {{
            font-size: 1rem;
            font-weight: 600;
        }}

        .card-metrics {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.75rem;
        }}

        .metric {{
            display: flex;
            flex-direction: column;
        }}

        .metric-label {{
            font-size: 0.75rem;
            color: #a1a1aa;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .metric-value {{
            font-size: 1.25rem;
            font-weight: 600;
            color: #f4f4f5;
        }}

        .charts-section {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 2rem;
            margin-bottom: 3rem;
        }}

        .chart-card {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 1.5rem;
        }}

        .chart-title {{
            font-size: 1.125rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }}

        footer {{
            text-align: center;
            color: #71717a;
            font-size: 0.875rem;
            padding-top: 2rem;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }}

        @media (max-width: 768px) {{
            .summary {{
                flex-direction: column;
                align-items: center;
            }}

            .charts-section {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Mindcore Benchmark Dashboard</h1>
            <p class="subtitle">Industry-standard evaluation for production AI memory</p>
        </header>

        <div class="summary">
            <div class="summary-card">
                <div class="summary-value">{passed}/{total}</div>
                <div class="summary-label">Benchmarks Passed</div>
            </div>
            <div class="summary-card">
                <div class="summary-value" style="color: {'#10b981' if pass_rate == 100 else '#eab308'}">{pass_rate:.0f}%</div>
                <div class="summary-label">Pass Rate</div>
            </div>
            <div class="summary-card">
                <div class="summary-value">{suite.get('suite', 'full').upper()}</div>
                <div class="summary-label">Suite</div>
            </div>
        </div>

        <section>
            <h2 class="section-title">
                <span class="material-icons">assessment</span>
                Benchmark Results
            </h2>
            <div class="benchmarks-grid">
                {benchmark_cards}
            </div>
        </section>

        <section class="charts-section">
            <div class="chart-card">
                <h3 class="chart-title">Latency Distribution (ms)</h3>
                <canvas id="latencyChart"></canvas>
            </div>
            <div class="chart-card">
                <h3 class="chart-title">Hot Path vs Cold Path (ms)</h3>
                <canvas id="costChart"></canvas>
            </div>
        </section>

        <footer>
            <p>Generated on {timestamp} | Mindcore Benchmark Suite</p>
        </footer>
    </div>

    <script>
        // Latency Chart
        const latencyCtx = document.getElementById('latencyChart').getContext('2d');
        new Chart(latencyCtx, {{
            type: 'bar',
            data: {{
                labels: {latency_labels},
                datasets: [
                    {{
                        label: 'p50',
                        data: {latency_p50},
                        backgroundColor: 'rgba(102, 126, 234, 0.8)',
                        borderRadius: 4,
                    }},
                    {{
                        label: 'p95',
                        data: {latency_p95},
                        backgroundColor: 'rgba(118, 75, 162, 0.8)',
                        borderRadius: 4,
                    }},
                    {{
                        label: 'p99',
                        data: {latency_p99},
                        backgroundColor: 'rgba(239, 68, 68, 0.8)',
                        borderRadius: 4,
                    }}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{
                        labels: {{ color: '#e4e4e7' }}
                    }}
                }},
                scales: {{
                    x: {{
                        ticks: {{ color: '#a1a1aa' }},
                        grid: {{ color: 'rgba(255, 255, 255, 0.1)' }}
                    }},
                    y: {{
                        ticks: {{ color: '#a1a1aa' }},
                        grid: {{ color: 'rgba(255, 255, 255, 0.1)' }}
                    }}
                }}
            }}
        }});

        // Cost Chart
        const costCtx = document.getElementById('costChart').getContext('2d');
        new Chart(costCtx, {{
            type: 'bar',
            data: {{
                labels: {cost_labels},
                datasets: [
                    {{
                        label: 'FLR (Hot Path)',
                        data: {flr_data},
                        backgroundColor: 'rgba(16, 185, 129, 0.8)',
                        borderRadius: 4,
                    }},
                    {{
                        label: 'CLST (Cold Path)',
                        data: {clst_data},
                        backgroundColor: 'rgba(234, 179, 8, 0.8)',
                        borderRadius: 4,
                    }}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{
                        labels: {{ color: '#e4e4e7' }}
                    }}
                }},
                scales: {{
                    x: {{
                        ticks: {{ color: '#a1a1aa' }},
                        grid: {{ color: 'rgba(255, 255, 255, 0.1)' }}
                    }},
                    y: {{
                        ticks: {{ color: '#a1a1aa' }},
                        grid: {{ color: 'rgba(255, 255, 255, 0.1)' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""


def main():
    """CLI entry point."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m mindcore.benchmarks.dashboard <results.json> [output.html]")
        sys.exit(1)

    results_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "dashboard.html"

    output = generate_dashboard(results_file, output_file)
    print(f"Dashboard generated: {output}")


if __name__ == "__main__":
    main()
