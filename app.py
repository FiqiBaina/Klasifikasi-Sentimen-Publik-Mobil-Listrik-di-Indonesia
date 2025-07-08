from flask import Flask, request, render_template_string, redirect, url_for
from flask import send_file
from flask import request, render_template_string
from sklearn.metrics import classification_report, accuracy_score
import joblib
import pandas as pd
import math
import numpy as np

model = joblib.load("model_sentiment.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder_sentiment.pkl")

app = Flask(__name__)
dataset = pd.read_csv("data_cleaned.csv", on_bad_lines='skip')

# Halaman Klasifikasi
HTML_FORM = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <title>Klasifikasi Sentimen Mobil Listrik</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Poppins', sans-serif; background-color: #1B1F27; margin: 0; padding: 0; }
        .sidebar { height: 100vh; width: 280px; position: fixed; background-color: #11141A; padding: 30px 20px; color: white; }
        .sidebar h2 { color: #F3B27C; margin-bottom: 30px;}
        .sidebar a { display: block; margin-bottom: 15px; text-decoration: none; font-weight: 500; }
        .sidebar a:hover { color: #F3B27C; }
        .content { margin-left: 280px; padding: 40px 20px; }
        .form-box { max-width: 600px; margin: auto; padding: 30px; border-radius: 10px; background-color: rgba(255, 255, 255, 0.5)}
        textarea::placeholder {font-style: italic;}
        .terbuka { color: #F3B27C; }
        .tidakterbuka { color: #fff; text-decoration: none; }
    </style>
</head>
<body>
    <div class="sidebar shadow">
        <h2>Sentimen App</h2>
        <a class="terbuka" href="/"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-input-cursor-text" viewBox="0 0 16 16">
        <path fill-rule="evenodd" d="M5 2a.5.5 0 0 1 .5-.5c.862 0 1.573.287 2.06.566.174.099.321.198.44.286.119-.088.266-.187.44-.286A4.17 4.17 0 0 1 10.5 1.5a.5.5 0 0 1 0 1c-.638 0-1.177.213-1.564.434a3.5 3.5 0 0 0-.436.294V7.5H9a.5.5 0 0 1 0 1h-.5v4.272c.1.08.248.187.436.294.387.221.926.434 1.564.434a.5.5 0 0 1 0 1 4.17 4.17 0 0 1-2.06-.566A5 5 0 0 1 8 13.65a5 5 0 0 1-.44.285 4.17 4.17 0 0 1-2.06.566.5.5 0 0 1 0-1c.638 0 1.177-.213 1.564-.434.188-.107.335-.214.436-.294V8.5H7a.5.5 0 0 1 0-1h.5V3.228a3.5 3.5 0 0 0-.436-.294A3.17 3.17 0 0 0 5.5 2.5.5.5 0 0 1 5 2"/>
        <path d="M10 5h4a1 1 0 0 1 1 1v4a1 1 0 0 1-1 1h-4v1h4a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2h-4zM6 5V4H2a2 2 0 0 0-2 2v4a2 2 0 0 0 2 2h4v-1H2a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1z"/>
        </svg>
        <span class="ms-2">Klasifikasi</span></a>
        <a class="tidakterbuka" href="/tabel"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-clipboard2-data" viewBox="0 0 16 16">
        <path d="M9.5 0a.5.5 0 0 1 .5.5.5.5 0 0 0 .5.5.5.5 0 0 1 .5.5V2a.5.5 0 0 1-.5.5h-5A.5.5 0 0 1 5 2v-.5a.5.5 0 0 1 .5-.5.5.5 0 0 0 .5-.5.5.5 0 0 1 .5-.5z"/>
        <path d="M3 2.5a.5.5 0 0 1 .5-.5H4a.5.5 0 0 0 0-1h-.5A1.5 1.5 0 0 0 2 2.5v12A1.5 1.5 0 0 0 3.5 16h9a1.5 1.5 0 0 0 1.5-1.5v-12A1.5 1.5 0 0 0 12.5 1H12a.5.5 0 0 0 0 1h.5a.5.5 0 0 1 .5.5v12a.5.5 0 0 1-.5.5h-9a.5.5 0 0 1-.5-.5z"/>
        <path d="M10 7a1 1 0 1 1 2 0v5a1 1 0 1 1-2 0zm-6 4a1 1 0 1 1 2 0v1a1 1 0 1 1-2 0zm4-3a1 1 0 0 0-1 1v3a1 1 0 1 0 2 0V9a1 1 0 0 0-1-1"/>
        </svg>
        <span class="ms-2">Data Komentar</span></a>
        <a class="tidakterbuka" href="/evaluasi"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-bar-chart-steps" viewBox="0 0 16 16">
        <path d="M.5 0a.5.5 0 0 1 .5.5v15a.5.5 0 0 1-1 0V.5A.5.5 0 0 1 .5 0M2 1.5a.5.5 0 0 1 .5-.5h4a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-4a.5.5 0 0 1-.5-.5zm2 4a.5.5 0 0 1 .5-.5h7a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-7a.5.5 0 0 1-.5-.5zm2 4a.5.5 0 0 1 .5-.5h6a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-6a.5.5 0 0 1-.5-.5zm2 4a.5.5 0 0 1 .5-.5h7a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-7a.5.5 0 0 1-.5-.5z"/>
        </svg>
        <span class="ms-2">Evaluasi Model</span></a>
    </div>

    <div class="content">
        <h1 class="text-white text-center mb-4"><strong><span style="color: #F3B27C;">Klasifikasi</span> Sentimen Mobil Listrik</strong></h1>

        <div class="form-box shadow">
            <form method="POST">
                <div class="mb-3">
                    <label for="komentar" class="form-label fw-bold">Komentar Anda:</label>
                    <textarea class="form-control" name="komentar" rows="4" required placeholder="Masukkan komentar anda..."></textarea>
                </div>
                <div class="d-grid">
                    <button type="submit" class="btn btn-warning" style="color: white; font-weight: 600;">Klasifikasi</button>
                </div>
            </form>
            {% if result %}
                <div class="alert alert-info mt-4">{{ result }}</div>
            {% endif %}
            
            {% if alert %}
                <div class="alert alert-success mt-2">{{ alert }}</div>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

# Halaman Tabel + Statistik (Gabungan)
HTML_TABLE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <title>Data Tabel & Statistik Sentimen</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Poppins', sans-serif; background-color: #1B1F27; margin: 0; padding: 0; }
        .sidebar { height: 100vh; width: 280px; position: fixed; background-color: #11141A; padding: 30px 20px; color: white; }
        .sidebar h2 { color: #F3B27C; margin-bottom: 30px; }
        .sidebar a { display: block; margin-bottom: 15px; text-decoration: none; font-weight: 500; }
        .sidebar a:hover { color: #F3B27C; }
        .content { margin-left: 280px; padding: 40px 20px; color: white; }
        .chart-box { width: 600px; }
        .table-box, .chart-box { background-color: #ffffff10; padding: 20px; border-radius: 10px; }
        .pagination {padding: 10px 15px; border-radius: 10px;}
        .page-link {background-color: #111111; color: white; border: 1px solid #333333;}
        .page-link:hover {background-color: #F3B27C; color: black; border-color: #F3B27C;}
        .page-item.active .page-link {background-color: #F3B27C; color: black; border-color: #F3B27C;}
        .page-item.disabled .page-link {background-color: #333333; color: #999999; border-color: #333333;}
        .terbuka { color: #F3B27C; }
        .tidakterbuka { color: #fff; text-decoration: none; }
    </style>
</head>
<body>
    <div class="sidebar">
        <h2>Sentimen App</h2>
        <a class="tidakterbuka" href="/"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-input-cursor-text" viewBox="0 0 16 16">
        <path fill-rule="evenodd" d="M5 2a.5.5 0 0 1 .5-.5c.862 0 1.573.287 2.06.566.174.099.321.198.44.286.119-.088.266-.187.44-.286A4.17 4.17 0 0 1 10.5 1.5a.5.5 0 0 1 0 1c-.638 0-1.177.213-1.564.434a3.5 3.5 0 0 0-.436.294V7.5H9a.5.5 0 0 1 0 1h-.5v4.272c.1.08.248.187.436.294.387.221.926.434 1.564.434a.5.5 0 0 1 0 1 4.17 4.17 0 0 1-2.06-.566A5 5 0 0 1 8 13.65a5 5 0 0 1-.44.285 4.17 4.17 0 0 1-2.06.566.5.5 0 0 1 0-1c.638 0 1.177-.213 1.564-.434.188-.107.335-.214.436-.294V8.5H7a.5.5 0 0 1 0-1h.5V3.228a3.5 3.5 0 0 0-.436-.294A3.17 3.17 0 0 0 5.5 2.5.5.5 0 0 1 5 2"/>
        <path d="M10 5h4a1 1 0 0 1 1 1v4a1 1 0 0 1-1 1h-4v1h4a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2h-4zM6 5V4H2a2 2 0 0 0-2 2v4a2 2 0 0 0 2 2h4v-1H2a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1z"/>
        </svg>
        <span class="ms-2">Klasifikasi</span></a>
        <a class="terbuka" href="/tabel"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-clipboard2-data" viewBox="0 0 16 16">
        <path d="M9.5 0a.5.5 0 0 1 .5.5.5.5 0 0 0 .5.5.5.5 0 0 1 .5.5V2a.5.5 0 0 1-.5.5h-5A.5.5 0 0 1 5 2v-.5a.5.5 0 0 1 .5-.5.5.5 0 0 0 .5-.5.5.5 0 0 1 .5-.5z"/>
        <path d="M3 2.5a.5.5 0 0 1 .5-.5H4a.5.5 0 0 0 0-1h-.5A1.5 1.5 0 0 0 2 2.5v12A1.5 1.5 0 0 0 3.5 16h9a1.5 1.5 0 0 0 1.5-1.5v-12A1.5 1.5 0 0 0 12.5 1H12a.5.5 0 0 0 0 1h.5a.5.5 0 0 1 .5.5v12a.5.5 0 0 1-.5.5h-9a.5.5 0 0 1-.5-.5z"/>
        <path d="M10 7a1 1 0 1 1 2 0v5a1 1 0 1 1-2 0zm-6 4a1 1 0 1 1 2 0v1a1 1 0 1 1-2 0zm4-3a1 1 0 0 0-1 1v3a1 1 0 1 0 2 0V9a1 1 0 0 0-1-1"/>
        </svg>
        <span class="ms-2">Data Komentar</span></a>
        <a class="tidakterbuka" href="/evaluasi"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-bar-chart-steps" viewBox="0 0 16 16">
        <path d="M.5 0a.5.5 0 0 1 .5.5v15a.5.5 0 0 1-1 0V.5A.5.5 0 0 1 .5 0M2 1.5a.5.5 0 0 1 .5-.5h4a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-4a.5.5 0 0 1-.5-.5zm2 4a.5.5 0 0 1 .5-.5h7a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-7a.5.5 0 0 1-.5-.5zm2 4a.5.5 0 0 1 .5-.5h6a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-6a.5.5 0 0 1-.5-.5zm2 4a.5.5 0 0 1 .5-.5h7a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-7a.5.5 0 0 1-.5-.5z"/>
        </svg>
        <span class="ms-2">Evaluasi Model</span></a>
    </div>

    <div class="content">
        <h2 class="mb-4">Statistik Sentimen Komentar</h2>
        <div class="chart-box mb-5">
            <canvas id="sentimentChart"></canvas>
        </div>
        
        <h2 class="mb-0">Data Komentar (Halaman {{ page }} dari {{ pages }})</h2>
        
        <div class="d-flex justify-content-between align-items-center my-3">
            <div class="mb-3">
                <a href="{{ url_for('unduh_dataset') }}" class="btn btn-sm btn-success fw-semibold">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-download" viewBox="0 0 16 16">
                        <path d="M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5"/>
                        <path d="M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708z"/>
                    </svg>
                    <span class="ms-2">Unduh Dataset</span>
                </a>
            </div>
            <form method="get" action="{{ url_for('tabel') }}">
                <div class="input-group">
                    <select class="form-select" name="filter">
                        <option value="">Semua Sentimen</option>
                        <option value="positif" {% if filter == 'positif' %}selected{% endif %}>Positif</option>
                        <option value="negatif" {% if filter == 'negatif' %}selected{% endif %}>Negatif</option>
                        <option value="netral" {% if filter == 'netral' %}selected{% endif %}>Netral</option>
                    </select>
                    <button class="btn btn-warning text-white fw-semibold" type="submit">Filter</button>
                </div>
            </form>
        </div>

        <div class="table-box">
            <div class="table-responsive">
                <table class="table table-striped">
                    <thead class="table-dark">
                        <tr>
                            {% for key in data[0].keys() %}
                                {% if key != 'id_komentar' %}
                                    <th>{{ key|capitalize }}</th>
                                {% endif %}
                            {% endfor %}
                        </tr>
                    </thead>
                    <tbody class="table-dark">
                        {% for row in data %}
                        <tr>
                            {% for key, value in row.items() %}
                                {% if key != 'id_komentar' %}
                                    <td>{{ value }}</td>
                                {% endif %}
                            {% endfor %}
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Pagination -->
        <nav class="mt-4" aria-label="...">
            <ul class="pagination justify-content-center">
                <li class="page-item {% if page <= 1 %}disabled{% endif %}">
                    <a class="page-link" href="{{ url_for('tabel', page=page-1) }}">Previous</a>
                </li>
                {% set start = page - 1 if page > 1 else 1 %}
                {% set end = start + 2 %}
                {% if end > pages %}
                    {% set end = pages %}
                    {% set start = end - 2 if end - 2 > 0 else 1 %}
                {% endif %}
                {% for p in range(start, end + 1) %}
                    <li class="page-item {% if p == page %}active{% endif %}">
                        <a class="page-link" href="{{ url_for('tabel', page=p) }}">{{ p }}</a>
                    </li>
                {% endfor %}
                <li class="page-item {% if page >= pages %}disabled{% endif %}">
                    <a class="page-link" href="{{ url_for('tabel', page=page+1) }}">Next</a>
                </li>
            </ul>
        </nav>
    </div>

    <script>
        const ctx = document.getElementById('sentimentChart').getContext('2d');
        const sentimentChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Positif', 'Negatif', 'Netral'],
                datasets: [{
                    label: 'Jumlah Komentar',
                    data: [{{ positif }}, {{ negatif }}, {{ netral }}],
                    backgroundColor: function(context) {
                        const chart = context.chart;
                        const {ctx, chartArea} = chart;

                        if (!chartArea) return;

                        const createGradient = (color1, color2) => {
                            const gradient = ctx.createRadialGradient(
                                (chartArea.left + chartArea.right) / 2,
                                (chartArea.top + chartArea.bottom) / 2,
                                0,
                                (chartArea.left + chartArea.right) / 2,
                                (chartArea.top + chartArea.bottom) / 2,
                                chartArea.width
                            );
                            gradient.addColorStop(0, color1);
                            gradient.addColorStop(1, color2);
                            return gradient;
                        };

                        return [
                            createGradient('rgba(40, 167, 69, 0.5)', 'rgba(40, 167, 69, 0)'),     // green blur
                            createGradient('rgba(220, 53, 69, 0.5)', 'rgba(220, 53, 69, 0)'),     // red blur
                            createGradient('rgba(255, 193, 7, 0.5)', 'rgba(255, 193, 7, 0)')      // yellow blur
                        ];
                    },
                    borderColor: ['#28a745', '#dc3545', '#ffc107'],
                    borderColor: ['#218838', '#c82333', '#e0a800'],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { stepSize: 1 }
                    }
                }
            }
        });
    </script>
</body>
</html>
"""

HTML_EVALUASI = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <title>Evaluasi Model</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Poppins', sans-serif; background-color: #1B1F27; margin: 0; padding: 0; }
        .sidebar { height: 100vh; width: 280px; position: fixed; background-color: #11141A; padding: 30px 20px; color: white; }
        .sidebar h2 { color: #F3B27C; margin-bottom: 30px; }
        .sidebar a { display: block; margin-bottom: 15px; text-decoration: none; font-weight: 500; }
        .sidebar a:hover { color: #F3B27C; }
        .content { margin-left: 280px; padding: 40px 20px; }
        .alert { color: black; }
        pre { background-color: #2d2d2d; padding: 15px; border-radius: 10px; color: white; }
        .terbuka { color: #F3B27C; }
        .tidakterbuka { color: #fff; text-decoration: none; }
        input { max-width: 300px; }
    </style>
</head>
<body>
    <div class="sidebar">
        <h2>Sentimen App</h2>
        <a class="tidakterbuka" href="/"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-input-cursor-text" viewBox="0 0 16 16">
        <path fill-rule="evenodd" d="M5 2a.5.5 0 0 1 .5-.5c.862 0 1.573.287 2.06.566.174.099.321.198.44.286.119-.088.266-.187.44-.286A4.17 4.17 0 0 1 10.5 1.5a.5.5 0 0 1 0 1c-.638 0-1.177.213-1.564.434a3.5 3.5 0 0 0-.436.294V7.5H9a.5.5 0 0 1 0 1h-.5v4.272c.1.08.248.187.436.294.387.221.926.434 1.564.434a.5.5 0 0 1 0 1 4.17 4.17 0 0 1-2.06-.566A5 5 0 0 1 8 13.65a5 5 0 0 1-.44.285 4.17 4.17 0 0 1-2.06.566.5.5 0 0 1 0-1c.638 0 1.177-.213 1.564-.434.188-.107.335-.214.436-.294V8.5H7a.5.5 0 0 1 0-1h.5V3.228a3.5 3.5 0 0 0-.436-.294A3.17 3.17 0 0 0 5.5 2.5.5.5 0 0 1 5 2"/>
        <path d="M10 5h4a1 1 0 0 1 1 1v4a1 1 0 0 1-1 1h-4v1h4a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2h-4zM6 5V4H2a2 2 0 0 0-2 2v4a2 2 0 0 0 2 2h4v-1H2a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1z"/>
        </svg>
        <span class="ms-2">Klasifikasi</span></a>
        <a class="tidakterbuka" href="/tabel"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-clipboard2-data" viewBox="0 0 16 16">
        <path d="M9.5 0a.5.5 0 0 1 .5.5.5.5 0 0 0 .5.5.5.5 0 0 1 .5.5V2a.5.5 0 0 1-.5.5h-5A.5.5 0 0 1 5 2v-.5a.5.5 0 0 1 .5-.5.5.5 0 0 0 .5-.5.5.5 0 0 1 .5-.5z"/>
        <path d="M3 2.5a.5.5 0 0 1 .5-.5H4a.5.5 0 0 0 0-1h-.5A1.5 1.5 0 0 0 2 2.5v12A1.5 1.5 0 0 0 3.5 16h9a1.5 1.5 0 0 0 1.5-1.5v-12A1.5 1.5 0 0 0 12.5 1H12a.5.5 0 0 0 0 1h.5a.5.5 0 0 1 .5.5v12a.5.5 0 0 1-.5.5h-9a.5.5 0 0 1-.5-.5z"/>
        <path d="M10 7a1 1 0 1 1 2 0v5a1 1 0 1 1-2 0zm-6 4a1 1 0 1 1 2 0v1a1 1 0 1 1-2 0zm4-3a1 1 0 0 0-1 1v3a1 1 0 1 0 2 0V9a1 1 0 0 0-1-1"/>
        </svg>
        <span class="ms-2">Data Komentar</span></a>
        <a class="terbuka" href="/evaluasi"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-bar-chart-steps" viewBox="0 0 16 16">
        <path d="M.5 0a.5.5 0 0 1 .5.5v15a.5.5 0 0 1-1 0V.5A.5.5 0 0 1 .5 0M2 1.5a.5.5 0 0 1 .5-.5h4a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-4a.5.5 0 0 1-.5-.5zm2 4a.5.5 0 0 1 .5-.5h7a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-7a.5.5 0 0 1-.5-.5zm2 4a.5.5 0 0 1 .5-.5h6a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-6a.5.5 0 0 1-.5-.5zm2 4a.5.5 0 0 1 .5-.5h7a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-7a.5.5 0 0 1-.5-.5z"/>
        </svg>
        <span class="ms-2">Evaluasi Model</span></a>
    </div>
    <div class="content">
        <h2 class="text-white">Evaluasi Model Sentimen</h2>
        <form method="POST" enctype="multipart/form-data" class="mb-4">
            <div class="mb-3">
                <label for="dataset" class="form-label text-white">Unggah File CSV:</label>
                <input type="file" class="form-control" name="dataset" required accept=".csv">
            </div>
            <button type="submit" class="btn btn-warning text-white">Evaluasi</button>
        </form>

        {% if error %}
            <div class="alert alert-danger">{{ error }}</div>
        {% endif %}

        {% if accuracy %}
            <h4 class="text-white">Akurasi Model: <span class="text-info">{{ accuracy }}</span></h4>
        {% endif %}

        {% if report %}
            <h4 class="text-white mt-4">Laporan Klasifikasi</h4>
            <pre>{{ report }}</pre>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    alert = None
    if request.method == "POST":
        komentar = request.form["komentar"]
        komentar_vector = tfidf.transform([komentar])
        pred = model.predict(komentar_vector)
        label = le.inverse_transform(pred)[0]
        result = f"Hasil klasifikasi: {label.capitalize()}"

        new_data = pd.DataFrame([{
            "id_komentar": "-",
            "nama_akun": "-",
            "tanggal": pd.Timestamp.now(),
            "text_cleaning": komentar,
            "sentimen": label.lower()
        }])
        try:
            new_data.to_csv("data_cleaned.csv", mode='a', header=False, index=False)
            alert = "Komentar berhasil ditambahkan ke dataset!"
        except Exception as e:
            alert = f"Gagal menyimpan komentar: {e}"

    return render_template_string(HTML_FORM, result=result, alert=alert)

@app.route("/tabel")
def tabel():
    page = int(request.args.get("page", 1))
    per_page = 10
    filter_sentimen = request.args.get("filter", "")
    
    # Filter dataset
    if filter_sentimen in ["positif", "negatif", "netral"]:
        filtered_data = dataset[dataset["sentimen"] == filter_sentimen]
    else:
        filtered_data = dataset

    total = len(filtered_data)
    pages = math.ceil(total / per_page)
    start = (page - 1) * per_page
    end = start + per_page
    data_paginated = filtered_data.iloc[start:end].to_dict(orient="records")

    counts = dataset['sentimen'].value_counts().to_dict()
    positif = counts.get('positif', 0)
    negatif = counts.get('negatif', 0)
    netral = counts.get('netral', 0)

    return render_template_string(
        HTML_TABLE,
        data=data_paginated,
        page=page,
        pages=pages,
        positif=positif,
        negatif=negatif,
        netral=netral,
        filter=filter_sentimen
    )


@app.route("/unduh")
def unduh_dataset():
    try:
        return send_file("data_cleaned.csv", as_attachment=True)
    except Exception as e:
        return f"Gagal mengunduh dataset: {e}", 500
    
@app.route("/evaluasi", methods=["GET", "POST"])
def evaluasi():
    error = None
    report = None
    accuracy = None

    if request.method == "POST":
        file = request.files.get("dataset")
        if not file:
            error = "Tidak ada file yang diunggah."
            return render_template_string(HTML_EVALUASI, error=error)

        try:
            df = pd.read_csv(file)

            if not {"text_cleaning", "sentimen"}.issubset(df.columns):
                error = (
                    "Dataset harus memiliki kolom 'text_cleaning' dan 'sentimen'. "
                    f"Kolom ditemukan: {list(df.columns)}"
                )
                return render_template_string(HTML_EVALUASI, error=error)

            # Bersihkan NaN dan label tidak dikenal
            df = df.dropna(subset=["text_cleaning", "sentimen"])
            df["sentimen"] = df["sentimen"].astype(str).str.strip().str.lower()

            valid_labels = set(le.classes_)
            df = df[df["sentimen"].isin(valid_labels)]

            if df.empty:
                error = (
                    "Semua label tidak dikenali model. Label valid: "
                    + ", ".join(valid_labels)
                )
                return render_template_string(HTML_EVALUASI, error=error)

            X = tfidf.transform(df["text_cleaning"].astype(str))
            y_true = le.transform(df["sentimen"])
            y_pred = model.predict(X)

            accuracy = round(accuracy_score(y_true, y_pred), 4)
            report = classification_report(y_true, y_pred, target_names=le.classes_)

        except Exception as e:
            error = f"Terjadi kesalahan: {e}"

    return render_template_string(HTML_EVALUASI, error=error, report=report, accuracy=accuracy)
    
if __name__ == "__main__":
    app.run(debug=True)
