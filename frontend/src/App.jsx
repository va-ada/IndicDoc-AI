import React, { useState, useEffect } from 'react';
import './App.css';
import { formatMetric } from './metrics.js';

// Override with VITE_API_URL in frontend/.env.local when the backend runs elsewhere
const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

function App() {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    const loadMetrics = async () => {
      try {
        const res = await fetch(`${API_URL}/metrics`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        setMetrics(await res.json());
      } catch (err) {
        console.error('Could not load metrics from the API:', err);
      }
    };
    loadMetrics();
  }, []);

  // Show the baseline: it is the only model with measured results so far
  const baseline = metrics?.baseline;

  return (
    <div className="app-container">
      <aside className="sidebar">
        <h2>IndicDoc AI</h2>
        <nav className="sidebar-nav">
          <a href="#" className="nav-link active">Dashboard</a>
          <a href="#" className="nav-link">Analyze</a>
          <a href="#" className="nav-link">Model Insights</a>
          <a href="#" className="nav-link">Evaluation</a>
        </nav>
      </aside>

      <main className="main-content">
        <section className="hero">
          <h1>Indic<span>Doc</span> AI</h1>
          <p>Understand Indian Documents with Deep Learning.</p>
        </section>

        <section className="dashboard">
          <h2>Baseline (Faster R-CNN)</h2>
          <div className="metrics-grid">
            <div className="metric-card">
              <h3>{formatMetric(baseline?.precision)}</h3>
              <p>Precision</p>
            </div>
            <div className="metric-card">
              <h3>{formatMetric(baseline?.recall)}</h3>
              <p>Recall</p>
            </div>
            <div className="metric-card">
              <h3>{formatMetric(baseline?.map50)}</h3>
              <p>mAP@50</p>
            </div>
            <div className="metric-card">
              <h3>{formatMetric(baseline?.inference_time, 's')}</h3>
              <p>Inference Time</p>
            </div>
          </div>
          <p className="metrics-note">
            {baseline ? baseline.note : 'Not evaluated yet, or the API is not running.'}
          </p>
        </section>
      </main>
    </div>
  );
}

export default App;
