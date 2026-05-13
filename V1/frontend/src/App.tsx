// App.tsx — Main application layout
import React, { useEffect, useState } from "react";
import { useBenchmarkSocket } from "./hooks/useBenchmarkSocket";
import { BenchmarkStatus } from "./components/BenchmarkStatus";
import { LiveInferenceViewer } from "./components/LiveInferenceViewer";
import { AccuracyChart } from "./components/AccuracyChart";
import { ModelList } from "./components/ModelList";
import { ResultsTable } from "./components/ResultsTable";

const API_BASE = "http://localhost:8000";

export default function App() {
  const { state, startBenchmark, stopBenchmark, reconnect } = useBenchmarkSocket();
  const [results, setResults] = useState<any>(null);

  // Fetch results when done
  useEffect(() => {
    if (state.isDone) {
      fetch(`${API_BASE}/benchmark/results`)
        .then((r) => r.json())
        .then(setResults)
        .catch(() => { });
    }
  }, [state.isDone]);

  return (
    <div className="app">
      {/* ── Top Nav ── */}
      <header className="top-nav">
        <div className="nav-logo">
          <span className="logo-icon">⚡</span>
          <span className="logo-text">Captcha Benchmark</span>
        </div>
        <div className="nav-right">
          <span
            className="conn-dot"
            data-status={state.status}
            title={`WebSocket: ${state.status}`}
          />
          <button className="btn-ghost" onClick={reconnect}>
            ↺ Reconnect
          </button>
        </div>
      </header>

      {/* ── Main Layout ── */}
      <div className="layout">
        {/* Sidebar */}
        <aside className="sidebar">
          <ModelList />
          <BenchmarkStatus
            state={state}
            onStart={startBenchmark}
            onStop={stopBenchmark}
          />
        </aside>

        {/* Main panel */}
        <main className="main-panel">
          <LiveInferenceViewer state={state} />
          <AccuracyChart accuracy={state.accuracy} />
          {state.isDone && results && <ResultsTable results={results} />}
        </main>
      </div>
    </div>
  );
}
