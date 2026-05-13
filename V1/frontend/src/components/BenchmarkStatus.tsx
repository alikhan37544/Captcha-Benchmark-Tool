// BenchmarkStatus.tsx — Progress bars, ETA, Start/Stop controls
import React from "react";
import type { BenchmarkState } from "../hooks/useBenchmarkSocket";

interface Props {
    state: BenchmarkState;
    onStart: () => void;
    onStop: () => void;
}

function formatEta(seconds: number | null): string {
    if (seconds === null || seconds < 0) return "—";
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return `${m}m ${s}s`;
}

export function BenchmarkStatus({ state, onStart, onStop }: Props) {
    const overallPct = Math.round(state.overallProgress * 100);
    const modelPct = Math.round(state.modelProgress * 100);

    return (
        <div className="status-card">
            <div className="status-header">
                <h2>Benchmark Status</h2>
                <div className="status-badge" data-status={state.isRunning ? "running" : state.isDone ? "done" : "idle"}>
                    {state.isRunning ? "● Running" : state.isDone ? "✓ Done" : "Idle"}
                </div>
            </div>

            {state.loadingModel && (
                <div className="loading-notice">
                    <div className="spinner" />
                    Loading model: <strong>{state.modelBeingLoaded}</strong>
                </div>
            )}

            <div className="progress-section">
                <div className="progress-label">
                    <span>Overall</span>
                    <span>{overallPct}%</span>
                </div>
                <div className="progress-track">
                    <div className="progress-fill overall" style={{ width: `${overallPct}%` }} />
                </div>
            </div>

            {state.currentModel && (
                <div className="progress-section">
                    <div className="progress-label">
                        <span>
                            {state.currentModel}{" "}
                            <span className="model-counter">
                                ({state.modelIndex + 1}/{state.totalModels})
                            </span>
                        </span>
                        <span>{modelPct}%</span>
                    </div>
                    <div className="progress-track">
                        <div className="progress-fill model" style={{ width: `${modelPct}%` }} />
                    </div>
                </div>
            )}

            <div className="stats-row">
                <div className="stat">
                    <span className="stat-label">ETA</span>
                    <span className="stat-value">{formatEta(state.etaSeconds)}</span>
                </div>
                <div className="stat">
                    <span className="stat-label">Image</span>
                    <span className="stat-value">{state.currentId || "—"}</span>
                </div>
                <div className="stat">
                    <span className="stat-label">Connection</span>
                    <span className="stat-value" data-conn={state.status}>{state.status}</span>
                </div>
            </div>

            <div className="controls">
                <button
                    className="btn btn-start"
                    onClick={onStart}
                    disabled={state.isRunning}
                >
                    ▶ Start Benchmark
                </button>
                <button
                    className="btn btn-stop"
                    onClick={onStop}
                    disabled={!state.isRunning}
                >
                    ■ Stop
                </button>
            </div>
        </div>
    );
}
