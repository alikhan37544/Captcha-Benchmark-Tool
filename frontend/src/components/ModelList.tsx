// ModelList.tsx — Displays configured models and their enabled state
import React, { useEffect, useState } from "react";

interface ModelInfo {
    name: string;
    type: string;
    enabled: boolean;
    model_id: string | null;
}

const API_BASE = "http://localhost:8000";

export function ModelList() {
    const [models, setModels] = useState<ModelInfo[]>([]);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        fetch(`${API_BASE}/benchmark/models`)
            .then((r) => r.json())
            .then(setModels)
            .catch(() => setError("Could not reach backend"));
    }, []);

    const TYPE_COLORS: Record<string, string> = {
        easyocr: "#22c55e",
        tesseract: "#14b8a6",
        huggingface: "#f59e0b",
        lmstudio: "#a78bfa",
        openai_compat: "#6366f1",
        custom: "#f97316",
    };

    return (
        <div className="model-list-card">
            <h2>Models</h2>
            {error ? (
                <p className="error-msg">{error}</p>
            ) : (
                <ul className="model-list">
                    {models.map((m) => (
                        <li key={m.name} className={`model-item ${m.enabled ? "enabled" : "disabled"}`}>
                            <span className="model-dot" style={{ background: m.enabled ? "#22c55e" : "#475569" }} />
                            <div className="model-info">
                                <span className="model-name">{m.name}</span>
                                <span
                                    className="model-type-badge"
                                    style={{ background: `${TYPE_COLORS[m.type] ?? "#64748b"}22`, color: TYPE_COLORS[m.type] ?? "#94a3b8", borderColor: `${TYPE_COLORS[m.type] ?? "#64748b"}55` }}
                                >
                                    {m.type}
                                </span>
                            </div>
                            {!m.enabled && <span className="disabled-tag">disabled</span>}
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
}
