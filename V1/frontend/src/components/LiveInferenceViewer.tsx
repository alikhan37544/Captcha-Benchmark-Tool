// LiveInferenceViewer.tsx — Shows current captcha image and prediction in real-time
import React, { useEffect, useRef, useState } from "react";
import type { BenchmarkState } from "../hooks/useBenchmarkSocket";

const API_BASE = "http://localhost:8000";

interface Props {
    state: BenchmarkState;
}

export function LiveInferenceViewer({ state }: Props) {
    const [flash, setFlash] = useState<"correct" | "wrong" | null>(null);
    const prevIdRef = useRef<string>("");

    useEffect(() => {
        if (state.currentId && state.currentId !== prevIdRef.current) {
            prevIdRef.current = state.currentId;
            setFlash(state.correct ? "correct" : "wrong");
            const t = setTimeout(() => setFlash(null), 600);
            return () => clearTimeout(t);
        }
    }, [state.currentId, state.correct]);

    const imgSrc = state.currentImage
        ? `${API_BASE}/captchas/${state.currentImage}`
        : null;

    return (
        <div className={`inference-card ${flash ? `flash-${flash}` : ""}`}>
            <h2>Live Inference</h2>

            <div className="captcha-frame">
                {imgSrc ? (
                    <img
                        key={state.currentImage}
                        src={imgSrc}
                        alt={`Captcha ${state.currentId}`}
                        className="captcha-img"
                    />
                ) : (
                    <div className="captcha-placeholder">
                        <span>Waiting for benchmark to start…</span>
                    </div>
                )}
            </div>

            <div className="prediction-row">
                <div className="pred-box expected">
                    <span className="pred-label">Expected</span>
                    <span className="pred-value">{state.expected || "—"}</span>
                </div>
                <div className={`pred-box predicted ${state.correct === null ? "" : state.correct ? "correct" : "wrong"}`}>
                    <span className="pred-label">Predicted</span>
                    <span className="pred-value">{state.predicted || "—"}</span>
                </div>
                <div className="pred-box verdict">
                    <span className="pred-label">Result</span>
                    <span className={`verdict-icon ${state.correct === null ? "" : state.correct ? "correct" : "wrong"}`}>
                        {state.correct === null ? "—" : state.correct ? "✓" : "✗"}
                    </span>
                </div>
            </div>

            {state.currentModel && (
                <div className="model-tag">
                    Model: <strong>{state.currentModel}</strong>
                </div>
            )}
        </div>
    );
}
