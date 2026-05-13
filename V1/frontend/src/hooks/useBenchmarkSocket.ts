// useBenchmarkSocket.ts — Custom hook for WebSocket connection to the benchmark backend
import { useCallback, useEffect, useRef, useState } from "react";

export type ConnectionStatus = "connecting" | "connected" | "disconnected" | "error";

export interface ProgressEvent {
    event: "progress";
    model: string;
    model_index: number;
    total_models: number;
    image: string;
    id: string;
    predicted: string;
    expected: string;
    correct: boolean;
    model_progress: number;
    overall_progress: number;
    accuracy: Record<string, number>;
    eta_s: number | null;
}

export interface BenchmarkState {
    status: ConnectionStatus;
    currentModel: string;
    modelIndex: number;
    totalModels: number;
    currentImage: string;
    currentId: string;
    predicted: string;
    expected: string;
    correct: boolean | null;
    modelProgress: number;
    overallProgress: number;
    accuracy: Record<string, number>;
    etaSeconds: number | null;
    isRunning: boolean;
    isDone: boolean;
    lastEvent: string;
    modelBeingLoaded: string;
    loadingModel: boolean;
}

const DEFAULT_STATE: BenchmarkState = {
    status: "disconnected",
    currentModel: "",
    modelIndex: 0,
    totalModels: 0,
    currentImage: "",
    currentId: "",
    predicted: "",
    expected: "",
    correct: null,
    modelProgress: 0,
    overallProgress: 0,
    accuracy: {},
    etaSeconds: null,
    isRunning: false,
    isDone: false,
    lastEvent: "",
    modelBeingLoaded: "",
    loadingModel: false,
};

const WS_URL = "ws://localhost:8000/ws/progress";
const API_BASE = "http://localhost:8000";

export function useBenchmarkSocket() {
    const [state, setState] = useState<BenchmarkState>(DEFAULT_STATE);
    const wsRef = useRef<WebSocket | null>(null);

    const connect = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) return;

        setState((s) => ({ ...s, status: "connecting" }));
        const ws = new WebSocket(WS_URL);
        wsRef.current = ws;

        ws.onopen = () => setState((s) => ({ ...s, status: "connected" }));
        ws.onclose = () => setState((s) => ({ ...s, status: "disconnected", isRunning: false }));
        ws.onerror = () => setState((s) => ({ ...s, status: "error" }));

        ws.onmessage = (e) => {
            try {
                const data = JSON.parse(e.data);

                if (data.event === "heartbeat") return;

                if (data.event === "progress") {
                    setState((s) => ({
                        ...s,
                        isRunning: true,
                        isDone: false,
                        loadingModel: false,
                        lastEvent: "progress",
                        currentModel: data.model,
                        modelIndex: data.model_index,
                        totalModels: data.total_models,
                        currentImage: data.image,
                        currentId: data.id,
                        predicted: data.predicted,
                        expected: data.expected,
                        correct: data.correct,
                        modelProgress: data.model_progress,
                        overallProgress: data.overall_progress,
                        accuracy: data.accuracy ?? s.accuracy,
                        etaSeconds: data.eta_s ?? null,
                    }));
                } else if (data.event === "model_loading") {
                    setState((s) => ({
                        ...s,
                        loadingModel: true,
                        modelBeingLoaded: data.model,
                        lastEvent: "model_loading",
                    }));
                } else if (data.event === "model_loaded") {
                    setState((s) => ({
                        ...s,
                        loadingModel: false,
                        lastEvent: "model_loaded",
                    }));
                } else if (data.event === "model_done") {
                    setState((s) => ({
                        ...s,
                        accuracy: { ...s.accuracy, [data.model]: data.accuracy },
                        lastEvent: "model_done",
                    }));
                } else if (data.event === "done") {
                    setState((s) => ({
                        ...s,
                        isRunning: false,
                        isDone: true,
                        overallProgress: 1,
                        modelProgress: 1,
                        lastEvent: "done",
                        accuracy: data.accuracy ?? s.accuracy,
                    }));
                } else if (data.event === "cancelled") {
                    setState((s) => ({ ...s, isRunning: false, lastEvent: "cancelled" }));
                } else if (data.event === "error") {
                    setState((s) => ({ ...s, isRunning: false, lastEvent: "error" }));
                }
            } catch {
                // ignore malformed frames
            }
        };
    }, []);

    useEffect(() => {
        connect();
        return () => wsRef.current?.close();
    }, [connect]);

    const startBenchmark = useCallback(async () => {
        setState((s) => ({ ...s, isRunning: false, isDone: false, accuracy: {}, overallProgress: 0, modelProgress: 0, lastEvent: "" }));
        await fetch(`${API_BASE}/benchmark/start`, { method: "POST" });
        if (wsRef.current?.readyState !== WebSocket.OPEN) connect();
    }, [connect]);

    const stopBenchmark = useCallback(async () => {
        await fetch(`${API_BASE}/benchmark/stop`, { method: "POST" });
    }, []);

    return { state, startBenchmark, stopBenchmark, reconnect: connect };
}
