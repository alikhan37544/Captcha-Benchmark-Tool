// ResultsTable.tsx — Sortable results table shown after benchmark completes
import React, { useMemo, useState } from "react";

interface Prediction {
    id: string;
    file: string;
    expected: string;
    predicted: string;
    correct: boolean;
}

interface ModelResult {
    accuracy: number;
    correct: number;
    total: number;
    duration_s: number;
    predictions: Prediction[];
}

interface Results {
    run_id: string;
    dataset: { total: number; labels_file: string };
    models: Record<string, ModelResult>;
}

interface Props {
    results: Results | null;
}

type SortKey = "id" | "expected" | "predicted" | "correct";

export function ResultsTable({ results }: Props) {
    const [selectedModel, setSelectedModel] = useState<string>("");
    const [sortKey, setSortKey] = useState<SortKey>("id");
    const [sortAsc, setSortAsc] = useState(true);
    const [filter, setFilter] = useState<"all" | "correct" | "wrong">("all");

    const modelNames = results ? Object.keys(results.models) : [];

    // Default selected model when results arrive
    const activeModel = selectedModel || modelNames[0] || "";
    const modelData = results?.models[activeModel];

    const rows = useMemo(() => {
        if (!modelData) return [];
        let data = [...modelData.predictions];
        if (filter === "correct") data = data.filter((p) => p.correct);
        if (filter === "wrong") data = data.filter((p) => !p.correct);
        data.sort((a, b) => {
            let va: string | boolean = a[sortKey];
            let vb: string | boolean = b[sortKey];
            if (typeof va === "boolean") va = va ? "1" : "0";
            if (typeof vb === "boolean") vb = vb ? "1" : "0";
            return sortAsc
                ? String(va).localeCompare(String(vb))
                : String(vb).localeCompare(String(va));
        });
        return data;
    }, [modelData, sortKey, sortAsc, filter]);

    const toggleSort = (k: SortKey) => {
        if (k === sortKey) setSortAsc((v) => !v);
        else { setSortKey(k); setSortAsc(true); }
    };

    if (!results) return null;

    return (
        <div className="results-card">
            <h2>Results — Run <code>{results.run_id}</code></h2>

            {/* Summary row */}
            <div className="summary-row">
                {modelNames.map((name) => {
                    const md = results.models[name];
                    return (
                        <button
                            key={name}
                            className={`model-tab ${activeModel === name ? "active" : ""}`}
                            onClick={() => setSelectedModel(name)}
                        >
                            {name}
                            <span className="acc-pill" style={{ background: accuracyColor(md.accuracy) }}>
                                {(md.accuracy * 100).toFixed(1)}%
                            </span>
                        </button>
                    );
                })}
            </div>

            {/* Filter */}
            <div className="filter-row">
                {(["all", "correct", "wrong"] as const).map((f) => (
                    <button
                        key={f}
                        className={`filter-btn ${filter === f ? "active" : ""}`}
                        onClick={() => setFilter(f)}
                    >
                        {f === "all" ? `All (${modelData?.total ?? 0})` : f === "correct" ? `✓ Correct (${modelData?.correct ?? 0})` : `✗ Wrong (${(modelData?.total ?? 0) - (modelData?.correct ?? 0)})`}
                    </button>
                ))}
            </div>

            {/* Table */}
            <div className="table-wrap">
                <table className="results-table">
                    <thead>
                        <tr>
                            {([["id", "ID"], ["expected", "Expected"], ["predicted", "Predicted"], ["correct", "Result"]] as [SortKey, string][]).map(([k, label]) => (
                                <th key={k} onClick={() => toggleSort(k)} className="sortable">
                                    {label} {sortKey === k ? (sortAsc ? "↑" : "↓") : ""}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {rows.map((row) => (
                            <tr key={row.id} className={row.correct ? "row-correct" : "row-wrong"}>
                                <td className="mono">{row.id}</td>
                                <td className="mono">{row.expected}</td>
                                <td className={`mono ${row.correct ? "text-correct" : "text-wrong"}`}>{row.predicted}</td>
                                <td>{row.correct ? <span className="badge-correct">✓</span> : <span className="badge-wrong">✗</span>}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

function accuracyColor(acc: number): string {
    if (acc >= 0.8) return "#16a34a";
    if (acc >= 0.5) return "#ca8a04";
    return "#dc2626";
}
