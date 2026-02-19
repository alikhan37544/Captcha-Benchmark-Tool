// AccuracyChart.tsx — Real-time horizontal bar chart of per-model accuracy
import React from "react";
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Cell,
    LabelList,
} from "recharts";

interface Props {
    accuracy: Record<string, number>;
}

const MODEL_COLORS = [
    "#6366f1", "#8b5cf6", "#a855f7", "#ec4899",
    "#f43f5e", "#f97316", "#eab308", "#22c55e",
    "#14b8a6", "#06b6d4", "#3b82f6", "#64748b",
];

export function AccuracyChart({ accuracy }: Props) {
    const data = Object.entries(accuracy).map(([name, acc], i) => ({
        name,
        accuracy: Math.round(acc * 10000) / 100, // to percentage with 2 decimals
        color: MODEL_COLORS[i % MODEL_COLORS.length],
    }));

    if (data.length === 0) {
        return (
            <div className="chart-card empty">
                <h2>Accuracy per Model</h2>
                <p className="empty-msg">Accuracy will appear here once the benchmark starts.</p>
            </div>
        );
    }

    return (
        <div className="chart-card">
            <h2>Accuracy per Model</h2>
            <ResponsiveContainer width="100%" height={Math.max(200, data.length * 52)}>
                <BarChart
                    layout="vertical"
                    data={data}
                    margin={{ top: 8, right: 64, left: 8, bottom: 8 }}
                >
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.07)" horizontal={false} />
                    <XAxis
                        type="number"
                        domain={[0, 100]}
                        tickFormatter={(v) => `${v}%`}
                        tick={{ fill: "#94a3b8", fontSize: 12 }}
                        axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                        tickLine={false}
                    />
                    <YAxis
                        type="category"
                        dataKey="name"
                        width={160}
                        tick={{ fill: "#e2e8f0", fontSize: 12 }}
                        axisLine={false}
                        tickLine={false}
                    />
                    <Tooltip
                        cursor={{ fill: "rgba(255,255,255,0.04)" }}
                        contentStyle={{
                            background: "#1e293b",
                            border: "1px solid rgba(255,255,255,0.1)",
                            borderRadius: 8,
                            color: "#f1f5f9",
                        }}
                        formatter={(value: number) => [`${value}%`, "Accuracy"]}
                    />
                    <Bar dataKey="accuracy" radius={[0, 4, 4, 0]} maxBarSize={28}>
                        {data.map((entry, i) => (
                            <Cell key={entry.name} fill={entry.color} />
                        ))}
                        <LabelList
                            dataKey="accuracy"
                            position="right"
                            formatter={(v: number) => `${v}%`}
                            style={{ fill: "#94a3b8", fontSize: 12 }}
                        />
                    </Bar>
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
}
