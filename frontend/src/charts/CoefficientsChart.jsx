import { useState } from "react";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Legend, ReferenceLine } from "recharts";

const MODEL_COLORS = {
  "Linear Regression (OLS)": "#2b7de9",
  "Ridge Regression (L2)": "#0e9aa7",
  "Lasso Regression (L1)": "#e8590c",
  "Elastic Net (L1+L2)": "#9b51e0",
  "Logistic Regression (No Penalty)": "#2b7de9",
  "Multinomial Logistic (No Penalty)": "#2b7de9",
  "Logistic Regression (L2 / Ridge)": "#0e9aa7",
  "Multinomial Logistic (L2 / Ridge)": "#0e9aa7",
  "Logistic Regression (L1 / Lasso)": "#e8590c",
  "Multinomial Logistic (L1 / Lasso)": "#e8590c",
  "Logistic Regression (Elastic Net)": "#9b51e0",
  "Multinomial Logistic (Elastic Net)": "#9b51e0",
};

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="tip" style={{ minWidth: 190 }}>
      <strong style={{ color: "var(--navy)" }}>{label}</strong>
      <div style={{ marginTop: 6, display: "flex", flexDirection: "column", gap: 3 }}>
        {payload.map((entry, index) => (
          <div key={`item-${index}`} style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
            <span style={{ color: entry.color, fontSize: "0.82rem", fontWeight: 600 }}>
              {entry.name}:
            </span>
            <span style={{ fontWeight: 700, fontSize: "0.82rem" }}>
              {entry.value !== undefined ? (entry.value > 0 ? `+${entry.value}` : entry.value) : "0"}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default function CoefficientsChart({ models = [], taskType = "regression" }) {
  const [selectedModel, setSelectedModel] = useState("ALL");

  if (!models || models.length === 0) return null;

  // Extract all feature names from the first model
  const features = models[0]?.coefficients?.map((c) => c.feature) || [];

  // Build merged data array for Recharts
  const chartData = features.map((feat) => {
    const row = { feature: feat };
    models.forEach((m) => {
      const coefObj = m.coefficients?.find((c) => c.feature === feat);
      row[m.model_name] = coefObj ? coefObj.coef : 0;
    });
    return row;
  });

  const activeModels = selectedModel === "ALL"
    ? models
    : models.filter((m) => m.model_name === selectedModel);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10 }}>
        <div>
          <h3 className="chart-title">Model Coefficients & Weight Comparison</h3>
          <p className="muted small">
            Compare coefficient weights across models. Notice how <strong>Lasso (L1)</strong> drives non-contributing features to exactly 0 (sparsity), while <strong>Ridge (L2)</strong> shrinks them smoothly.
          </p>
        </div>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <label className="small muted" style={{ fontWeight: 600 }}>Filter Model:</label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            style={{ padding: "4px 8px", borderRadius: 6, fontSize: "0.82rem" }}
          >
            <option value="ALL">Compare All Models</option>
            {models.map((m) => (
              <option key={m.model_name} value={m.model_name}>
                {m.model_name}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="chart" style={{ height: Math.max(280, chartData.length * 42) }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            layout="vertical"
            data={chartData}
            margin={{ top: 10, right: 30, bottom: 10, left: 90 }}
          >
            <CartesianGrid stroke="#e3eaf1" strokeDasharray="3 3" horizontal={false} />
            <XAxis
              type="number"
              tick={{ fontSize: 11 }}
              tickFormatter={(v) => (v > 0 ? `+${v}` : v)}
            />
            <YAxis
              type="category"
              dataKey="feature"
              tick={{ fontSize: 12, fill: "#0B2A4A", fontWeight: 500 }}
              width={90}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: 12, paddingTop: 10 }}
              iconType="circle"
            />
            <ReferenceLine x={0} stroke="#8fa8c2" strokeWidth={1.5} />
            {activeModels.map((m) => (
              <Bar
                key={m.model_name}
                dataKey={m.model_name}
                fill={MODEL_COLORS[m.model_name] || "#0e9aa7"}
                radius={[0, 4, 4, 0]}
                barSize={selectedModel === "ALL" ? 8 : 16}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
