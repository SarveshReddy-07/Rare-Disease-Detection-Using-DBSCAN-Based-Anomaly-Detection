import { useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  ReferenceLine,
} from "recharts";

const CustomBVTooltip = ({ active, payload, label, isRegression }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  const paramLabel = isRegression ? `α = ${d?.alpha} (log₁₀: ${label})` : `C = ${d?.C} (log₁₀: ${label})`;

  return (
    <div className="tip" style={{ minWidth: 200 }}>
      <strong style={{ color: "var(--navy)" }}>{paramLabel}</strong>
      <div style={{ marginTop: 6, display: "flex", flexDirection: "column", gap: 4 }}>
        {payload.map((entry, index) => (
          <div key={`entry-${index}`} style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
            <span style={{ color: entry.color, fontSize: "0.82rem", fontWeight: 600 }}>{entry.name}:</span>
            <span style={{ fontWeight: 700, fontSize: "0.82rem" }}>
              {isRegression ? entry.value : `${entry.value}%`}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default function BiasVarianceChart({ biasVarianceData, taskType = "regression" }) {
  const [viewMode, setViewMode] = useState("regularization"); // 'regularization' | 'learning_curve'
  if (!biasVarianceData) return null;

  const isReg = taskType === "regression";
  const curve = biasVarianceData.curve || [];
  const learningCurve = biasVarianceData.learning_curve || [];
  const optimalParam = isReg ? biasVarianceData.optimal_alpha : biasVarianceData.optimal_C;
  const currentParam = isReg ? biasVarianceData.current_alpha : biasVarianceData.current_C;
  const optimalLog = optimalParam ? Math.round(Math.log10(optimalParam) * 100) / 100 : null;
  const currentLog = currentParam ? Math.round(Math.log10(currentParam) * 100) / 100 : null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 10 }}>
        <div>
          <h3 className="chart-title">The Bias-Variance Trade-Off Curve</h3>
          <p className="muted small">
            {isReg
              ? "Training Error vs Test Error across regularization strength log₁₀(α). Low α causes high variance (overfitting), while excessively high α causes high bias (underfitting)."
              : "Training Accuracy vs Test Accuracy across inverse regularization log₁₀(C). Large C risks overfitting, whereas tiny C over-penalizes and causes underfitting."}
          </p>
        </div>
        {isReg && learningCurve.length > 0 && (
          <div style={{ display: "flex", gap: 4, background: "#eef3f9", padding: 3, borderRadius: 8 }}>
            <button
              className={`btn btn-sm ${viewMode === "regularization" ? "btn-primary" : "btn-secondary"}`}
              style={{ fontSize: "0.78rem", padding: "4px 10px" }}
              onClick={() => setViewMode("regularization")}
            >
              Penalty Sweep (α)
            </button>
            <button
              className={`btn btn-sm ${viewMode === "learning_curve" ? "btn-primary" : "btn-secondary"}`}
              style={{ fontSize: "0.78rem", padding: "4px 10px" }}
              onClick={() => setViewMode("learning_curve")}
            >
              Learning Curve (Data Size)
            </button>
          </div>
        )}
      </div>

      {viewMode === "regularization" ? (
        <>
          {/* Visual Trade-Off Region Annotations */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr 1fr",
              gap: 8,
              textAlign: "center",
              fontSize: "0.76rem",
              fontWeight: 600,
              padding: "6px 12px",
              background: "#f4f8fb",
              borderRadius: 8,
              border: "1px dashed var(--line)",
            }}
          >
            <div style={{ color: isReg ? "#d9480f" : "#1971c2" }}>
              {isReg ? "← High Variance (Low α Overfitting)" : "← High Bias (Small C Underfitting)"}
            </div>
            <div style={{ color: "#2b8a3e" }}>
              ★ Optimal Trade-Off Sweet Spot ★
            </div>
            <div style={{ color: isReg ? "#1971c2" : "#d9480f" }}>
              {isReg ? "High Bias (Heavy α Underfitting) →" : "High Variance (Large C Overfitting) →"}
            </div>
          </div>

          <div className="chart" style={{ height: 320 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={curve} margin={{ top: 15, right: 30, bottom: 15, left: 20 }}>
                <CartesianGrid stroke="#e3eaf1" strokeDasharray="3 3" />
                <XAxis
                  dataKey={isReg ? "log_alpha" : "log_C"}
                  tick={{ fontSize: 11 }}
                  label={{
                    value: isReg ? "Regularisation Strength log₁₀(α)" : "Inverse Penalty log₁₀(C)",
                    position: "insideBottom",
                    offset: -8,
                    fontSize: 12,
                    fill: "var(--muted)",
                  }}
                />
                <YAxis
                  tick={{ fontSize: 11 }}
                  label={{
                    value: isReg ? "Mean Squared Error (MSE)" : "Accuracy (%)",
                    angle: -90,
                    position: "insideLeft",
                    fontSize: 12,
                    fill: "var(--muted)",
                  }}
                />
                <Tooltip content={<CustomBVTooltip isRegression={isReg} />} />
                <Legend wrapperStyle={{ fontSize: 12, paddingTop: 6 }} iconType="plainline" />

                {optimalLog !== null && (
                  <ReferenceLine
                    x={optimalLog}
                    stroke="#1f9d6b"
                    strokeDasharray="4 4"
                    strokeWidth={2}
                    label={{
                      value: `Optimal (${isReg ? "α" : "C"} ≈ ${optimalParam})`,
                      position: "top",
                      fill: "#1f9d6b",
                      fontSize: 11,
                      fontWeight: 600,
                    }}
                  />
                )}

                {currentLog !== null && currentLog !== optimalLog && (
                  <ReferenceLine
                    x={currentLog}
                    stroke="#e8590c"
                    strokeWidth={2}
                    label={{
                      value: `Current (${isReg ? "α" : "C"}=${currentParam})`,
                      position: "insideTopRight",
                      fill: "#e8590c",
                      fontSize: 11,
                      fontWeight: 600,
                    }}
                  />
                )}

                {isReg ? (
                  <>
                    <Line
                      type="monotone"
                      dataKey="ridge_train_mse"
                      name="Ridge Train MSE (Bias proxy)"
                      stroke="#0e9aa7"
                      strokeWidth={2}
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="ridge_test_mse"
                      name="Ridge Test MSE (Generalisation)"
                      stroke="#2b7de9"
                      strokeWidth={2.5}
                      dot={{ r: 3 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="lasso_test_mse"
                      name="Lasso Test MSE"
                      stroke="#e8590c"
                      strokeWidth={1.8}
                      strokeDasharray="4 2"
                      dot={false}
                    />
                  </>
                ) : (
                  <>
                    <Line
                      type="monotone"
                      dataKey="l2_train_acc"
                      name="L2 Train Acc %"
                      stroke="#0e9aa7"
                      strokeWidth={2}
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="l2_test_acc"
                      name="L2 Test Acc % (Generalisation)"
                      stroke="#2b7de9"
                      strokeWidth={2.5}
                      dot={{ r: 3 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="l1_test_acc"
                      name="L1 Test Acc %"
                      stroke="#e8590c"
                      strokeWidth={1.8}
                      strokeDasharray="4 2"
                      dot={false}
                    />
                  </>
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      ) : (
        <div className="chart" style={{ height: 320 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={learningCurve} margin={{ top: 15, right: 30, bottom: 15, left: 20 }}>
              <CartesianGrid stroke="#e3eaf1" strokeDasharray="3 3" />
              <XAxis
                dataKey="pct"
                tickFormatter={(v) => `${v}%`}
                tick={{ fontSize: 11 }}
                label={{
                  value: "Training Set Size (% of Dataset)",
                  position: "insideBottom",
                  offset: -8,
                  fontSize: 12,
                  fill: "var(--muted)",
                }}
              />
              <YAxis
                tick={{ fontSize: 11 }}
                label={{
                  value: "Mean Squared Error",
                  angle: -90,
                  position: "insideLeft",
                  fontSize: 12,
                  fill: "var(--muted)",
                }}
              />
              <Tooltip />
              <Legend wrapperStyle={{ fontSize: 12, paddingTop: 6 }} />
              <Line
                type="monotone"
                dataKey="train_mse"
                name="Training Error"
                stroke="#0e9aa7"
                strokeWidth={2}
                dot={{ r: 4 }}
              />
              <Line
                type="monotone"
                dataKey="val_mse"
                name="Cross-Validation Error"
                stroke="#e8590c"
                strokeWidth={2.5}
                dot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
