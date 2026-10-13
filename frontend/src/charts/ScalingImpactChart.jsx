import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Legend, ReferenceLine } from "recharts";

export default function ScalingImpactChart({ scalingData, taskType = "regression" }) {
  if (!scalingData || !scalingData.comparison) return null;

  const isReg = taskType === "regression";
  const comparison = scalingData.comparison;
  const summary = scalingData.summary || {};

  // Prepare chart data comparing scaled vs unscaled coefficients
  const chartData = comparison.map((item) => ({
    feature: item.feature,
    std_dev: item.std_dev,
    "Scaled (StandardScaler)": isReg ? item.ridge_scaled_coef : item.scaled_coef,
    "Unscaled (Raw Features)": isReg ? item.ridge_unscaled_coef : item.unscaled_coef,
  }));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div>
        <h3 className="chart-title">Feature Scaling Sensitivity & Weight Distortion</h3>
        <p className="muted small">
          In clinical data, features have vastly different physiological scales (e.g. WBC count ≈ 7,000 cells/µL vs Creatinine ≈ 1.0 mg/dL).
          Without <code>StandardScaler</code>, regularization penalties treat all weights equally: features with large numerical ranges get artificially tiny weights and escape regularization!
        </p>
      </div>

      {/* Quick Comparison Metrics Strip */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
          gap: 12,
        }}
      >
        <div style={{ background: "#eef6fd", padding: "12px 14px", borderRadius: 10, borderLeft: "4px solid #2b7de9" }}>
          <div className="muted small" style={{ fontWeight: 600 }}>Standardized Model (With Scaling)</div>
          <div style={{ fontSize: "1.3rem", fontWeight: 700, color: "var(--navy)", marginTop: 2 }}>
            {isReg ? `R² = ${summary.scaled_test_r2}` : `Accuracy = ${summary.scaled_test_accuracy}%`}
          </div>
          <div className="small muted">
            {isReg ? `Test MSE: ${summary.scaled_test_mse}` : "Features standardized to µ=0, σ=1"}
          </div>
        </div>

        <div style={{ background: "#fff5f0", padding: "12px 14px", borderRadius: 10, borderLeft: "4px solid #e8590c" }}>
          <div className="muted small" style={{ fontWeight: 600 }}>Unscaled Model (Raw Features)</div>
          <div style={{ fontSize: "1.3rem", fontWeight: 700, color: "var(--warn)", marginTop: 2 }}>
            {isReg ? `R² = ${summary.unscaled_test_r2}` : `Accuracy = ${summary.unscaled_test_accuracy}%`}
          </div>
          <div className="small muted">
            {isReg ? `Test MSE: ${summary.unscaled_test_mse}` : "Susceptible to scale distortion"}
          </div>
        </div>

        <div style={{ background: "#f8f9fa", padding: "12px 14px", borderRadius: 10, borderLeft: "4px solid #0e9aa7" }}>
          <div className="muted small" style={{ fontWeight: 600 }}>Dataset Unit Scale Disparity</div>
          <div style={{ fontSize: "1.3rem", fontWeight: 700, color: "var(--navy)", marginTop: 2 }}>
            {summary.max_scale_disparity_ratio}x
          </div>
          <div className="small muted">Ratio of largest to smallest feature std dev</div>
        </div>
      </div>

      {/* Feature scale disparity table */}
      <div style={{ overflowX: "auto", border: "1px solid var(--line)", borderRadius: 10 }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem" }}>
          <thead>
            <tr style={{ background: "#eef3f9", textAlign: "left" }}>
              <th style={{ padding: "8px 12px", color: "var(--navy)" }}>Clinical Feature</th>
              <th style={{ padding: "8px 12px", color: "var(--navy)" }}>Raw Std Dev (Scale)</th>
              <th style={{ padding: "8px 12px", color: "#0e9aa7" }}>Standardized Weight (Scaled)</th>
              <th style={{ padding: "8px 12px", color: "#e8590c" }}>Raw Weight (Unscaled)</th>
              <th style={{ padding: "8px 12px", color: "var(--navy)" }}>Scaling Consequence</th>
            </tr>
          </thead>
          <tbody>
            {comparison.map((item, idx) => {
              const scaledW = isReg ? item.ridge_scaled_coef : item.scaled_coef;
              const unscaledW = isReg ? item.ridge_unscaled_coef : item.unscaled_coef;
              const isLargeScale = item.std_dev > 100;
              const isSmallScale = item.std_dev < 2;

              return (
                <tr key={item.feature} style={{ borderTop: "1px solid var(--line)", background: idx % 2 === 0 ? "#fff" : "#fafbfc" }}>
                  <td style={{ padding: "8px 12px", fontWeight: 600, color: "var(--navy)" }}>
                    {item.feature}
                  </td>
                  <td style={{ padding: "8px 12px", fontFamily: "monospace" }}>
                    {item.std_dev}
                  </td>
                  <td style={{ padding: "8px 12px", fontWeight: 600, color: "#0e9aa7", fontFamily: "monospace" }}>
                    {scaledW > 0 ? `+${scaledW}` : scaledW}
                  </td>
                  <td style={{ padding: "8px 12px", fontWeight: 600, color: "#e8590c", fontFamily: "monospace" }}>
                    {unscaledW > 0 ? `+${unscaledW}` : unscaledW}
                  </td>
                  <td style={{ padding: "8px 12px", fontSize: "0.8rem", color: "var(--muted)" }}>
                    {isLargeScale ? (
                      <span style={{ color: "#d9480f" }}>⚠️ Unscaled weight compressed due to large unit</span>
                    ) : isSmallScale ? (
                      <span style={{ color: "#1971c2" }}>⚖️ Highly sensitive to raw L2 penalty</span>
                    ) : (
                      <span>Balanced</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Chart: Scaled vs Unscaled comparison */}
      <div className="chart" style={{ height: Math.max(260, chartData.length * 36) }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            layout="vertical"
            data={chartData}
            margin={{ top: 10, right: 30, bottom: 10, left: 90 }}
          >
            <CartesianGrid stroke="#e3eaf1" strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis
              type="category"
              dataKey="feature"
              tick={{ fontSize: 12, fill: "#0B2A4A", fontWeight: 500 }}
              width={90}
            />
            <Tooltip />
            <Legend wrapperStyle={{ fontSize: 12, paddingTop: 6 }} iconType="circle" />
            <ReferenceLine x={0} stroke="#8fa8c2" strokeWidth={1.5} />
            <Bar dataKey="Scaled (StandardScaler)" fill="#0e9aa7" radius={[0, 4, 4, 0]} barSize={10} />
            <Bar dataKey="Unscaled (Raw Features)" fill="#e8590c" radius={[0, 4, 4, 0]} barSize={10} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
