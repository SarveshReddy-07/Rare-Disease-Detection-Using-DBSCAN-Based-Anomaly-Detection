import { useEffect, useState, useMemo } from "react";
import { Link } from "react-router-dom";
import { api } from "../services/api.js";
import { useApp } from "../components/AppContext.jsx";
import { PageHeader, Card, StatCard, EmptyState, Loader, ErrorBanner } from "../components/ui.jsx";
import CoefficientsChart from "../charts/CoefficientsChart.jsx";
import BiasVarianceChart from "../charts/BiasVarianceChart.jsx";
import ScalingImpactChart from "../charts/ScalingImpactChart.jsx";

const REG_ALPHA_PRESETS = [0.001, 0.01, 0.1, 1.0, 10.0, 50.0];
const CLS_C_PRESETS = [0.01, 0.1, 1.0, 10.0, 100.0];

export default function LinearModels() {
  const { results: r, prep, upload, ready } = useApp();

  // Targets and metadata
  const [targetsData, setTargetsData] = useState(null);
  const [loadingTargets, setLoadingTargets] = useState(true);
  const [error, setError] = useState("");

  // Configuration form state
  const [taskType, setTaskType] = useState("regression"); // 'regression' | 'classification'
  const [targetCol, setTargetCol] = useState("");
  const [selectedFeatures, setSelectedFeatures] = useState([]);
  const [alpha, setAlpha] = useState(1.0);
  const [C, setC] = useState(1.0);
  const [l1Ratio, setL1Ratio] = useState(0.5);
  const [applyScaling, setApplyScaling] = useState(true);
  const [testSize, setTestSize] = useState(0.25);

  // Training results state
  const [data, setData] = useState(null);
  const [training, setTraining] = useState(false);
  const [activeTab, setActiveTab] = useState("models"); // 'models' | 'regularisation' | 'bias_variance' | 'scaling' | 'reasoning'

  // Load available target columns on mount
  useEffect(() => {
    let isMounted = true;
    setLoadingTargets(true);
    api.linearModelsTargets()
      .then((res) => {
        if (!isMounted) return;
        setTargetsData(res);
        // Default target selection
        if (taskType === "regression" && res.default_regression_target) {
          setTargetCol(res.default_regression_target);
        } else if (taskType === "classification" && res.default_classification_target) {
          setTargetCol(res.default_classification_target);
        }
        if (res.available_features) {
          setSelectedFeatures(res.available_features);
        }
      })
      .catch((err) => {
        if (!isMounted) return;
        setError(err.message || "Failed to load candidate target variables.");
      })
      .finally(() => {
        if (isMounted) setLoadingTargets(false);
      });

    return () => {
      isMounted = false;
    };
  }, []);

  // When taskType toggles, update targetCol
  const handleTaskTypeChange = (newType) => {
    setTaskType(newType);
    if (!targetsData) return;
    if (newType === "regression") {
      setTargetCol(targetsData.default_regression_target || "");
    } else {
      setTargetCol(targetsData.default_classification_target || "");
    }
  };

  // Run model training
  const handleTrainModels = async (overrides = {}) => {
    if (!targetCol) {
      setError("Please select a target variable.");
      return;
    }
    setTraining(true);
    setError("");

    const currentTask = overrides.taskType !== undefined ? overrides.taskType : taskType;
    const currentTarget = overrides.targetCol !== undefined ? overrides.targetCol : targetCol;
    const currentAlpha = overrides.alpha !== undefined ? overrides.alpha : alpha;
    const currentC = overrides.C !== undefined ? overrides.C : C;
    const currentL1Ratio = overrides.l1Ratio !== undefined ? overrides.l1Ratio : l1Ratio;
    const currentScaling = overrides.applyScaling !== undefined ? overrides.applyScaling : applyScaling;

    try {
      const res = await api.trainLinearModels({
        task_type: currentTask,
        target_column: currentTarget,
        feature_columns: selectedFeatures.length > 0 ? selectedFeatures : undefined,
        test_size: testSize,
        alpha: currentAlpha,
        C: currentC,
        l1_ratio: currentL1Ratio,
        apply_scaling: currentScaling,
      });
      setData(res);
    } catch (err) {
      setError(err.message || "Failed to train linear models.");
    } finally {
      setTraining(false);
    }
  };

  // Auto-train on first target load
  useEffect(() => {
    if (targetsData && targetCol && !data && !training) {
      handleTrainModels();
    }
  }, [targetsData, targetCol]);

  // Handle feature checkbox toggle
  const toggleFeature = (feat) => {
    if (selectedFeatures.includes(feat)) {
      if (selectedFeatures.length <= 1) return; // Keep at least one
      setSelectedFeatures(selectedFeatures.filter((f) => f !== feat));
    } else {
      setSelectedFeatures([...selectedFeatures, feat]);
    }
  };

  const selectAllFeatures = () => {
    if (targetsData?.available_features) {
      setSelectedFeatures(targetsData.available_features);
    }
  };

  if (!ready || loadingTargets) {
    return <Loader text="Inspecting candidate clinical targets and preparing linear models..." />;
  }

  if (!upload && !prep && !r) {
    return (
      <EmptyState
        title="No Dataset Uploaded"
        text="Upload a structured CSV dataset to train and compare linear supervised models, evaluate regularisation, and explore bias-variance trade-offs."
        to="/upload"
        action="Upload Dataset"
      />
    );
  }

  const isReg = taskType === "regression";
  const regressionTargets = targetsData?.regression_targets || [];
  const classificationTargets = targetsData?.classification_targets || [];
  const availableFeatures = targetsData?.available_features || [];

  const bestModel = data?.models?.find((m) => m.model_name === data?.best_model_name) || data?.models?.[0];
  const bvDiagnosis = data?.bias_variance?.diagnosis;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <PageHeader
        title="Supervised Linear Models & Theoretical Reasoning"
        subtitle="Linear Regression, Ridge, Lasso, Elastic Net, Logistic Regression, and Multinomial Logistic Regression with deep analysis of regularisation, bias-variance trade-off, and feature scaling."
      >
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          {r && (
            <Link to="/confusion-matrix" className="btn btn-secondary" style={{ fontSize: "0.85rem" }}>
              ← Confusion Matrix
            </Link>
          )}
          {r && (
            <Link to="/anomalies" className="btn btn-secondary" style={{ fontSize: "0.85rem" }}>
              Anomaly Records →
            </Link>
          )}
        </div>
      </PageHeader>

      <ErrorBanner message={error} onClose={() => setError("")} />

      {/* Model Configuration Control Card */}
      <Card title="Model Configuration & Hyperparameters">
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 16 }}>
          {/* Task Type & Target Selector */}
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div>
              <label className="small muted" style={{ fontWeight: 600, display: "block", marginBottom: 4 }}>
                Supervised Learning Task:
              </label>
              <div style={{ display: "flex", gap: 8 }}>
                <button
                  type="button"
                  className={`btn ${isReg ? "btn-primary" : "btn-secondary"}`}
                  style={{ flex: 1, fontSize: "0.86rem" }}
                  onClick={() => handleTaskTypeChange("regression")}
                >
                  📈 Continuous (Regression)
                </button>
                <button
                  type="button"
                  className={`btn ${!isReg ? "btn-primary" : "btn-secondary"}`}
                  style={{ flex: 1, fontSize: "0.86rem" }}
                  onClick={() => handleTaskTypeChange("classification")}
                >
                  🏷️ Categorical (Classification)
                </button>
              </div>
            </div>

            <div>
              <label className="small muted" style={{ fontWeight: 600, display: "block", marginBottom: 4 }}>
                Target Variable ({isReg ? "Continuous Dependent Variable" : "Discrete Category / Label"}):
              </label>
              <select
                value={targetCol}
                onChange={(e) => setTargetCol(e.target.value)}
                style={{ width: "100%", padding: "7px 10px", borderRadius: 8, fontSize: "0.9rem" }}
              >
                {isReg ? (
                  regressionTargets.map((t) => (
                    <option key={t.column} value={t.column}>
                      {t.column} (Range: {t.min} to {t.max}, μ={t.mean}) {t.is_dbscan_feature ? "★ DBSCAN Anomaly Score" : ""}
                    </option>
                  ))
                ) : (
                  classificationTargets.map((t) => (
                    <option key={t.column} value={t.column}>
                      {t.column} ({t.type === "binary" ? "Binary (2 classes)" : `Multinomial (${t.unique_count} classes)`}) {t.is_dbscan_feature ? "★ DBSCAN Flag" : ""}
                    </option>
                  ))
                )}
              </select>
            </div>
          </div>

          {/* Regularization Parameters */}
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {isReg ? (
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                  <label className="small muted" style={{ fontWeight: 600 }}>
                    Regularization Strength (α / Lambda):
                  </label>
                  <span style={{ fontWeight: 700, color: "var(--navy)", fontSize: "0.9rem" }}>
                    α = {alpha}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.001"
                  max="50"
                  step="0.05"
                  value={alpha}
                  onChange={(e) => setAlpha(parseFloat(e.target.value))}
                  style={{ width: "100%", marginTop: 4 }}
                />
                <div style={{ display: "flex", gap: 4, marginTop: 4, flexWrap: "wrap" }}>
                  <span className="small muted" style={{ marginRight: 4 }}>Presets:</span>
                  {REG_ALPHA_PRESETS.map((p) => (
                    <button
                      key={p}
                      type="button"
                      className={`btn btn-sm ${alpha === p ? "btn-primary" : "btn-secondary"}`}
                      style={{ fontSize: "0.72rem", padding: "2px 6px" }}
                      onClick={() => setAlpha(p)}
                    >
                      {p}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                  <label className="small muted" style={{ fontWeight: 600 }}>
                    Inverse Regularization (C = 1/λ):
                  </label>
                  <span style={{ fontWeight: 700, color: "var(--navy)", fontSize: "0.9rem" }}>
                    C = {C}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.01"
                  max="50"
                  step="0.1"
                  value={C}
                  onChange={(e) => setC(parseFloat(e.target.value))}
                  style={{ width: "100%", marginTop: 4 }}
                />
                <div style={{ display: "flex", gap: 4, marginTop: 4, flexWrap: "wrap" }}>
                  <span className="small muted" style={{ marginRight: 4 }}>Presets:</span>
                  {CLS_C_PRESETS.map((p) => (
                    <button
                      key={p}
                      type="button"
                      className={`btn btn-sm ${C === p ? "btn-primary" : "btn-secondary"}`}
                      style={{ fontSize: "0.72rem", padding: "2px 6px" }}
                      onClick={() => setC(p)}
                    >
                      {p}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Elastic Net Mixing Ratio */}
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                <label className="small muted" style={{ fontWeight: 600 }}>
                  Elastic Net L1 Ratio (ρ):
                </label>
                <span style={{ fontWeight: 700, color: "var(--navy)", fontSize: "0.85rem" }}>
                  {l1Ratio} ({l1Ratio === 1 ? "Pure Lasso L1" : l1Ratio === 0 ? "Pure Ridge L2" : `${Math.round(l1Ratio * 100)}% L1 / ${Math.round((1 - l1Ratio) * 100)}% L2`})
                </span>
              </div>
              <input
                type="range"
                min="0.05"
                max="0.95"
                step="0.05"
                value={l1Ratio}
                onChange={(e) => setL1Ratio(parseFloat(e.target.value))}
                style={{ width: "100%", marginTop: 4 }}
              />
            </div>
          </div>

          {/* Feature Scaling & Train/Test Split */}
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", background: "#f8fafc", padding: "8px 12px", borderRadius: 8, border: "1px solid var(--line)" }}>
              <div>
                <strong style={{ fontSize: "0.88rem", color: "var(--navy)" }}>Feature Scaling (StandardScaler)</strong>
                <p className="muted small" style={{ margin: 0 }}>
                  Normalizes features to zero mean and unit variance (µ=0, σ=1).
                </p>
              </div>
              <label style={{ display: "flex", alignItems: "center", cursor: "pointer", gap: 6, fontWeight: 600, fontSize: "0.85rem" }}>
                <input
                  type="checkbox"
                  checked={applyScaling}
                  onChange={(e) => setApplyScaling(e.target.checked)}
                />
                {applyScaling ? "Active" : "Disabled"}
              </label>
            </div>

            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <label className="small muted" style={{ fontWeight: 600 }}>Train / Test Split:</label>
              <select
                value={testSize}
                onChange={(e) => setTestSize(parseFloat(e.target.value))}
                style={{ padding: "4px 8px", borderRadius: 6, fontSize: "0.82rem" }}
              >
                <option value={0.2}>80% Train / 20% Test</option>
                <option value={0.25}>75% Train / 25% Test (Standard)</option>
                <option value={0.3}>70% Train / 30% Test</option>
              </select>
            </div>

            <button
              type="button"
              className="btn btn-primary"
              disabled={training}
              onClick={() => handleTrainModels()}
              style={{ marginTop: "auto", padding: "10px 16px", fontWeight: 600 }}
            >
              {training ? "Fitting & Analyzing Models..." : "⚡ Train & Analyze Models"}
            </button>
          </div>
        </div>

        {/* Predictor Features Multi-Select Drawer */}
        <div style={{ marginTop: 16, paddingTop: 14, borderTop: "1px solid var(--line)" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
            <span className="small muted" style={{ fontWeight: 600 }}>
              Included Features ({selectedFeatures.length} of {availableFeatures.length}):
            </span>
            <button
              type="button"
              className="link small"
              onClick={selectAllFeatures}
            >
              Select All Numeric Features
            </button>
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            {availableFeatures.map((f) => {
              const isSelected = selectedFeatures.includes(f);
              const isCurrentTarget = f === targetCol;
              if (isCurrentTarget) return null;
              return (
                <button
                  key={f}
                  type="button"
                  onClick={() => toggleFeature(f)}
                  style={{
                    padding: "4px 10px",
                    borderRadius: 20,
                    fontSize: "0.8rem",
                    border: `1px solid ${isSelected ? "var(--teal)" : "#cfdbe7"}`,
                    background: isSelected ? "rgba(14, 154, 167, 0.12)" : "#fff",
                    color: isSelected ? "var(--navy)" : "var(--muted)",
                    fontWeight: isSelected ? 600 : 400,
                    cursor: "pointer",
                  }}
                >
                  {isSelected ? "✓ " : "+ "}
                  {f}
                </button>
              );
            })}
          </div>
        </div>
      </Card>

      {/* Primary KPI Strip */}
      {data && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: 12 }}>
          <StatCard
            label="Target Variable"
            value={data.target_column}
            hint={isReg ? `Continuous (${data.train_samples} train / ${data.test_samples} test)` : `${data.sub_type.toUpperCase()} (${data.classes?.join(", ")})`}
          />
          <StatCard
            label="Top Performing Model"
            value={bestModel?.model_name.split("(")[0].trim() || "-"}
            tone="ok"
            hint={isReg ? `Test R² = ${bestModel?.test_r2} (MSE: ${bestModel?.test_mse})` : `Test Accuracy = ${bestModel?.test_accuracy}% (F1: ${bestModel?.f1_score}%)`}
          />
          <StatCard
            label="Bias-Variance Status"
            value={bvDiagnosis?.status.split("(")[0].trim() || "Optimal"}
            tone={bvDiagnosis?.tone === "ok" ? "ok" : "warn"}
            hint={isReg ? `Optimal α ≈ ${data.bias_variance.optimal_alpha} (Current: ${alpha})` : `Optimal C ≈ ${data.bias_variance.optimal_C} (Current: ${C})`}
          />
          <StatCard
            label="L1 Sparsity (Feature Selection)"
            value={`${data.models.find((m) => m.model_name.includes("Lasso") || m.model_name.includes("L1"))?.active_features || 0} / ${data.models[0]?.total_features}`}
            hint="Features retained by Lasso penalty"
          />
        </div>
      )}

      {/* Navigation Tabs for Deep Dive Reasoning */}
      {data && (
        <div style={{ display: "flex", gap: 8, borderBottom: "2px solid var(--line)", paddingBottom: 2, overflowX: "auto" }}>
          <button
            type="button"
            className={`btn btn-sm ${activeTab === "models" ? "btn-primary" : "btn-secondary"}`}
            style={{ borderRadius: "8px 8px 0 0", padding: "8px 16px", fontWeight: 600 }}
            onClick={() => setActiveTab("models")}
          >
            📊 1. Model Benchmark ({isReg ? "OLS / Ridge / Lasso / Elastic Net" : "Logistic / Multinomial"})
          </button>
          <button
            type="button"
            className={`btn btn-sm ${activeTab === "regularisation" ? "btn-primary" : "btn-secondary"}`}
            style={{ borderRadius: "8px 8px 0 0", padding: "8px 16px", fontWeight: 600 }}
            onClick={() => setActiveTab("regularisation")}
          >
            ⚖️ 2. Regularisation Reasoning (L1 vs L2)
          </button>
          <button
            type="button"
            className={`btn btn-sm ${activeTab === "bias_variance" ? "btn-primary" : "btn-secondary"}`}
            style={{ borderRadius: "8px 8px 0 0", padding: "8px 16px", fontWeight: 600 }}
            onClick={() => setActiveTab("bias_variance")}
          >
            📉 3. Bias-Variance Trade-Off Curve
          </button>
          <button
            type="button"
            className={`btn btn-sm ${activeTab === "scaling" ? "btn-primary" : "btn-secondary"}`}
            style={{ borderRadius: "8px 8px 0 0", padding: "8px 16px", fontWeight: 600 }}
            onClick={() => setActiveTab("scaling")}
          >
            📏 4. Feature Scaling Analysis
          </button>
          <button
            type="button"
            className={`btn btn-sm ${activeTab === "reasoning" ? "btn-primary" : "btn-secondary"}`}
            style={{ borderRadius: "8px 8px 0 0", padding: "8px 16px", fontWeight: 600 }}
            onClick={() => setActiveTab("reasoning")}
          >
            🧬 5. Clinical Synthesis & DBSCAN Context
          </button>
        </div>
      )}

      {/* TAB 1: MODEL BENCHMARK */}
      {data && activeTab === "models" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          {/* Comparative Model Cards Grid */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 16 }}>
            {data.models.map((m) => {
              const isBest = m.model_name === data.best_model_name;
              return (
                <div
                  key={m.model_name}
                  className="card"
                  style={{
                    border: isBest ? "2px solid #0e9aa7" : "1px solid var(--line)",
                    boxShadow: isBest ? "0 4px 14px rgba(14, 154, 167, 0.15)" : undefined,
                    position: "relative",
                  }}
                >
                  {isBest && (
                    <span
                      style={{
                        position: "absolute",
                        top: -10,
                        right: 14,
                        background: "#0e9aa7",
                        color: "#fff",
                        padding: "2px 8px",
                        borderRadius: 12,
                        fontSize: "0.72rem",
                        fontWeight: 700,
                      }}
                    >
                      ★ Top Generalization
                    </span>
                  )}
                  <h3 style={{ fontSize: "1.05rem", color: "var(--navy)" }}>{m.model_name}</h3>
                  <p className="muted small" style={{ minHeight: 36 }}>{m.description || "Linear decision surface / loss minimization"}</p>

                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 10 }}>
                    {isReg ? (
                      <>
                        <div className="stat" style={{ padding: "8px 10px" }}>
                          <div className="stat-label">Test R² Score</div>
                          <div className="stat-value" style={{ fontSize: "1.2rem", color: m.test_r2 > 0 ? "var(--navy)" : "var(--warn)" }}>
                            {m.test_r2}
                          </div>
                          <div className="stat-hint">Train R²: {m.train_r2}</div>
                        </div>
                        <div className="stat" style={{ padding: "8px 10px" }}>
                          <div className="stat-label">Test RMSE</div>
                          <div className="stat-value" style={{ fontSize: "1.2rem" }}>
                            {m.test_rmse}
                          </div>
                          <div className="stat-hint">MSE: {m.test_mse}</div>
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="stat" style={{ padding: "8px 10px" }}>
                          <div className="stat-label">Test Accuracy</div>
                          <div className="stat-value" style={{ fontSize: "1.2rem", color: "var(--navy)" }}>
                            {m.test_accuracy}%
                          </div>
                          <div className="stat-hint">Train: {m.train_accuracy}%</div>
                        </div>
                        <div className="stat" style={{ padding: "8px 10px" }}>
                          <div className="stat-label">F1-Score</div>
                          <div className="stat-value" style={{ fontSize: "1.2rem" }}>
                            {m.f1_score}%
                          </div>
                          <div className="stat-hint">Precision: {m.precision}%</div>
                        </div>
                      </>
                    )}
                  </div>

                  <div style={{ marginTop: 12, paddingTop: 10, borderTop: "1px solid var(--line)", display: "flex", justifyContent: "space-between", fontSize: "0.82rem" }}>
                    <span className="muted">Active Biomarkers:</span>
                    <strong style={{ color: "var(--navy)" }}>
                      {m.active_features} / {m.total_features}
                      {m.sparsity_pct > 0 && <span style={{ color: "var(--warn)", marginLeft: 4 }}>({m.sparsity_pct}% sparse)</span>}
                    </strong>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Detailed Performance Metrics Table */}
          <Card title="Comparative Model Metrics Matrix" subtitle="Side-by-side performance indicators evaluated on the held-out test partition">
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem" }}>
                <thead>
                  <tr style={{ background: "#eef3f9", textAlign: "left" }}>
                    <th style={{ padding: "10px 12px", color: "var(--navy)" }}>Model Name</th>
                    {isReg ? (
                      <>
                        <th style={{ padding: "10px 12px", color: "var(--navy)" }}>Train R²</th>
                        <th style={{ padding: "10px 12px", color: "var(--navy)" }}>Test R²</th>
                        <th style={{ padding: "10px 12px", color: "var(--navy)" }}>Test MSE</th>
                        <th style={{ padding: "10px 12px", color: "var(--navy)" }}>Test RMSE</th>
                        <th style={{ padding: "10px 12px", color: "var(--navy)" }}>Test MAE</th>
                        <th style={{ padding: "10px 12px", color: "var(--navy)" }}>Active Features</th>
                        <th style={{ padding: "10px 12px", color: "var(--navy)" }}>Intercept</th>
                      </>
                    ) : (
                      <>
                        <th style={{ padding: "10px 12px", color: "var(--navy)" }}>Train Accuracy</th>
                        <th style={{ padding: "10px 12px", color: "var(--navy)" }}>Test Accuracy</th>
                        <th style={{ padding: "10px 12px", color: "var(--navy)" }}>Precision</th>
                        <th style={{ padding: "10px 12px", color: "var(--navy)" }}>Recall</th>
                        <th style={{ padding: "10px 12px", color: "var(--navy)" }}>F1-Score</th>
                        <th style={{ padding: "10px 12px", color: "var(--navy)" }}>Log-Loss</th>
                        <th style={{ padding: "10px 12px", color: "var(--navy)" }}>Active Features</th>
                      </>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {data.models.map((m, idx) => (
                    <tr
                      key={m.model_name}
                      style={{
                        borderTop: "1px solid var(--line)",
                        background: m.model_name === data.best_model_name ? "rgba(14, 154, 167, 0.05)" : idx % 2 === 0 ? "#fff" : "#fafbfc",
                      }}
                    >
                      <td style={{ padding: "10px 12px", fontWeight: 600, color: "var(--navy)" }}>
                        {m.model_name}
                        {m.model_name === data.best_model_name && " ★"}
                      </td>
                      {isReg ? (
                        <>
                          <td style={{ padding: "10px 12px" }}>{m.train_r2}</td>
                          <td style={{ padding: "10px 12px", fontWeight: 700, color: m.test_r2 > 0 ? "var(--navy)" : "var(--warn)" }}>{m.test_r2}</td>
                          <td style={{ padding: "10px 12px" }}>{m.test_mse}</td>
                          <td style={{ padding: "10px 12px" }}>{m.test_rmse}</td>
                          <td style={{ padding: "10px 12px" }}>{m.test_mae}</td>
                          <td style={{ padding: "10px 12px" }}>{m.active_features} / {m.total_features}</td>
                          <td style={{ padding: "10px 12px" }}>{m.intercept}</td>
                        </>
                      ) : (
                        <>
                          <td style={{ padding: "10px 12px" }}>{m.train_accuracy}%</td>
                          <td style={{ padding: "10px 12px", fontWeight: 700, color: "var(--navy)" }}>{m.test_accuracy}%</td>
                          <td style={{ padding: "10px 12px" }}>{m.precision}%</td>
                          <td style={{ padding: "10px 12px" }}>{m.recall}%</td>
                          <td style={{ padding: "10px 12px" }}>{m.f1_score}%</td>
                          <td style={{ padding: "10px 12px" }}>{m.log_loss ?? "-"}</td>
                          <td style={{ padding: "10px 12px" }}>{m.active_features} / {m.total_features}</td>
                        </>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Predictions Sample Preview */}
          {isReg && data.predictions_preview && (
            <Card title="Actual vs Predicted Values Sample (Test Partition)" subtitle="Comparison of true patient biomarker values vs model prediction">
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem" }}>
                  <thead>
                    <tr style={{ background: "#eef3f9", textAlign: "left" }}>
                      <th style={{ padding: "6px 10px", color: "var(--navy)" }}>Test Sample</th>
                      <th style={{ padding: "6px 10px", color: "var(--navy)" }}>Actual {data.target_column}</th>
                      <th style={{ padding: "6px 10px", color: "#0e9aa7" }}>Predicted ({bestModel?.model_name.split("(")[0]})</th>
                      <th style={{ padding: "6px 10px", color: "var(--muted)" }}>Residual Error (Actual - Pred)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.predictions_preview.slice(0, 10).map((row) => (
                      <tr key={row.id} style={{ borderTop: "1px solid var(--line)" }}>
                        <td style={{ padding: "6px 10px" }}>Patient #{row.id}</td>
                        <td style={{ padding: "6px 10px", fontWeight: 600 }}>{row.actual}</td>
                        <td style={{ padding: "6px 10px", color: "#0e9aa7", fontWeight: 600 }}>{row.predicted}</td>
                        <td style={{ padding: "6px 10px", color: Math.abs(row.residual) > 10 ? "var(--warn)" : "var(--muted)" }}>
                          {row.residual > 0 ? `+${row.residual}` : row.residual}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </div>
      )}

      {/* TAB 2: REGULARISATION DEEP-DIVE */}
      {data && activeTab === "regularisation" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <Card>
            <CoefficientsChart models={data.models} taskType={taskType} />
          </Card>

          {/* Theoretical Foundations of Regularization */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 16 }}>
            <Card title="L2 Regularization (Ridge)" subtitle="Ridge regression / Tikhonov regularization">
              <div style={{ fontSize: "0.86rem", color: "var(--ink)", display: "flex", flexDirection: "column", gap: 8 }}>
                <div style={{ background: "#eef6fd", padding: "8px 10px", borderRadius: 6, fontFamily: "monospace", fontSize: "0.85rem" }}>
                  Loss = RSS + α · ∑ wⱼ²
                </div>
                <p>
                  <strong>Mechanism:</strong> Penalizes the squared Euclidean L2 norm of the weight vector.
                </p>
                <p>
                  <strong>Why it matters:</strong> Shrinks coefficients smoothly towards zero without setting them to exact zero. It stabilizes the inversion of <code>XᵀX</code> when clinical features are correlated (multicollinearity), preventing wildly oscillating weights.
                </p>
                <div style={{ background: "#f0fdf4", padding: "6px 10px", borderRadius: 6, fontSize: "0.8rem", color: "#166534" }}>
                  ✓ Best when all clinical biomarkers contribute small signals.
                </div>
              </div>
            </Card>

            <Card title="L1 Regularization (Lasso)" subtitle="Least Absolute Shrinkage and Selection Operator">
              <div style={{ fontSize: "0.86rem", color: "var(--ink)", display: "flex", flexDirection: "column", gap: 8 }}>
                <div style={{ background: "#fff5f0", padding: "8px 10px", borderRadius: 6, fontFamily: "monospace", fontSize: "0.85rem" }}>
                  Loss = RSS + α · ∑ |wⱼ|
                </div>
                <p>
                  <strong>Mechanism:</strong> Penalizes the Manhattan L1 norm of the weight vector.
                </p>
                <p>
                  <strong>Geometric Reason for Sparsity:</strong> The L1 constraint forms a sharp diamond / polytope with pointed vertices on the axes. The ellipsoidal RSS contours touch these corners first, driving uninformative biomarker weights to <strong>exact zero</strong>.
                </p>
                <div style={{ background: "#fef3c7", padding: "6px 10px", borderRadius: 6, fontSize: "0.8rem", color: "#92400e" }}>
                  ✓ Automatic Feature Selection: isolates the most critical clinical indicators.
                </div>
              </div>
            </Card>

            <Card title="Elastic Net (L1 + L2 Combination)" subtitle="Zou & Hastie convex combination">
              <div style={{ fontSize: "0.86rem", color: "var(--ink)", display: "flex", flexDirection: "column", gap: 8 }}>
                <div style={{ background: "#f3e8ff", padding: "8px 10px", borderRadius: 6, fontFamily: "monospace", fontSize: "0.82rem" }}>
                  Loss = RSS + α · [ ρ∑|wⱼ| + ½(1-ρ)∑wⱼ² ]
                </div>
                <p>
                  <strong>Mechanism:</strong> Blends L1 sparsity with L2 grouped selection via parameter <code>ρ (l1_ratio)</code>.
                </p>
                <p>
                  <strong>Clinical Utility:</strong> In genomics and blood panels, groups of biomarkers often correlate strongly. While pure Lasso arbitrarily picks only one feature from a correlated group, Elastic Net selects the entire biological group while shrinking noise.
                </p>
                <div style={{ background: "#e0e7ff", padding: "6px 10px", borderRadius: 6, fontSize: "0.8rem", color: "#3730a3" }}>
                  ✓ Ideal for high-dimensional medical panels with correlated lab tests.
                </div>
              </div>
            </Card>
          </div>
        </div>
      )}

      {/* TAB 3: BIAS-VARIANCE TRADE-OFF */}
      {data && activeTab === "bias_variance" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          {/* Bias-Variance Diagnostic Banner */}
          <div
            style={{
              background: bvDiagnosis?.tone === "ok" ? "var(--ok-bg)" : "var(--warn-bg)",
              border: `1px solid ${bvDiagnosis?.tone === "ok" ? "var(--ok)" : "var(--warn)"}`,
              borderRadius: 10,
              padding: "16px 20px",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
              <strong style={{ fontSize: "1.05rem", color: bvDiagnosis?.tone === "ok" ? "#166534" : "#92400e" }}>
                Current Diagnostic: {bvDiagnosis?.status}
              </strong>
              <span className="small muted">
                Tested with {isReg ? `α = ${alpha}` : `C = ${C}`}
              </span>
            </div>
            <p style={{ margin: 0, fontSize: "0.9rem", color: "var(--navy)" }}>
              {bvDiagnosis?.explanation}
            </p>
          </div>

          <Card>
            <BiasVarianceChart biasVarianceData={data.bias_variance} taskType={taskType} />
          </Card>

          {/* Theoretical Bias-Variance Decomposition */}
          <Card title="Mathematical Formulation: The Bias-Variance Trade-Off">
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 16 }}>
              <div style={{ background: "#f8fafc", padding: "14px", borderRadius: 10, border: "1px solid var(--line)" }}>
                <strong style={{ color: "var(--navy)", fontSize: "0.95rem" }}>Error Decomposition</strong>
                <p style={{ fontFamily: "monospace", fontSize: "0.88rem", margin: "8px 0" }}>
                  E[(y - f̂(x))²] = Bias[f̂(x)]² + Var[f̂(x)] + σ²
                </p>
                <ul style={{ paddingLeft: 18, margin: 0, fontSize: "0.82rem", color: "var(--muted)" }}>
                  <li><strong>Bias²:</strong> Error from erroneous assumptions / over-simplifying the true clinical pattern.</li>
                  <li><strong>Variance:</strong> Sensitivity to small fluctuations in the training partition.</li>
                  <li><strong>σ²:</strong> Irreducible clinical measurement noise.</li>
                </ul>
              </div>

              <div style={{ background: "#f8fafc", padding: "14px", borderRadius: 10, border: "1px solid var(--line)" }}>
                <strong style={{ color: "var(--navy)", fontSize: "0.95rem" }}>High Variance (Overfitting)</strong>
                <p className="small muted" style={{ margin: "6px 0" }}>
                  Occurs when regularization is too weak (α → 0 or C → ∞).
                </p>
                <div style={{ fontSize: "0.82rem", color: "var(--ink)" }}>
                  The model fits idiosyncratic patient noise. Training error is very low, but test generalization error surges.
                </div>
              </div>

              <div style={{ background: "#f8fafc", padding: "14px", borderRadius: 10, border: "1px solid var(--line)" }}>
                <strong style={{ color: "var(--navy)", fontSize: "0.95rem" }}>High Bias (Underfitting)</strong>
                <p className="small muted" style={{ margin: "6px 0" }}>
                  Occurs when regularization is too aggressive (α → ∞ or C → 0).
                </p>
                <div style={{ fontSize: "0.82rem", color: "var(--ink)" }}>
                  Coefficients are crushed towards zero, causing the model to miss authentic clinical biomarker relationships. Both training and test errors are high.
                </div>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* TAB 4: FEATURE SCALING ANALYSIS */}
      {data && activeTab === "scaling" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <Card>
            <ScalingImpactChart scalingData={data.feature_scaling} taskType={taskType} />
          </Card>

          <Card title="Why Feature Scaling is Non-Negotiable in Medical Machine Learning">
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 16 }}>
              <div style={{ padding: 12, background: "#f8fafc", borderRadius: 8 }}>
                <strong style={{ color: "var(--navy)" }}>1. Equal Penalty Pressure</strong>
                <p className="small muted">
                  The regularisation term <code>α · ∑ wⱼ²</code> applies the exact same numerical penalty to every weight regardless of the underlying feature’s physiological unit.
                </p>
                <p className="small" style={{ margin: 0 }}>
                  A biomarker measured in thousands (e.g. Platelets ≈ 250,000) naturally needs a tiny weight (e.g. 0.00004), effectively rendering its regularisation penalty near zero! Meanwhile, biomarkers measured in single digits (e.g. Creatinine ≈ 1.2) are crushed by excessive penalty.
                </p>
              </div>

              <div style={{ padding: 12, background: "#f8fafc", borderRadius: 8 }}>
                <strong style={{ color: "var(--navy)" }}>2. Condition Number & Solver Convergence</strong>
                <p className="small muted">
                  Without scaling, the condition number of <code>XᵀX</code> expands dramatically, causing gradient descent and Newton-Raphson solvers to oscillate and fail to converge within standard iteration budgets.
                </p>
                <p className="small" style={{ margin: 0 }}>
                  With <code>StandardScaler</code>, the contours of the loss function become spherical, enabling fast, symmetric convergence.
                </p>
              </div>

              <div style={{ padding: 12, background: "#f8fafc", borderRadius: 8 }}>
                <strong style={{ color: "var(--navy)" }}>3. Interpretable Clinical Odds Ratios</strong>
                <p className="small muted">
                  In Logistic Regression, odds ratios <code>exp(wⱼ)</code> represent the multiplicative change in odds per 1-unit increase in the feature.
                </p>
                <p className="small" style={{ margin: 0 }}>
                  Standardizing features makes odds ratios directly comparable: an odds ratio for a 1 standard-deviation change allows clinicians to rank biomarker importance on a fair scale.
                </p>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* TAB 5: CLINICAL SYNTHESIS & REASONING */}
      {data && activeTab === "reasoning" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <Card title="Clinical Synthesis: Supervised Models vs Unsupervised DBSCAN">
            <div style={{ display: "flex", flexDirection: "column", gap: 14, fontSize: "0.9rem" }}>
              <p>
                In rare disease detection, unsupervised clustering (DBSCAN) and supervised linear modeling address complementary sides of the diagnostic challenge:
              </p>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                <div style={{ background: "#eef6fd", padding: "14px 16px", borderRadius: 10, borderLeft: "4px solid #2b7de9" }}>
                  <strong style={{ color: "var(--navy)" }}>Unsupervised DBSCAN (Anomaly Detection)</strong>
                  <ul style={{ paddingLeft: 18, marginTop: 8, fontSize: "0.85rem" }}>
                    <li>Requires <strong>no pre-existing labels</strong> or disease diagnoses.</li>
                    <li>Identifies patients whose multidimensional biomarker combinations depart from all dense physiological clusters (noise cluster -1).</li>
                    <li>Ideal for exploratory discovery of novel, ultra-rare disease presentations.</li>
                  </ul>
                </div>

                <div style={{ background: "#f0fdf4", padding: "14px 16px", borderRadius: 10, borderLeft: "4px solid #1f9d6b" }}>
                  <strong style={{ color: "var(--navy)" }}>Supervised Linear & Logistic Models</strong>
                  <ul style={{ paddingLeft: 18, marginTop: 8, fontSize: "0.85rem" }}>
                    <li>Learn direct mappings from clinical features to known target outcomes (e.g. Anomaly Score, confirmed disease status, or physiological thresholds).</li>
                    <li>Provide <strong>transparent, auditable coefficients</strong> and odds ratios that clinical reviewers can verify against medical literature.</li>
                    <li>L1 Lasso regularisation automatically identifies the minimal panel of lab tests required to flag at-risk profiles.</li>
                  </ul>
                </div>
              </div>

              <div style={{ background: "#fff7e8", border: "1px solid #f3dfb4", borderRadius: 10, padding: "14px 18px", color: "#7a5200" }}>
                <strong>Research & Educational Caveat:</strong> Neither DBSCAN anomaly flags nor linear regression predictions represent definitive clinical diagnoses. They are statistical screening tools designed to assist qualified healthcare professionals in prioritizing unusual profiles for specialized diagnostic workups.
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
