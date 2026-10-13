"""Supervised Linear Models, Regularisation, Bias-Variance Trade-off, and Feature Scaling.

Includes:
1. Linear Regression (OLS)
2. Ridge Regression (L2 penalty)
3. Lasso Regression (L1 penalty)
4. Elastic Net Regression (L1 + L2 penalty)
5. Logistic Regression (Binary classification with sigmoid)
6. Multinomial Logistic Regression (Multi-class classification with softmax)

Provides in-depth reasoning and empirical evaluations for:
- Regularisation (coefficient shrinkage, feature selection/sparsity, L1 vs L2)
- Bias-Variance Trade-off (training vs test error curves across alpha, learning curves)
- Feature Scaling (impact of StandardScaler vs raw unscaled data on weights and loss)
"""
import warnings
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge

warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def get_available_targets(
    df: pd.DataFrame,
    pre: Optional[Dict[str, Any]] = None,
    result_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Identify continuous targets (for regression) and categorical targets (for classification)."""
    source_df = result_df if result_df is not None else df
    cols_to_check = [c for c in source_df.columns if c not in ["Record"]]

    id_cols = set((pre.get("id_columns") or []) if pre else [])
    # Also ignore generic id patterns
    valid_cols = [c for c in cols_to_check if c not in id_cols]

    regression_targets = []
    classification_targets = []

    for c in valid_cols:
        series = source_df[c].dropna()
        if len(series) < 10:
            continue

        is_numeric = pd.api.types.is_numeric_dtype(series)
        nunique = series.nunique()

        # Check classification candidate
        if 2 <= nunique <= 20:
            val_counts = series.value_counts().to_dict()
            unique_vals = [str(k) for k in val_counts.keys()]
            classification_targets.append({
                "column": c,
                "type": "binary" if nunique == 2 else "multinomial",
                "unique_count": int(nunique),
                "unique_values": unique_vals[:10],
                "is_dbscan_feature": c in ["Potential_Anomaly", "DBSCAN_Cluster", "Anomaly_Status"],
            })

        # Check regression candidate
        if is_numeric and nunique > 5:
            # Exclude obvious boolean flags or cluster integers
            if c not in ["Potential_Anomaly", "DBSCAN_Cluster", "Record"]:
                regression_targets.append({
                    "column": c,
                    "min": float(np.round(series.min(), 3)),
                    "max": float(np.round(series.max(), 3)),
                    "mean": float(np.round(series.mean(), 3)),
                    "std": float(np.round(series.std(), 3)),
                    "unique_count": int(nunique),
                    "is_dbscan_feature": c == "Anomaly_Score",
                })

    # Available predictor features (numeric only by default)
    numeric_features = [
        c for c in valid_cols
        if pd.api.types.is_numeric_dtype(source_df[c])
        and c not in ["Record", "Potential_Anomaly", "DBSCAN_Cluster"]
    ]

    # Pick smart defaults
    default_reg = regression_targets[0]["column"] if regression_targets else None
    # For classification, prefer 'Potential_Anomaly' or 'DBSCAN_Cluster' if available
    default_cls = None
    for ct in classification_targets:
        if ct["column"] == "Potential_Anomaly":
            default_cls = "Potential_Anomaly"
            break
        if ct["column"] == "DBSCAN_Cluster":
            default_cls = "DBSCAN_Cluster"
    if not default_cls and classification_targets:
        default_cls = classification_targets[0]["column"]

    return {
        "regression_targets": regression_targets,
        "classification_targets": classification_targets,
        "available_features": numeric_features,
        "default_regression_target": default_reg,
        "default_classification_target": default_cls,
    }


def _prepare_data(
    source_df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    task_type: str = "regression",
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Optional[LabelEncoder], pd.DataFrame]:
    """Clean missing values, isolate X and y, and perform train/test split."""
    if target_col not in source_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # Determine predictor features
    if not feature_cols:
        feature_cols = [
            c for c in source_df.columns
            if c != target_col
            and pd.api.types.is_numeric_dtype(source_df[c])
            and c not in ["Record", "Potential_Anomaly", "DBSCAN_Cluster", "Anomaly_Status"]
        ]
    else:
        feature_cols = [c for c in feature_cols if c in source_df.columns and c != target_col]

    if len(feature_cols) == 0:
        raise ValueError("At least 1 numerical feature is required for training.")

    # Prepare DataFrame without NaNs in target
    sub_df = source_df[[target_col] + feature_cols].copy()
    sub_df = sub_df.dropna(subset=[target_col])

    # Impute missing values in features with column median
    for col in feature_cols:
        if sub_df[col].isna().any():
            sub_df[col] = sub_df[col].fillna(sub_df[col].median())

    X = sub_df[feature_cols].values.astype(float)
    y_raw = sub_df[target_col].values

    label_encoder = None
    if task_type == "classification":
        # Ensure categorical labels are mapped to 0..K-1
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str))
        stratify = y if len(np.unique(y)) > 1 and np.min(np.bincount(y)) >= 2 else None
    else:
        y = y_raw.astype(float)
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    return X_train, X_test, y_train, y_test, feature_cols, label_encoder, sub_df


def run_linear_regression_suite(
    source_df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    test_size: float = 0.25,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    apply_scaling: bool = True,
) -> Dict[str, Any]:
    """Train and evaluate Linear Regression, Ridge, Lasso, and Elastic Net models.

    Computes regularisation coefficient shrinkage, bias-variance trade-off curves,
    and scaling impact side-by-side.
    """
    X_train, X_test, y_train, y_test, features, _, clean_df = _prepare_data(
        source_df, target_col, feature_cols, task_type="regression", test_size=test_size
    )

    # 1. Feature Scaling
    scaler = StandardScaler()
    if apply_scaling:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    # 2. Fit the 4 Linear Models
    models = {
        "Linear Regression (OLS)": LinearRegression(),
        "Ridge Regression (L2)": Ridge(alpha=max(alpha, 1e-6), random_state=42),
        "Lasso Regression (L1)": Lasso(alpha=max(alpha, 1e-4), max_iter=5000, random_state=42),
        "Elastic Net (L1+L2)": ElasticNet(
            alpha=max(alpha, 1e-4), l1_ratio=np.clip(l1_ratio, 0.01, 0.99), max_iter=5000, random_state=42
        ),
    }

    results = []
    fitted_models = {}

    for name, m in models.items():
        m.fit(X_train_scaled, y_train)
        fitted_models[name] = m

        y_train_pred = m.predict(X_train_scaled)
        y_test_pred = m.predict(X_test_scaled)

        r2_train = float(r2_score(y_train, y_train_pred))
        r2_test = float(r2_score(y_test, y_test_pred))
        mse_train = float(mean_squared_error(y_train, y_train_pred))
        mse_test = float(mean_squared_error(y_test, y_test_pred))
        mae_test = float(mean_absolute_error(y_test, y_test_pred))
        rmse_test = float(np.sqrt(mse_test))

        coefs = m.coef_.flatten()
        non_zero_count = int(np.sum(np.abs(coefs) > 1e-5))

        coef_dict = [
            {
                "feature": f,
                "coef": float(np.round(c, 4)),
                "abs_coef": float(np.round(abs(c), 4)),
                "is_zero": bool(abs(c) <= 1e-5),
            }
            for f, c in zip(features, coefs)
        ]

        results.append({
            "model_name": name,
            "train_r2": round(r2_train, 4),
            "test_r2": round(r2_test, 4),
            "train_mse": round(mse_train, 4),
            "test_mse": round(mse_test, 4),
            "test_rmse": round(rmse_test, 4),
            "test_mae": round(mae_test, 4),
            "intercept": round(float(m.intercept_), 4),
            "active_features": non_zero_count,
            "total_features": len(features),
            "sparsity_pct": round((1.0 - (non_zero_count / len(features))) * 100, 1),
            "coefficients": coef_dict,
        })

    # 3. Regularisation Path & Bias-Variance Analysis across Alphas
    # Alpha sweep spanning 10^-3 to 10^3
    alphas = np.logspace(-3, 3, 25)
    bias_variance_curve = []
    reg_path_ridge = []
    reg_path_lasso = []
    best_test_mse = float("inf")
    optimal_alpha = float(alphas[0])

    for a in alphas:
        # Ridge
        r_model = Ridge(alpha=a, random_state=42).fit(X_train_scaled, y_train)
        r_train_mse = float(mean_squared_error(y_train, r_model.predict(X_train_scaled)))
        r_test_mse = float(mean_squared_error(y_test, r_model.predict(X_test_scaled)))

        # Lasso
        l_model = Lasso(alpha=a, max_iter=3000, random_state=42).fit(X_train_scaled, y_train)
        l_train_mse = float(mean_squared_error(y_train, l_model.predict(X_train_scaled)))
        l_test_mse = float(mean_squared_error(y_test, l_model.predict(X_test_scaled)))

        if l_test_mse < best_test_mse:
            best_test_mse = l_test_mse
            optimal_alpha = float(a)

        bias_variance_curve.append({
            "alpha": round(float(a), 4),
            "log_alpha": round(float(np.log10(a)), 2),
            "ridge_train_mse": round(r_train_mse, 4),
            "ridge_test_mse": round(r_test_mse, 4),
            "lasso_train_mse": round(l_train_mse, 4),
            "lasso_test_mse": round(l_test_mse, 4),
        })

        # Record coefficients for top features in regularisation path
        reg_path_ridge.append({
            "alpha": round(float(a), 4),
            "log_alpha": round(float(np.log10(a)), 2),
            **{f: round(float(c), 4) for f, c in zip(features, r_model.coef_)},
        })
        reg_path_lasso.append({
            "alpha": round(float(a), 4),
            "log_alpha": round(float(np.log10(a)), 2),
            **{f: round(float(c), 4) for f, c in zip(features, l_model.coef_)},
        })

    # Current model bias-variance diagnosis
    curr_ridge_res = next(r for r in results if "Ridge" in r["model_name"])
    train_mse = curr_ridge_res["train_mse"]
    test_mse = curr_ridge_res["test_mse"]
    gen_gap = test_mse - train_mse

    if alpha < optimal_alpha * 0.1 and gen_gap > 0.2 * train_mse:
        bv_diagnosis = {
            "status": "Overfitting (High Variance)",
            "tone": "warn",
            "explanation": (
                f"With small regularisation (α={alpha}), the model fits the training set closely "
                f"(Train MSE: {train_mse}) but error on unseen data is higher (Test MSE: {test_mse}). "
                "Increasing α will regularise the weights and reduce variance."
            ),
        }
    elif alpha > optimal_alpha * 10:
        bv_diagnosis = {
            "status": "Underfitting (High Bias)",
            "tone": "warn",
            "explanation": (
                f"With excessive regularisation (α={alpha}), the penalty shrinks coefficients too close "
                "to zero, causing both training and test errors to rise. Relax α towards the optimal balance."
            ),
        }
    else:
        bv_diagnosis = {
            "status": "Optimal Bias-Variance Balance",
            "tone": "ok",
            "explanation": (
                f"Regularisation strength α={alpha} maintains balanced predictive accuracy across both "
                f"training and test partitions (Test MSE: {test_mse}). Generalization error is minimised."
            ),
        }

    # 4. Feature Scaling Comparison
    # Fit Ridge & Lasso without scaling to demonstrate scale sensitivity
    r_unscaled = Ridge(alpha=max(alpha, 1e-6), random_state=42).fit(X_train, y_train)
    l_unscaled = Lasso(alpha=max(alpha, 1e-4), max_iter=5000, random_state=42).fit(X_train, y_train)

    scaling_comparison = []
    r_scaled = fitted_models["Ridge Regression (L2)"]
    l_scaled = fitted_models["Lasso Regression (L1)"]

    # Calculate standard deviation of each raw feature to demonstrate scale disparity
    feature_stds = np.std(X_train, axis=0)

    for idx, f in enumerate(features):
        scaling_comparison.append({
            "feature": f,
            "std_dev": round(float(feature_stds[idx]), 3),
            "ridge_scaled_coef": round(float(r_scaled.coef_[idx]), 4),
            "ridge_unscaled_coef": round(float(r_unscaled.coef_[idx]), 4),
            "lasso_scaled_coef": round(float(l_scaled.coef_[idx]), 4),
            "lasso_unscaled_coef": round(float(l_unscaled.coef_[idx]), 4),
        })

    # Scaling metrics summary
    scaling_impact_summary = {
        "scaled_test_r2": curr_ridge_res["test_r2"],
        "unscaled_test_r2": round(float(r2_score(y_test, r_unscaled.predict(X_test))), 4),
        "scaled_test_mse": curr_ridge_res["test_mse"],
        "unscaled_test_mse": round(float(mean_squared_error(y_test, r_unscaled.predict(X_test))), 4),
        "max_scale_disparity_ratio": round(float(np.max(feature_stds) / max(np.min(feature_stds), 1e-6)), 1),
    }

    # 5. Learning Curve (Sample size vs Train/Validation Error)
    train_sizes, train_scores, test_scores = learning_curve(
        Ridge(alpha=alpha, random_state=42),
        X_train_scaled,
        y_train,
        train_sizes=np.linspace(0.2, 1.0, 5),
        scoring="neg_mean_squared_error",
        cv=3,
        random_state=42,
    )
    learning_curve_data = [
        {
            "sample_size": int(sz),
            "pct": int(round((sz / len(X_train)) * 100)),
            "train_mse": round(float(-np.mean(tr)), 4),
            "val_mse": round(float(-np.mean(te)), 4),
        }
        for sz, tr, te in zip(train_sizes, train_scores, test_scores)
    ]

    # Predictions sample for actual vs predicted scatter
    best_model_name = max(results, key=lambda x: x["test_r2"])["model_name"]
    best_model = fitted_models[best_model_name]
    y_test_pred_best = best_model.predict(X_test_scaled)
    preview_count = min(len(y_test), 30)
    predictions_preview = [
        {
            "id": i + 1,
            "actual": round(float(y_test[i]), 3),
            "predicted": round(float(y_test_pred_best[i]), 3),
            "residual": round(float(y_test[i] - y_test_pred_best[i]), 3),
        }
        for i in range(preview_count)
    ]

    return {
        "task_type": "regression",
        "target_column": target_col,
        "feature_columns": features,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "models": results,
        "best_model_name": best_model_name,
        "bias_variance": {
            "curve": bias_variance_curve,
            "optimal_alpha": round(optimal_alpha, 4),
            "current_alpha": float(alpha),
            "diagnosis": bv_diagnosis,
            "learning_curve": learning_curve_data,
        },
        "regularisation": {
            "ridge_path": reg_path_ridge,
            "lasso_path": reg_path_lasso,
            "alpha": float(alpha),
            "l1_ratio": float(l1_ratio),
        },
        "feature_scaling": {
            "applied": apply_scaling,
            "comparison": scaling_comparison,
            "summary": scaling_impact_summary,
        },
        "predictions_preview": predictions_preview,
    }


def run_logistic_regression_suite(
    source_df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    test_size: float = 0.25,
    C: float = 1.0,
    l1_ratio: float = 0.5,
    apply_scaling: bool = True,
) -> Dict[str, Any]:
    """Train and evaluate Binary and Multinomial Logistic Regression models.

    Compares unregularized, L2 (Ridge), L1 (Lasso), and Elastic Net penalties.
    Analyzes regularisation, bias-variance trade-off across C, and feature scaling.
    """
    X_train, X_test, y_train, y_test, features, label_enc, clean_df = _prepare_data(
        source_df, target_col, feature_cols, task_type="classification", test_size=test_size
    )

    classes = [str(c) for c in label_enc.classes_]
    n_classes = len(classes)
    is_binary = n_classes == 2

    # 1. Feature Scaling
    scaler = StandardScaler()
    if apply_scaling:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    # 2. Configure Models
    # For binary: standard Logistic Regression with sigmoid
    # For multiclass (>2): Multinomial Logistic Regression with softmax
    multi_class_param = "multinomial" if not is_binary else "auto"

    model_configs = [
        {
            "name": "Logistic Regression (No Penalty)" if is_binary else "Multinomial Logistic (No Penalty)",
            "clf": LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000, random_state=42),
            "desc": "Unregularized maximum likelihood estimation (standard linear boundary)",
        },
        {
            "name": "Logistic Regression (L2 / Ridge)" if is_binary else "Multinomial Logistic (L2 / Ridge)",
            "clf": LogisticRegression(
                penalty="l2", C=max(C, 1e-4), solver="lbfgs", max_iter=1000, random_state=42
            ),
            "desc": "L2 regularization: shrinks log-odds coefficients smoothly",
        },
        {
            "name": "Logistic Regression (L1 / Lasso)" if is_binary else "Multinomial Logistic (L1 / Lasso)",
            "clf": LogisticRegression(
                penalty="l1", C=max(C, 1e-4), solver="saga", max_iter=2000, random_state=42
            ),
            "desc": "L1 regularization: drives non-informative biomarker coefficients to 0",
        },
        {
            "name": "Logistic Regression (Elastic Net)" if is_binary else "Multinomial Logistic (Elastic Net)",
            "clf": LogisticRegression(
                penalty="elasticnet",
                C=max(C, 1e-4),
                l1_ratio=np.clip(l1_ratio, 0.01, 0.99),
                solver="saga",
                max_iter=2000,
                random_state=42,
            ),
            "desc": "Elastic Net: balanced compromise between sparsity (L1) and stability (L2)",
        },
    ]

    results = []
    fitted_models = {}

    for cfg in model_configs:
        name = cfg["name"]
        clf = cfg["clf"]
        clf.fit(X_train_scaled, y_train)
        fitted_models[name] = clf

        y_train_pred = clf.predict(X_train_scaled)
        y_test_pred = clf.predict(X_test_scaled)

        train_acc = float(accuracy_score(y_train, y_train_pred))
        test_acc = float(accuracy_score(y_test, y_test_pred))

        avg_type = "binary" if is_binary else "weighted"
        precision = float(precision_score(y_test, y_test_pred, average=avg_type, zero_division=0))
        recall = float(recall_score(y_test, y_test_pred, average=avg_type, zero_division=0))
        f1 = float(f1_score(y_test, y_test_pred, average=avg_type, zero_division=0))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred).tolist()

        # Log-loss
        y_test_proba = clf.predict_proba(X_test_scaled)
        try:
            test_loss = float(log_loss(y_test, y_test_proba, labels=list(range(n_classes))))
        except Exception:
            test_loss = None

        # ROC AUC if binary
        auc_score = None
        if is_binary and len(np.unique(y_test)) == 2:
            try:
                auc_score = float(roc_auc_score(y_test, y_test_proba[:, 1]))
            except Exception:
                auc_score = None

        # Extract Coefficients and Odds Ratios
        # For binary, clf.coef_ shape is (1, n_features)
        # For multiclass, shape is (n_classes, n_features)
        coef_dict = []
        if is_binary:
            coefs = clf.coef_[0]
            non_zero = int(np.sum(np.abs(coefs) > 1e-5))
            coef_dict = [
                {
                    "feature": f,
                    "coef": float(np.round(c, 4)),
                    "odds_ratio": float(np.round(np.exp(np.clip(c, -10, 10)), 3)),
                    "abs_coef": float(np.round(abs(c), 4)),
                    "is_zero": bool(abs(c) <= 1e-5),
                }
                for f, c in zip(features, coefs)
            ]
        else:
            # Multiclass: summarize mean absolute weight per feature
            mean_abs_coefs = np.mean(np.abs(clf.coef_), axis=0)
            non_zero = int(np.sum(mean_abs_coefs > 1e-5))
            coef_dict = [
                {
                    "feature": f,
                    "coef": float(np.round(mean_abs_coefs[idx], 4)),
                    "abs_coef": float(np.round(mean_abs_coefs[idx], 4)),
                    "is_zero": bool(mean_abs_coefs[idx] <= 1e-5),
                }
                for idx, f in enumerate(features)
            ]

        results.append({
            "model_name": name,
            "description": cfg["desc"],
            "train_accuracy": round(train_acc * 100, 2),
            "test_accuracy": round(test_acc * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "log_loss": round(test_loss, 4) if test_loss is not None else None,
            "roc_auc": round(auc_score, 4) if auc_score is not None else None,
            "active_features": non_zero,
            "total_features": len(features),
            "sparsity_pct": round((1.0 - (non_zero / len(features))) * 100, 1),
            "confusion_matrix": cm,
            "coefficients": coef_dict,
        })

    # 3. Bias-Variance & Regularisation Path across C (Inverse of Lambda)
    # When C is large -> low regularisation (risk of overfitting / high variance)
    # When C is small -> strong regularisation (risk of underfitting / high bias)
    c_values = np.logspace(-3, 3, 20)
    bias_variance_curve = []
    reg_path_l2 = []
    reg_path_l1 = []
    best_acc = -1.0
    optimal_C = 1.0

    for c_val in c_values:
        # L2
        clf_l2 = LogisticRegression(penalty="l2", C=c_val, solver="lbfgs", max_iter=1000, random_state=42)
        clf_l2.fit(X_train_scaled, y_train)
        l2_tr_acc = float(accuracy_score(y_train, clf_l2.predict(X_train_scaled)))
        l2_te_acc = float(accuracy_score(y_test, clf_l2.predict(X_test_scaled)))

        # L1
        clf_l1 = LogisticRegression(penalty="l1", C=c_val, solver="saga", max_iter=1000, random_state=42)
        clf_l1.fit(X_train_scaled, y_train)
        l1_tr_acc = float(accuracy_score(y_train, clf_l1.predict(X_train_scaled)))
        l1_te_acc = float(accuracy_score(y_test, clf_l1.predict(X_test_scaled)))

        if l2_te_acc > best_acc:
            best_acc = l2_te_acc
            optimal_C = float(c_val)

        bias_variance_curve.append({
            "C": round(float(c_val), 4),
            "log_C": round(float(np.log10(c_val)), 2),
            "l2_train_acc": round(l2_tr_acc * 100, 2),
            "l2_test_acc": round(l2_te_acc * 100, 2),
            "l1_train_acc": round(l1_tr_acc * 100, 2),
            "l1_test_acc": round(l1_te_acc * 100, 2),
        })

        # Top feature coefficients along path
        if is_binary:
            reg_path_l2.append({
                "C": round(float(c_val), 4),
                "log_C": round(float(np.log10(c_val)), 2),
                **{f: round(float(c), 4) for f, c in zip(features, clf_l2.coef_[0])},
            })
            reg_path_l1.append({
                "C": round(float(c_val), 4),
                "log_C": round(float(np.log10(c_val)), 2),
                **{f: round(float(c), 4) for f, c in zip(features, clf_l1.coef_[0])},
            })

    # Bias-variance diagnosis for current C
    curr_l2 = next(r for r in results if "L2 / Ridge" in r["model_name"])
    tr_acc = curr_l2["train_accuracy"]
    te_acc = curr_l2["test_accuracy"]
    acc_gap = tr_acc - te_acc

    if C > optimal_C * 10 and acc_gap > 12:
        bv_diagnosis = {
            "status": "Overfitting (High Variance)",
            "tone": "warn",
            "explanation": (
                f"With large C={C} (very weak regularisation), the logistic decision boundary twists "
                f"to fit training samples (Train Acc: {tr_acc}%) but generalises poorly (Test Acc: {te_acc}%). "
                "Lower C to increase penalty and reduce variance."
            ),
        }
    elif C < optimal_C * 0.1 and te_acc < (tr_acc - 5):
        bv_diagnosis = {
            "status": "Underfitting (High Bias)",
            "tone": "warn",
            "explanation": (
                f"With small C={C} (excessive regularisation), the model's coefficients are excessively "
                f"shrunk, over-simplifying the decision boundary (Test Acc: {te_acc}%). Increase C."
            ),
        }
    else:
        bv_diagnosis = {
            "status": "Optimal Bias-Variance Balance",
            "tone": "ok",
            "explanation": (
                f"Parameter C={C} achieves a strong compromise between training precision and test "
                f"generalization (Test Accuracy: {te_acc}%). Regularization is well-calibrated."
            ),
        }

    # 4. Feature Scaling Comparison
    clf_unscaled = LogisticRegression(
        penalty="l2", C=max(C, 1e-4), solver="lbfgs", max_iter=1000, random_state=42
    )
    clf_unscaled.fit(X_train, y_train)
    unscaled_acc = float(accuracy_score(y_test, clf_unscaled.predict(X_test)))

    feature_stds = np.std(X_train, axis=0)
    scaling_comparison = []

    l2_scaled_clf = fitted_models[curr_l2["model_name"]]

    for idx, f in enumerate(features):
        scaled_c = float(l2_scaled_clf.coef_[0][idx] if is_binary else np.mean(np.abs(l2_scaled_clf.coef_[:, idx])))
        unscaled_c = float(clf_unscaled.coef_[0][idx] if is_binary else np.mean(np.abs(clf_unscaled.coef_[:, idx])))
        scaling_comparison.append({
            "feature": f,
            "std_dev": round(float(feature_stds[idx]), 3),
            "scaled_coef": round(scaled_c, 4),
            "unscaled_coef": round(unscaled_c, 4),
        })

    scaling_impact_summary = {
        "scaled_test_accuracy": curr_l2["test_accuracy"],
        "unscaled_test_accuracy": round(unscaled_acc * 100, 2),
        "max_scale_disparity_ratio": round(float(np.max(feature_stds) / max(np.min(feature_stds), 1e-6)), 1),
    }

    best_model_name = max(results, key=lambda x: x["test_accuracy"])["model_name"]

    return {
        "task_type": "classification",
        "sub_type": "binary" if is_binary else "multinomial",
        "target_column": target_col,
        "classes": classes,
        "feature_columns": features,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "models": results,
        "best_model_name": best_model_name,
        "bias_variance": {
            "curve": bias_variance_curve,
            "optimal_C": round(optimal_C, 4),
            "current_C": float(C),
            "diagnosis": bv_diagnosis,
        },
        "regularisation": {
            "l2_path": reg_path_l2,
            "l1_path": reg_path_l1,
            "C": float(C),
            "l1_ratio": float(l1_ratio),
        },
        "feature_scaling": {
            "applied": apply_scaling,
            "comparison": scaling_comparison,
            "summary": scaling_impact_summary,
        },
    }
