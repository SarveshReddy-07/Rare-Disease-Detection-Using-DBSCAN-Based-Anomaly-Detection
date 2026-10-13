# RARE DISEASE DETECTION USING DBSCAN-BASED ANOMALY DETECTION

**Course:** Machine Learning (B.Tech CSE)  
**Academic Year:** 2026–2027  
**Department:** Department of Computer Science and Engineering  
**Institution:** Koneru Lakshmaiah Education Foundation (KLH Hyderabad Off-Campus, Bowrampet, Hyderabad, Telangana – 500 043)  

### Team Members:
- **Kodeboina Priyadarshini** (2520030067)
- **Sambavaram Sarveswara Reddy** (2520030049)
- **Jampani Poornesh Kumar** (2520030055)

**Project Guides:**
- **G. Ravindra Babu**, Assistant Professor, Department of Computer Science and Engineering  
- **Dr. Nirmala Jyothi**, Associate Professor, Department of Computer Science and Engineering  

---

> **Academic & Clinical Disclaimer:**  
> This project is developed strictly for research and educational purposes as part of the B.Tech Machine Learning curriculum. Density-based spatial clustering and anomaly scoring highlight unusual medical records within an uploaded sample cohort for prioritization. They do not constitute a medical diagnosis and must never supersede qualified medical evaluation.

---

## 1. Executive Summary

Rare diseases pose an extraordinary challenge in modern healthcare: individually they affect fewer than 1 in 2,000 individuals, but collectively they afflict hundreds of millions worldwide. Because their symptoms frequently mimic common ailments and verified positive training records are vanishingly scarce, traditional **supervised classification models fail due to severe class imbalance or complete lack of ground-truth labels**.

This project presents a full-stack, research-grade machine learning platform that reframes rare disease identification as an **Unsupervised Anomaly Detection problem using DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**. By learning the high-dimensional density topology of typical patient biomarker profiles, DBSCAN segregates dense, normative physiological cohorts into arbitrary-shaped clusters and identifies isolated data points in low-density regions as **Noise Points (Cluster ID `-1`)**. These noise points represent prospective rare disease candidates, atypical pathological manifestations, or anomalous clinical profiles requiring accelerated clinical investigation.

### Core Modules & Algorithmic Foundations:
1. **Adaptive Ingestion & Preprocessing Pipeline:** Automated delimiter/encoding sniffing, missing value imputation (median for continuous metrics, most-frequent for categorical fields), one-hot encoding, and $Z$-score feature standardisation (`StandardScaler`).
2. **Deterministic Parameter Selection (Elbow Heuristic):** Automated $k$-nearest neighbour distance graph generation and maximum curvature elbow detection to objectively select the critical neighbourhood radius parameter $\varepsilon$ (Epsilon) alongside density threshold `min_samples`.
3. **Density-Based Spatial Clustering & Anomaly Scoring:** Scikit-learn DBSCAN execution over the multi-dimensional scaled biomarker matrix, complemented by a continuous $k$-NN Euclidean anomaly isolation score to quantify the severity of patient isolation.
4. **Manifold Visualization & Dimensionality Reduction:** Principal Component Analysis (PCA) 2D/3D projections preserving maximum biomarker variance, interactive cluster/noise scatter plots, and cluster density distribution profiles.
5. **Collinearity & Biomarker Association Analysis:** Full-rank Pearson correlation heatmaps, interactive 2D regression scatter plots with trendline slope diagnostics, and point-biserial biomarker anomaly association rankings.
6. **Multi-Criterion Diagnostic Validation (Confusion Matrix):** Ground-truth benchmarking and clinical criterion proxy evaluation (calculating True Positive Rate / Sensitivity, Specificity, Precision, F1-Score, and False Negative analysis).
7. **Supervised Linear Models & Deep Theoretical Reasoning:**
   - **Continuous Biomarker Regression:** Ordinary Least Squares (OLS), Ridge Regression ($L_2$), Lasso Regression ($L_1$), and Elastic Net ($L_1 + L_2$).
   - **Categorical Clinical Classification:** Binary Logistic Regression and Multinomial Logistic Regression (multiclass softmax).
   - **Regularisation Geometry & Feature Sparsity:** Geometric proof and empirical visualisation of $L_1$ diamond constraint corners inducing exact zero coefficients vs $L_2$ circle smooth spherical shrinkage.
   - **Bias-Variance Trade-off & Learning Diagnostics:** Empirical error curves across penalty strengths ($\alpha$ / $C$), learning curves over training subset sizes, and automated regime diagnosis (High Variance / Overfitting vs Optimal vs High Bias / Underfitting).
   - **Feature Scaling Sensitivity:** Side-by-side empirical verification of models trained with $Z$-score scaling vs raw unscaled inputs, demonstrating why disparate clinical units bias Euclidean distance metrics and gradient descent conditioning.
8. **Real-Time Patient Profile Assessment:** Inference engine comparing newly inputted patient clinical parameters against core cluster densities to compute a comparative dataset-similarity anomaly factor.

---

## 2. Real-World Clinical Challenges & ML Algorithm Mapping

| Clinical Healthcare Challenge | Traditional ML Limitation | Algorithm Applied | Project Module | Complexity |
|---|---|---|---|---|
| **Label Scarcity in Rare Diseases** | Supervised classifiers (Random Forest, SVM) overfit or collapse on extreme class imbalance ($< 0.1\%$ positives). | **Unsupervised DBSCAN (Density-Based Spatial Clustering)** | `ml/dbscan_model.py`<br>`ml/anomaly_detection.py` | Time: $O(n^2)$ (or $O(n \log n)$ spatial)<br>Space: $O(n)$ |
| **Heuristic Parameter Guesswork** | Arbitrary choice of $\varepsilon$ results in either a single giant cluster or 100% noise points. | **$k$-Distance Graph + Knee/Elbow Curvature Heuristic** | `ml/dbscan_model.py` | Time: $O(n \cdot k \log n)$<br>Space: $O(n)$ |
| **Grading Severity of Outliers** | Binary `-1` noise classification does not distinguish mild outliers from extreme anomalies. | **$k$-th Nearest Neighbour Anomaly Distance Score** | `ml/anomaly_detection.py` | Time: $O(n \cdot k)$<br>Space: $O(n)$ |
| **High-Dimensional Biomarker Inspection** | Clinicians cannot visualize $>3$ physiological dimensions simultaneously. | **Principal Component Analysis (PCA)** | `ml/visualization.py` | Time: $O(\min(n^3, d^3))$<br>Space: $O(n \cdot d)$ |
| **Redundant Biomarker Collinearity** | Hidden collinear features distort Euclidean distance metrics, overweighting correlated markers. | **Pearson Correlation Matrix & 2D Linear Regression** | `ml/correlation.py` | Time: $O(d^2 \cdot n)$<br>Space: $O(d^2)$ |
| **Validation Without Pure Labels** | Unsupervised models cannot be verified with standard accuracy metrics alone. | **Multi-Source Confusion Matrix (Sensitivity, Specificity, F1)** | `ml/evaluation.py` | Time: $O(n)$<br>Space: $O(1)$ |
| **Biomarker Overfitting & Noisy Features** | Unconstrained linear models assign non-zero weights to irrelevant lab noise. | **Lasso ($L_1$) & Ridge ($L_2$) Regularisation** | `ml/linear_models.py` | Time: $O(n \cdot d^2)$<br>Space: $O(d)$ |
| **Lab Unit Disparity (pg/mL vs cells/mcL)** | Disparate scales allow large-magnitude variables (e.g. Platelets: $250,000$) to completely dominate small biomarkers (e.g. TSH: $2.5$). | **StandardScaler ($Z$-Score Normalisation)** | `ml/preprocessing.py` | Time: $O(n \cdot d)$<br>Space: $O(d)$ |

---

## 3. Project Architecture & Folder Structure

The project is structured as a decoupled full-stack architecture. The backend exposes high-performance asynchronous REST endpoints via FastAPI and scikit-learn, while the frontend provides an interactive, reactive clinical dashboard built with React 18, Vite, and Recharts.

```text
rare-disease-dbscan/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py                 # FastAPI endpoints (upload, preprocess, dbscan, linear models, etc.)
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── preprocessing.py          # Encoding/delimiter sniffing, imputation, StandardScaler
│   │   ├── dbscan_model.py           # k-distance elbow heuristic & DBSCAN clustering
│   │   ├── anomaly_detection.py      # Noise point identification & k-NN anomaly scoring
│   │   ├── visualization.py          # PCA 2D/3D projections & distribution charts
│   │   ├── correlation.py            # Pearson correlation matrix & 2D regression scatter
│   │   ├── evaluation.py             # Diagnostic confusion matrix, Sensitivity, Specificity, F1
│   │   └── linear_models.py          # OLS, Ridge, Lasso, Elastic Net, Logistic Regression suites
│   ├── sample_data/
│   │   └── DEMO_synthetic_medical_profiles.csv   # 500-patient synthetic clinical benchmark dataset
│   ├── make_demo_data.py             # Synthetic physiological cohort generator
│   ├── main.py                       # FastAPI application entry point & CORS configuration
│   └── requirements.txt              # Python runtime dependencies
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Layout.jsx            # Clinical workflow stepper & navigation header
│   │   │   ├── Card.jsx              # Reusable UI card container
│   │   │   ├── Table.jsx             # Paginated clinical data grid
│   │   │   └── StatCard.jsx          # Metric cards for key performance indicators
│   │   ├── pages/
│   │   │   ├── Upload.jsx            # CSV drag-and-drop & demo dataset loader
│   │   │   ├── Inspect.jsx           # Feature type detection & ID/label role assignment
│   │   │   ├── Preprocess.jsx        # Transformation pipeline preview & scaled matrix stats
│   │   │   ├── Config.jsx            # Interactive k-distance graph & EPS/MinPts parameter tuner
│   │   │   ├── Dashboard.jsx         # Summary cards, PCA visualizer, cluster overview
│   │   │   ├── Correlation.jsx       # Pearson correlation heatmap & 2D scatter regressions
│   │   │   ├── ConfusionMatrix.jsx   # Clinical validation metrics & benchmark comparisons
│   │   │   ├── LinearModels.jsx      # Supervised models, regularisation, bias-variance & scaling
│   │   │   ├── Anomalies.jsx         # Searchable anomaly table & CSV export
│   │   │   ├── PatientAssessment.jsx # Single-patient profile assessment tool
│   │   │   └── About.jsx             # Academic documentation & workflow explanation
│   │   ├── charts/
│   │   │   ├── BiasVarianceChart.jsx # Interactive trade-off & learning curve graphs
│   │   │   ├── CoefficientsChart.jsx # Shrinkage paths & L1 diamond vs L2 circle visualization
│   │   │   └── ScalingImpactChart.jsx# Empirical Scaled vs Unscaled performance comparison
│   │   ├── services/
│   │   │   └── api.js                # Centralized Axios/fetch REST client
│   │   ├── App.jsx                   # React Router routing table
│   │   └── index.css                 # Custom clinical design system & responsive styling
│   ├── index.html                    # Application entry HTML
│   ├── package.json                  # Node.js dependencies & scripts
│   └── vite.config.js                # Vite build and development configuration
├── Ml Abstract.pdf                   # Submitted B.Tech ML Project Abstract (KL University)
└── README.md                         # Comprehensive project documentation
```

---

## 4. Feature Details & Theoretical Foundations

### Feature 1: Clinical Preprocessing & $Z$-Score Standardization
- **Problem:** Clinical data arrives with disparate units (e.g., White Blood Cell counts in thousands vs. Serum Ferritin in ng/mL), missing lab values, and mixed data types. In Euclidean distance-based algorithms, unscaled high-magnitude variables dominate spatial distances.
- **Formulation:** Continuous features are standardized using the $Z$-score transformation:
  $$z = \frac{x - \mu}{\sigma}$$
  where $\mu$ is the empirical feature mean and $\sigma$ is the standard deviation. Missing numerical entries undergo median imputation to resist outlier skewness; categorical columns undergo one-hot encoding after high-cardinality and free-text screening.

### Feature 2: Deterministic Parameter Selection ($k$-Distance Graph & Elbow Heuristic)
- **Problem:** DBSCAN requires two parameters: neighbourhood radius $\varepsilon$ and minimum points `min_samples` ($MinPts$). Selecting $\varepsilon$ manually leads to trial-and-error bias.
- **Algorithm:**
  1. Set $k = MinPts = \max(4, 2 \times d)$ where $d$ is the number of active features.
  2. Compute the Euclidean distance from every point $p_i$ to its $k$-th nearest neighbour:
     $$d_k(p_i) = \text{dist}(p_i, \text{NN}_k(p_i))$$
  3. Sort all $d_k(p_i)$ in ascending order and plot the resulting 1D curve.
  4. Detect the knee/elbow point using the maximum perpendicular distance from the line connecting the curve endpoints $(x_1, y_1)$ to $(x_n, y_n)$:
     $$d_{\perp}(i) = \frac{|(y_n - y_1) x_i - (x_n - x_1) y_i + x_n y_1 - y_n x_1|}{\sqrt{(y_n - y_1)^2 + (x_n - x_1)^2}}$$
  5. The point $i^* = \arg\max_i d_{\perp}(i)$ yields the optimal $\varepsilon^*$.

### Feature 3: DBSCAN Density Clustering & $k$-NN Anomaly Scoring
- **Formal Definitions:**
  - **$\varepsilon$-Neighbourhood:** $N_\varepsilon(p) = \{q \in D \mid \text{dist}(p, q) \le \varepsilon\}$
  - **Core Point:** A point $p$ is a core point if $|N_\varepsilon(p)| \ge MinPts$.
  - **Direct Density-Reachability:** $q$ is directly density-reachable from $p$ if $p$ is a core point and $q \in N_\varepsilon(p)$.
  - **Density-Reachability & Connectivity:** Transitive chain of direct reachability; clusters are maximal density-connected components.
  - **Noise Point (Anomaly):**
    $$\text{Noise} = \{p \in D \mid \forall C_j, p \notin C_j\} \iff \text{Cluster Label } = -1$$
- **Continuous Anomaly Severity Score:** To rank noise severity, every patient receives a score equal to their Euclidean distance to the $k$-th nearest neighbour in scaled feature space:
  $$\text{Anomaly Score}(p) = d_k(p) = \|\mathbf{z}_p - \mathbf{z}_{\text{NN}_k(p)}\|_2$$
  Higher scores represent patients who reside further into isolated, sparse regions of the physiological space.

### Feature 4: Principal Component Analysis (PCA) Dimensionality Reduction
- **Problem:** Visualizing multidimensional patient clusters in 2D and 3D without distorting core cluster boundaries.
- **Mathematical Derivation:**
  Given centered data matrix $\mathbf{Z} \in \mathbb{R}^{n \times d}$, compute the sample covariance matrix:
  $$\mathbf{\Sigma} = \frac{1}{n - 1} \mathbf{Z}^T \mathbf{Z}$$
  Perform eigendecomposition $\mathbf{\Sigma} \mathbf{v}_i = \lambda_i \mathbf{v}_i$, sorted such that $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d$. Project data onto top $k=2$ or $k=3$ eigenvectors:
  $$\mathbf{Z}_{\text{proj}} = \mathbf{Z} \mathbf{V}_k$$
  The explained variance ratio for component $i$ is given by $\frac{\lambda_i}{\sum_{j=1}^d \lambda_j}$.

### Feature 5: Biomarker Correlation Matrix & Linear Association
- **Problem:** Identifying pairs of collinear clinical biomarkers and quantifying which physiological indicators correlate most strongly with anomalous status.
- **Formulation:** Pearson correlation coefficient between features $X$ and $Y$:
  $$r_{XY} = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^n (y_i - \bar{y})^2}}$$
  Point-biserial correlation is computed between continuous biomarkers and binary anomaly status to generate an automated ranking of biomarkers associated with rare profile emergence.

### Feature 6: Diagnostic Confusion Matrix & Clinical Validation Metrics
- **Problem:** Evaluating unsupervised anomaly detection when ground-truth labels, clinical diagnostic thresholds, or multi-model consensus markers are available.
- **Metrics Calculated:**
  - **Sensitivity (Recall / True Positive Rate):** $\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}$ (Ability to identify true rare disease cases)
  - **Specificity (True Negative Rate):** $\text{TNR} = \frac{\text{TN}}{\text{TN} + \text{FP}}$ (Ability to correctly flag normal profiles)
  - **Precision (Positive Predictive Value):** $\text{PPV} = \frac{\text{TP}}{\text{TP} + \text{FP}}$ (Proportion of flagged anomalies that are genuine cases)
  - **F1-Score:** Harmonic mean balancing sensitivity and precision:
    $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$$

---

### Feature 7: Supervised Linear Regression Suite & Regularisation

#### Continuous Formulation
For predicting quantitative clinical targets (e.g. inflammatory index, viral load, recovery duration):
$$\hat{y} = \mathbf{w}^T \mathbf{x} + b$$

1. **Ordinary Least Squares (OLS):**
   $$\min_{\mathbf{w}, b} \frac{1}{2n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 \implies \mathbf{w}_{\text{OLS}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$
2. **Ridge Regression ($L_2$ Regularisation):**
   $$\min_{\mathbf{w}, b} \frac{1}{2n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \alpha \|\mathbf{w}\|_2^2 = \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \alpha \sum_{j=1}^d w_j^2$$
   Analytical solution: $\mathbf{w}_{\text{Ridge}} = (\mathbf{X}^T \mathbf{X} + 2n\alpha \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$. Shrinks all coefficients smoothly toward zero but never forces them to exact zero.
3. **Lasso Regression ($L_1$ Regularisation):**
   $$\min_{\mathbf{w}, b} \frac{1}{2n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \alpha \|\mathbf{w}\|_1 = \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \alpha \sum_{j=1}^d |w_j|$$
4. **Elastic Net ($L_1 + L_2$ Compromise):**
   $$\min_{\mathbf{w}, b} \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \alpha \cdot \rho \|\mathbf{w}\|_1 + \frac{\alpha (1 - \rho)}{2} \|\mathbf{w}\|_2^2$$
   where $\rho \in [0, 1]$ is the `l1_ratio`. Combines Lasso feature selection with Ridge handling of multicollinear biomarker groups.

#### Geometric Proof of Lasso Sparsity vs. Ridge Shrinkage
In constrained optimization, minimizing the quadratic loss under regularisation corresponds to finding the point of tangency between the elliptical loss contours $\frac{1}{2n}\|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 = c$ and the constraint region:
- **$L_1$ Constraint Region ($\sum |w_j| \le t$):** Forms a diamond (polytope with sharp vertices on the coordinate axes in $\mathbb{R}^d$). Because the corners lie directly on the axes where one or more $w_j = 0$, the expanding elliptical loss contours are geometrically much more likely to intersect a sharp corner first. Hence, **Lasso sets non-informative coefficients strictly to zero (automatic feature selection)**.
- **$L_2$ Constraint Region ($\sum w_j^2 \le t$):** Forms a smooth hypersphere (circle in $\mathbb{R}^2$). The boundary has no corners, so the point of tangency almost never occurs exactly on a coordinate axis. Thus, **Ridge shrinks weights continuously without setting any to zero**.

---

### Feature 8: Logistic & Multinomial Classification

#### Binary Logistic Regression
For binary clinical targets (e.g. `Anomaly_Status`, disease positive vs negative):
$$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
Optimized via regularized negative log-likelihood (binary cross-entropy):
$$\mathcal{L}(\mathbf{w}) = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \ln \hat{p}_i + (1 - y_i) \ln(1 - \hat{p}_i) \right] + \frac{1}{2C} \|\mathbf{w}\|_2^2$$
where $C = \frac{1}{\lambda}$ is the inverse regularisation strength.

#### Multinomial Logistic Regression (Softmax)
For multiclass clinical conditions ($K \ge 3$ distinct diagnostic states):
$$P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$$
Loss function: Multiclass cross-entropy with $L_2$ or $L_1$ penalty across all class weight vectors.

---

### Feature 9: Bias-Variance Decomposition & Learning Diagnostics

#### Mathematical Foundation
For any supervised estimator $\hat{f}(\mathbf{x})$ trained on dataset $\mathcal{D}$, the expected generalization mean squared error decomposes into three additive components:
$$\mathbb{E}_{\mathcal{D}}\left[(y - \hat{f}(\mathbf{x}))^2\right] = \underbrace{\left(\text{Bias}[\hat{f}(\mathbf{x})]\right)^2}_{\text{Underfitting}} + \underbrace{\text{Var}[\hat{f}(\mathbf{x})]}_{\text{Overfitting}} + \underbrace{\sigma^2}_{\text{Irreducible Noise}}$$
where:
$$\text{Bias}[\hat{f}(\mathbf{x})] = \mathbb{E}[\hat{f}(\mathbf{x})] - f(\mathbf{x}), \quad \text{Var}[\hat{f}(\mathbf{x})] = \mathbb{E}\left[(\hat{f}(\mathbf{x}) - \mathbb{E}[\hat{f}(\mathbf{x})])^2\right]$$

#### Diagnostic Regimes Automated by the System:
1. **High Variance (Overfitting):**
   - **Characteristics:** $\text{Train Error} \ll \text{Test Error}$ (Large Generalization Gap $\Delta > 15\%$).
   - **Root Cause:** Model has high complexity; it memorizes training noise and biomarker idiosyncrasies.
   - **Prescription:** Increase regularisation penalty ($\uparrow \alpha$ in regression, $\downarrow C$ in classification), prune features with Lasso, or acquire more training records.
2. **Optimal Trade-off:**
   - **Characteristics:** $\text{Test Error}$ reaches a global minimum; Generalization Gap is small and stable.
3. **High Bias (Underfitting):**
   - **Characteristics:** Both $\text{Train Error}$ and $\text{Test Error}$ are high; increasing data size yields no further improvement.
   - **Root Cause:** Model is excessively constrained by overly strong regularisation ($\alpha \to \infty$) or cannot represent the underlying relationship.
   - **Prescription:** Relax regularisation penalty ($\downarrow \alpha$, $\uparrow C$) or introduce non-linear interactions.

---

### Feature 10: Empirical Feature Scaling Impact Analysis

The application runs a controlled experiment evaluating models trained **With StandardScaler** vs **Without Scaling (Raw Features)**:
- **Gradient Conditioning:** Unscaled features create elongated, elliptical loss surfaces. Gradient descent oscillates orthogonal to narrow valleys, resulting in slow convergence or numerical instability.
- **Regularisation Fairness:** Penalties $\alpha \sum w_j^2$ or $\alpha \sum |w_j|$ penalize all coefficients equally. If feature $X_1$ is measured in small units (e.g. 0.001 to 0.005), its true physical weight $w_1$ must be large. Regularisation disproportionately shrinks $w_1$ simply because of its measurement scale. Standardisation ensures all biomarkers compete equally.

---

## 5. Deliberately Excluded Machine Learning Approaches

To maintain a rigorous, methodologically sound clinical project and avoid flawed design patterns during academic evaluation, the following ML approaches were deliberately excluded:

| Excluded Approach | Technical Reason for Deliberate Exclusion |
|---|---|
| **K-Means Clustering** | Assumes spherical, convex clusters of equal variance. Requires pre-specifying $k$ (which is unknown in rare disease discovery). Most critically, **K-Means has no concept of noise** and forces rare anomaly points into normal patient clusters, corrupting centroids. |
| **Supervised Deep Neural Networks** | When positive rare disease cases comprise $< 0.1\%$ or $0\%$ of available clinical records, deep neural nets suffer from catastrophic collapse or memorize the rare cohort. Deep learning requires large labelled cohorts and lacks the interpretability needed for clinical diagnostics. |
| **Synthetic Oversampling (SMOTE) on Unlabelled Data** | Synthetically interpolating between sparse minority points generates physiologically implausible medical profiles that violate biological pathways. |
| **Agglomerative Hierarchical Clustering (Raw)** | Has $O(n^3)$ time complexity ($O(n^2)$ space) which scales poorly on clinical datasets, and lacks DBSCAN's deterministic core-point density-reachability guarantees. |
| **One-Class SVM with RBF Kernel (Sole Metric)** | Highly sensitive to kernel width $\gamma$ and soft-margin $\nu$. Acts as an opaque black-box without DBSCAN's intuitive spatial clustering and $k$-distance elbow justification. |

---

## 6. Setup, Compilation, and Execution Guide

### Prerequisites
- **Operating System:** Windows 10/11, macOS, or Ubuntu Linux
- **Python:** Python 3.10+ installed ([python.org](https://www.python.org/downloads/))
- **Node.js:** Node.js 18+ (LTS) installed ([nodejs.org](https://nodejs.org))

---

### Step 1: Backend Setup (FastAPI & Scikit-Learn)

Open a new terminal window:
```bash
# Navigate to the backend directory
cd rare-disease-dbscan/backend

# Create a Python virtual environment
python -m venv .venv

# Activate the virtual environment
# macOS / Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat

# Install all required Python packages
pip install -r requirements.txt

# Start the FastAPI server
uvicorn main:app --reload --port 8000
```
- The backend API will be live at: `http://localhost:8000`
- Interactive Swagger API Documentation: `http://localhost:8000/docs`

---

### Step 2: Frontend Setup (React 18 & Vite)

Open a second terminal window:
```bash
# Navigate to the frontend directory
cd rare-disease-dbscan/frontend

# Install all Node.js dependencies
npm install

# Start the Vite development server
npm run dev
```
- The frontend will be live at: `http://localhost:5173`

---

### Step 3: End-to-End Workflow Execution
1. Open `http://localhost:5173` in your browser.
2. Navigate to **Upload Dataset**, drag in your clinical CSV or click **Load DEMO dataset** (500 synthetic patient profiles).
3. On **Inspection**, verify and confirm column roles (ID / label / features).
4. On **Preprocessing**, examine the imputation, one-hot encoding, and $Z$-score scaling transformations.
5. On **DBSCAN setup**, inspect the automated $k$-distance elbow curve and click **Run DBSCAN**.
6. On **Dashboard**, inspect clinical summary cards, PCA 2D/3D projections, and cluster distributions.
7. Navigate through **Correlation graph**, **Confusion matrix**, and **Linear models** to explore deep theoretical reasoning.
8. Assess custom patient profiles via **Assess patient** and export clinical findings using **Download Results CSV**.

---

## 7. REST API Reference

| HTTP Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload` | Upload and inspect a custom clinical dataset CSV |
| `POST` | `/api/upload-sample` | Load the bundled 500-record synthetic medical dataset |
| `POST` | `/api/analyze` | Infer schema, column data types, missing rates, and suggest roles |
| `POST` | `/api/preprocess` | Execute missing imputation, one-hot encoding, and $Z$-score standardization |
| `GET` | `/api/kdistance` | Generate $k$-distance elbow curve data and knee $\varepsilon$ point |
| `POST` | `/api/dbscan` | Fit DBSCAN over the scaled matrix and compute anomaly metrics |
| `GET` | `/api/results` | Retrieve summary statistics, cluster counts, and anomaly tallies |
| `GET` | `/api/correlation` | Compute full Pearson correlation matrix and 2D scatter regressions |
| `GET` | `/api/confusion-matrix` | Generate diagnostic confusion matrix against selected reference targets |
| `GET` | `/api/linear-models/targets` | Discover available numerical and categorical target columns |
| `POST` | `/api/linear-models/train` | Train OLS, Ridge, Lasso, Elastic Net, or Logistic Regression suite with regularisation analysis |
| `GET` | `/api/anomalies` | Fetch paginated, searchable patient anomaly records |
| `GET` | `/api/visualization` | Retrieve PCA projections and feature distribution charts |
| `GET` | `/api/patient-form` | Retrieve metadata schema for single-patient assessment inputs |
| `POST` | `/api/assess-patient` | Compute anomaly factor for a newly presented patient profile |
| `GET` | `/api/download` | Export processed dataset containing cluster IDs and anomaly scores as CSV |
| `POST` | `/api/reset` | Reset in-memory session state |

---

## 8. Viva Voce & Academic Defense Questions

### Q1: Why use DBSCAN for rare disease detection instead of K-Means or Supervised Classifiers?
**Answer:** Rare diseases represent $<0.1\%$ of cases and are largely unlabelled. Supervised classifiers require balanced positive and negative labels; with extreme class imbalance, they predict the majority class for all instances. K-Means fails because:
1. It requires specifying the number of clusters $k$ in advance (unknown in heterogeneous medical data).
2. It assumes spherical, convex clusters of equal variance, whereas real clinical data forms irregular, non-linear manifolds.
3. K-Means forces every outlier into the nearest cluster, shifting cluster centroids and masking rare conditions. In contrast, DBSCAN clusters dense regions of arbitrary shape and isolates low-density points directly into an explicit **Noise / Outlier class (label `-1`)**.

### Q2: How are the DBSCAN hyperparameters $\varepsilon$ (Epsilon) and $MinPts$ (min_samples) chosen objectively?
**Answer:** Rather than arbitrary guesswork, we employ the **$k$-Nearest Neighbour ($k$-NN) distance graph elbow heuristic**:
1. We set $k = MinPts = \max(4, 2 \times d)$ (following the standard rule of thumb for dimensional space $d$).
2. For every data point, we compute its Euclidean distance to its $k$-th nearest neighbour in scaled space and sort the values in ascending order.
3. Points inside dense clusters have low $k$-NN distances and form a flat plateau. Outliers have large distances, causing a sharp upward surge.
4. The point of maximum curvature (the "elbow" or knee point, identified via maximum perpendicular distance from the secant line) indicates the threshold where density drops sharply. This value is selected as $\varepsilon$.

### Q3: Why is Feature Scaling ($Z$-Score Standardisation) mandatory before running DBSCAN?
**Answer:** DBSCAN relies on Euclidean distance in $\mathbb{R}^d$:
$$\text{dist}(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{j=1}^d (x_j - y_j)^2}$$
If features are unscaled, a feature with large numerical range (e.g., Platelet count: $150,000 - 450,000$) completely dominates a feature with small range (e.g., Blood pH: $7.35 - 7.45$). A minute percentage variation in platelets would generate thousands of times more Euclidean distance than a fatal variation in pH. $Z$-score standardization ($z = \frac{x - \mu}{\sigma}$) centers all features at mean 0 and unit variance 1, ensuring all biomarkers contribute equally to spatial density.

### Q4: Geometrically, why does Lasso ($L_1$) regularisation create exact zero coefficients while Ridge ($L_2$) only shrinks them?
**Answer:** Both regularisation methods can be viewed as minimizing the sum of squared errors subject to a constraint on the parameter vector:
- **Lasso constraint:** $\sum_{j=1}^d |w_j| \le t$ (an $L_1$ ball forming a diamond / rhombus with sharp vertices on the coordinate axes).
- **Ridge constraint:** $\sum_{j=1}^d w_j^2 \le t$ (an $L_2$ ball forming a smooth circle / hypersphere).

The optimal parameters occur where the elliptical contours of the unconstrained OLS loss function first touch the constraint boundary. Because the $L_1$ ball has sharp vertices directly on the axes (where one or more $w_j = 0$), the expanding loss ellipse is far more likely to intersect a vertex than an edge. Hence, **Lasso sets non-essential biomarker weights exactly to zero, performing automated feature selection**. The smooth $L_2$ circle has no vertices, so the tangency point almost never lies exactly on an axis, resulting in shrinkage without sparsity.

### Q5: How does the Bias-Variance Trade-off manifest in Ridge/Lasso regularisation and Logistic Regression?
**Answer:**
- When regularisation penalty $\alpha$ is small (or $C$ is large): The model is unconstrained and fits training data closely. **Variance is high and Bias is low**, risking overfitting to sample-specific clinical noise.
- When regularisation penalty $\alpha$ is increased (or $C$ is decreased): The weights are constrained toward zero. The model becomes simpler and more stable across different training samples (**Variance decreases**), but if regularisation is too severe, the model loses the capacity to capture true biomarker relationships (**Bias increases**).
- The goal is to identify the sweet spot where $\text{MSE} = \text{Bias}^2 + \text{Variance} + \sigma^2$ reaches its global minimum on validation data.

---

## 9. Known Limitations

1. **Cohort-Relative Anomaly Detection:** Anomaly scores and cluster memberships are evaluated relative to the uploaded cohort. An individual identified as anomalous in a healthy cohort might be deemed normative in an acute intensive-care cohort.
2. **Curse of Dimensionality:** When datasets contain dozens of one-hot encoded categorical columns, Euclidean distances between points begin to converge to similar values, degrading DBSCAN density contrast. Pre-filtering low-variance columns is advised.
3. **Single-User In-Memory Architecture:** The backend maintains an in-memory session state designed for demonstrator and research evaluation, rather than multi-tenant concurrent sessions.
4. **Clinical Verification Prerequisite:** Algorithmic noise points flag statistical outliers, which may include rare diseases, novel syndromes, atypical drug reactions, or measurement errors; they must always be verified by clinical specialists.
