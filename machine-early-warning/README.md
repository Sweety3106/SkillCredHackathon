# Machine Failure Early Warning System

A machine learning and deep learning system for predictive maintenance on industrial turbofan engines (NASA C-MAPSS dataset). The system processes multivariate sensor time series within rolling windows to forecast approaching end-of-life failures before they occur.

---

## Architecture & Pipeline Overview

```
[Raw Sensor Data] 
       │
       ▼
 [Windowing (P1)] ──► [Feature Extraction (P1)]
       │                          │
       ▼                          ▼
[LSTM Model (P3)]      [Baseline Model (P2)]
       │                          │
       └──────────┬───────────────┘
                  │
                  ▼
      [Predictions Contract]
                  │
                  ▼
       [Evaluation & Metrics (P4)]
                  │
                  ├─► metrics.json
                  ├─► per_machine_metrics.csv
                  ├─► threshold_sweep.csv
                  ├─► model_comparison.csv
                  └─► outputs/figures/*.png
```

---

## Evaluation Methodology (P4 Contribution)

Predictive maintenance systems cannot be evaluated using generic classification accuracy alone. Because normal operating cycles vastly outnumber pre-failure states, naive accuracy produces misleadingly optimistic results while failing to answer core operational questions: *Did maintenance receive sufficient advance warning before catastrophic failure?* and *Are operations disrupted by excessive false alarms?*

Our evaluation framework is designed around these operational realities:

### 1. Test-Split Isolation & Ground Truth
- **Split Invariant**: Model evaluation is strictly performed on the held-out test split (`split == "test"`). Training and validation rows are excluded from final reported performance.
- **C-MAPSS Ground Truth**: In NASA C-MAPSS, engines run until failure (or until a recorded cutoff with known remaining useful life). Actual failure cycle is derived from:
  $$\text{Actual Failure Cycle} = \text{window\_end} + \text{RUL}_{\text{end}}$$
  This formulation guarantees exact ground truth without synthetic assumptions or data leakage.

### 2. Imbalance-Aware Classification Metrics
- **Precision, Recall, & F1**: Precision measures the proportion of alerts that correspond to genuine near-failure states, while Recall measures the proportion of critical windows successfully captured. The F1 score reflects the harmonic mean between alert validity and coverage.
- **PR-AUC (Precision-Recall Area Under Curve)**: In heavy class imbalance, ROC curves and ROC-AUC present overly optimistic views because true negatives dwarf false positives. PR-AUC focuses exclusively on the minority positive class (imminent failure), providing an unskewed measure of ranking performance.

### 3. Machine-Level False Alarm Rate (FAR)
- **Window-Level vs. Machine-Level**: Standard metrics count false positive windows. However, in factory operations, maintenance teams care whether a healthy asset triggers unnecessary shutdowns.
- **Definition**:
  $$\text{FAR} = \frac{\text{Total Non-Failing Machines with } \ge 1 \text{ False Alert}}{\text{Total Non-Failing Machines}}$$
  A non-failing machine that never crosses the risk threshold has a machine false alarm score of $0$; any breach marks it as a false alarm.

### 4. Advance Warning Lead Time
Lead time measures the operational runway provided to engineers before an engine reaches zero remaining useful life:
- For each failing test machine, the system identifies the **first cycle** where predicted risk crosses the operational threshold before failure:
  $$\text{Lead Time} = \text{Actual Failure Cycle} - \text{First Alert Cycle}$$
- **Machines Caught vs. Missed**:
  - **Caught**: $\text{First Alert Cycle} < \text{Actual Failure Cycle}$ ($\text{Lead Time} > 0$).
  - **Missed**: The model fails to breach the threshold before failure or never raises an alert.
- Metrics report: Mean, Median, Min, and Max Lead Time (in operating cycles) across all caught units.

### 5. Operational Threshold Sweep
Rather than arbitrarily fixing the risk decision threshold at $0.50$, the pipeline sweeps thresholds from $0.10$ to $0.90$ in steps of $0.05$:
- Evaluates the operational trade-off: lower thresholds maximize Recall and Lead Time but increase False Alarm Rate; higher thresholds reduce false alarms at the cost of missed failures.
- Sweep results are persisted to `outputs/metrics/threshold_sweep.csv` and plotted in `outputs/figures/threshold_sweep.png`.

### 6. Data & RUL Leakage Audit
To prevent spurious results, an automated audit executes prior to metric computation:
1. **RUL Feature Check**: Verifies that no input feature contains `rul`, `remaining_useful_life`, or `failure_event`.
2. **Machine Split Disjointness**: Verifies that train machine IDs and test machine IDs are mutually exclusive ($\text{Train} \cap \text{Test} = \emptyset$).
3. **Test-Only Enforcement**: Confirms evaluation data belongs exclusively to `split == "test"`.
4. **Temporal Windowing**: Confirms historical ordering ($\text{window\_start} < \text{window\_end}$) without future sensor leakage.
5. **Threshold Integrity**: Checks that thresholds were locked a priori or tuned on validation splits, not overfitted directly to final test labels.

---

## Evaluation Artifacts & Outputs

All evaluation artifacts are automatically written to `outputs/`:

```
outputs/
├── metrics.json                         # Primary machine-readable evaluation report
├── metrics/
│   ├── metrics.json                     # Mirror of primary metrics report
│   ├── per_machine_metrics.csv          # Machine-by-machine lead times, alert cycles, and status
│   ├── threshold_sweep.csv              # 17-point threshold sensitivity analysis table
│   └── model_comparison.csv             # Head-to-head comparison table (Baseline vs. LSTM)
└── figures/
    ├── confusion_matrix_baseline.png    # Baseline confusion matrix with cell percentages
    ├── confusion_matrix_lstm.png        # LSTM confusion matrix (when LSTM predictions available)
    ├── pr_curve.png                     # Precision-Recall curves with PR-AUC and no-skill baseline
    ├── threshold_sweep.png              # Multi-panel sensitivity chart (PR/FAR & Lead Time)
    ├── risk_trajectory.png              # Single-asset timeline showing Alert, Failure, and Lead Time
    ├── feature_importance.png           # Leakage-audited top 15 model feature importances
    └── sensor_traces.png                # Top model-associated sensor signals (non-causal traces)
```

---

## Running Evaluation

### Prerequisites
Install dependencies (Python 3.10+):
```bash
pip install numpy pandas scikit-learn matplotlib pyarrow pytest
```

### Execute Evaluation CLI
Run evaluation on predictions:
```bash
# Auto-discovers outputs/predictions/predictions.parquet or fixtures
python -m src.evaluate

# Custom predictions file and operating threshold
python -m src.evaluate --predictions outputs/predictions.parquet --threshold 0.50 --dataset FD001
```

### Run Evaluation Unit Tests
```bash
python -m pytest tests/test_evaluate.py -v
```
