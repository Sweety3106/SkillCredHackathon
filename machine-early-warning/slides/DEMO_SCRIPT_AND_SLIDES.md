# 4-Minute Demo Script & Presentation Slides (P4 Role)

> **Role Note (from ICD)**: P4 speaks during the demo. P6 drives the dashboard screen. Nobody else talks unless a judge asks them directly.

---

## 4-Minute Timed Pitch Script

### 0:00 – 0:30 | Part 1: The Problem
> *"Judges, unplanned industrial equipment breakdown is extraordinarily expensive. In aviation and heavy manufacturing, an engine failure during operation costs hundreds of thousands of dollars in downtime and emergency repairs. But machines do not die randomly—they exhibit subtle micro-vibrations, temperature anomalies, and pressure drifts dozens of cycles before failure.
>
> The problem is not simply predicting failure; the job is predicting it **early enough** for a maintenance crew to take action, while keeping false alarms low so the crew doesn't ignore the system."*

---

### 0:30 – 1:00 | Part 2: Our Approach & Pipeline Architecture
*(P6 shows Architecture / Pipeline overview)*

> *"We engineered a dual-model predictive early-warning system on the NASA C-MAPSS dataset.
> 1. We slice multivariate sensor readings into 30-cycle rolling windows and ask: **'Will this engine fail within the next 30 cycles?'**
> 2. We train an LSTM sequence neural network directly on normalized sensor dynamics to capture non-linear temporal degradation, compared against an engineered-feature gradient boosted baseline.
> 3. Critically, to prevent data leakage, we split train and test **strictly by machine ID**—no engine appears in both splits."*

---

### 1:00 – 2:15 | Part 3: Live Demo (The Core Differentiator)
*(P6 switches to Screen 1: Fleet Overview, then clicks into Machine 48 on Screen 2: Risk Over Time)*

> *"Here is our live dashboard. On the fleet overview, machines are ranked by urgency. 
> 
> Let's look at **Machine 48**:
> - Look at the risk trajectory: for the first 180 cycles, the engine runs normally with risk below 0.05.
> - At cycle 190, the LSTM detects an abnormal drift and breaches our operational threshold of 0.50. The system triggers an amber **ALERT at cycle 190**.
> - The machine reaches end-of-life and **fails at cycle 231**.
> - That shaded window between ALERT and FAILURE represents **41 cycles of advance warning lead time**—more than enough time for a scheduled overhaul without taking the plant offline.
> 
> Below the trajectory, look at the top model-associated sensor signals: **Sensor 11 (Static Pressure at HPC)** and **Sensor 9 (Core Speed)** start drifting upward at cycle 185, driving the alarm."*

---

### 2:15 – 3:00 | Part 4: Rigorous Test Results & Business Threshold
*(P6 switches to Screen 4: Model Results / Threshold Sweep)*

> *"On our held-out test split of 3,661 unseen windows across 20 engines:
> - **Recall**: 95.5% on critical failure-soon windows.
> - **PR-AUC**: 0.977 — which measures ranking performance under real-world class imbalance.
> - **Lead Time**: Our system achieved a **mean lead time of 35.95 cycles** (median 32.5 cycles, minimum 23 cycles).
> - **Machines Caught**: **20 out of 20 test engines (100%)** were caught before failure with zero missed breakdowns.
> 
> **Why Threshold 0.50?**
> We performed a 17-point threshold sweep from 0.10 to 0.90. We didn't pick 0.50 blindly. In our economic cost model, a false alarm costs ₹5,000 for an inspection, while a missed failure costs ₹200,000 in catastrophic downtime. At 0.50, we catch 100% of failures with a 0% machine false alarm rate on non-failing assets."*

---

### 3:00 – 3:30 | Part 5: Bonus Features & Honest Model Comparison
> *"We implemented two operational innovations:
> 1. **Alert Deduplication**: Standard systems fire alerts on every subsequent window, spamming technicians with 36 alerts per failing machine. Our deduplication algorithm reduces this to **4.3 actionable alerts per machine—an 88.1% reduction** in alert fatigue.
> 2. **Honest Engineering Trade-Off**: The LSTM achieves an exceptional 0.977 PR-AUC and 35.95 cycles of warning time. While it takes longer to train than the baseline, its sequence-aware memory makes it uniquely suited for temporal degradation modeling."*

---

### 3:30 – 4:00 | Part 6: Summary & Q&A
> *"In summary: 100% of failing test machines caught, average 36 cycles of warning runway, zero machine false alarms, and 88% alert deduplication. We are ready for your questions."*

---

## Cheat Sheet: Anticipated Judge Questions & Answers

| Question | Who Answers | Exact Answer |
|---|---|---|
| *"How did you split train and test?"* | **P1 / P4** | *"By machine ID, never by row. 80 engines in train/val, 20 engines in held-out test. Splitting by row would leak 29 of 30 readings from adjacent windows, producing fake 99% accuracy."* |
| *"What is your class balance?"* | **P4** | *"In our test windows, 16.9% are positive (failure within 30 cycles) and 83.1% are normal. That's why we use PR-AUC rather than accuracy or ROC-AUC."* |
| *"Is your model just detecting that a machine is old?"* | **P4** | *"No. In our feature audit, time and RUL are strictly excluded. The model detects physical degradation patterns in sensors 11, 9, 12, and 14 regardless of whether the engine's total lifespan is 150 cycles or 360 cycles."* |
| *"What does a false alarm cost you?"* | **P4** | *"In our cost model: a false alarm costs ₹5,000 for a technician inspection. A missed failure costs ₹200,000 in catastrophic plant downtime. We optimized the threshold to minimize this exact rupee risk curve."* |
| *"Why 30 cycles for the window and horizon?"* | **P1 / P4** | *"30 cycles provides sufficient temporal context to capture trend slope and variance, while a 30-cycle horizon gives maintenance crews roughly 1 to 2 shifts of operational runway to schedule service."* |

---

## Slide Outline for Presentation Deck

- **Slide 1**: Title — Machine Failure Early-Warning System (Team of 6)
- **Slide 2**: The Challenge — Unplanned Downtime vs. False Alarms
- **Slide 3**: The Pipeline — Windowing, LSTM Sequence Modeling, and Zero-Leakage Split
- **Slide 4**: Live Case Study — Machine 48 (Alert at Cycle 190, Died at Cycle 231, 41-Cycle Lead Time)
- **Slide 5**: Evaluation Scorecard — 35.95 Cycles Mean Lead Time, 100% Machines Caught, 0.977 PR-AUC
- **Slide 6**: Operational ROI — 88.1% Alert Deduplication & Economic Cost Optimization
