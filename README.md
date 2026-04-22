# Telecom Customer Churn Prediction & Retention Analytics System 📉➡️📈

> Turning churn prediction into actionable retention intelligence.

A full-stack Machine Learning platform that identifies telecom customers at risk of churning — before they leave. The system goes beyond a simple binary prediction by explaining *why* a customer might churn, *when* it is likely to happen, and *which* customers deserve the most attention based on their business value.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Our Solution](#our-solution)
- [Unique Features](#unique-features)
- [End-to-End ML Pipeline](#end-to-end-ml-pipeline)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Model Performance](#model-performance)
- [Output Screens](#output-screens)
- [Key Benefits](#key-benefits)
- [Applications & Future Scope](#applications--future-scope)
- [Team](#team)

---

## Problem Statement

Customer churn is one of the most expensive problems in the telecom industry — and most businesses only realize a customer has churned after it's too late to act.

| Statistic | Impact |
|---|---|
| 💸 $1.6 Trillion lost globally per year due to churn | Source: Accenture Research |
| 🔁 Acquiring a new customer costs 5–7× more than retaining one | Direct revenue leakage |
| 🙈 Reactive approach | Churn discovered only after the customer has already left |
| 🎯 No prioritization | Retention budgets wasted on low-risk customers |

---

## Our Solution

An intelligent churn prediction platform that tackles all four dimensions of the churn problem:

| Step | What It Does |
|---|---|
| **01 — Predict** | Identifies customers likely to churn using a trained Random Forest model (93% accuracy) |
| **02 — Explain** | Highlights the top 5 key factors driving each prediction using SHAP-style feature importance |
| **03 — Anticipate** | Estimates *when* churn is likely using survival analysis and tenure-based patterns |
| **04 — Prioritise** | Ranks customers by churn risk × Customer Lifetime Value (CLV) for targeted retention |

---

## Unique Features

### 🔍 Explainable AI (XAI)
- SHAP-style feature importance for every single prediction
- Visual bar chart showing the **top 5 churn risk drivers** per customer
- Shows which features *hurt* vs *help* retention likelihood
- Enables data-driven conversations with customers and stakeholders

### ⏱️ Churn Timeline (Survival Analysis)
- Survival curve plotted across a **1–36 month window**
- Plain-language churn window output — e.g., *"30–60 days"*
- Based on churn probability combined with tenure patterns
- Enables time-sensitive, proactive outreach campaigns

### 💰 Customer Lifetime Value (CLV) Scoring
- Expected CLV calculated in **Indian Rupees (₹)**
- Formula: Monthly Revenue × Retention Likelihood
- Customers automatically tiered as **Critical / High / Medium / Low** priority
- Focuses retention budget where the financial impact matters most

### 🖥️ Live Web Dashboard
- Flask-powered responsive web application
- Real-time predictions with instant results
- Interactive **Chart.js** analytics on the Insights page
- Summary dashboards for understanding churn patterns across the customer base

---

## ![End to End ML Pipeline](https://drive.google.com/file/d/13CRaTvWWVqxfWcMg0GqjStprkUEA0vOq/view?usp=drive_link)

```
Raw Customer Data
       ↓
┌─────────────────────────────┐
│     Data Preprocessing       │
│  Null handling, encoding,    │
│  feature engineering         │
└────────────┬────────────────┘
             ↓
┌─────────────────────────────┐
│    Class Balancing           │
│    SMOTEENN                  │
│  (Oversampling + Cleaning)   │
└────────────┬────────────────┘
             ↓
┌─────────────────────────────┐
│    Model Training            │
│    Random Forest Classifier  │
│    Ensemble of decision trees│
└────────────┬────────────────┘
             ↓
┌──────────────────────────────────────────────────┐
│               Prediction Outputs                  │
│                                                   │
│  ┌──────────────┐  ┌────────────┐  ┌──────────┐  │
│  │  Churn Risk  │  │  Timeline  │  │   CLV    │  │
│  │  Score (%)   │  │  Survival  │  │ Scoring  │  │
│  └──────────────┘  └────────────┘  └──────────┘  │
└──────────────────────────┬───────────────────────┘
                           ↓
┌─────────────────────────────┐
│    Explainability Layer      │
│    SHAP Feature Importance   │
│    Top 5 drivers per customer│
└────────────┬────────────────┘
             ↓
┌─────────────────────────────┐
│    Flask Web Dashboard       │
│    Real-time results &       │
│    interactive analytics     │
└─────────────────────────────┘
```

### Why Random Forest?

- Ensemble of decision trees → reduces overfitting, improves accuracy
- Works exceptionally well with structured/tabular customer data
- Handles non-linear relationships between features
- Robust to noise and irrelevant feature variations

### Why SMOTEENN?

Real-world churn datasets are heavily imbalanced (far fewer churners than loyal customers). SMOTEENN combines:
- **SMOTE** — Synthetic Minority Over-sampling to generate new churn samples
- **ENN (Edited Nearest Neighbours)** — Cleans overlapping/noisy samples post-oversampling

This ensures the model learns churn patterns effectively rather than defaulting to predicting the majority class.

---

## ![System Architecture](https://drive.google.com/file/d/1_5En0JVdoJPMQ6Hk2EINUhqKpOjsK0fm/view?usp=drive_link)

```
┌──────────────────────────────────────┐
│           User Interface              │
│     Flask Web App + Chart.js          │
└─────────────────┬────────────────────┘
                  ↓
┌──────────────────────────────────────┐
│         Prediction Engine             │
│   Random Forest Model (.pkl)          │
│   Feature Preprocessing Pipeline     │
└─────────────────┬────────────────────┘
                  ↓
┌──────────────────────────────────────┐
│        Analytics Modules              │
│  SHAP Explainer │ Survival Analysis  │
│  CLV Calculator │ Priority Scoring   │
└──────────────────────────────────────┘
```



---

## Tech Stack

| Layer | Technology |
|---|---|
| **ML Model** | Random Forest (Scikit-learn) |
| **Class Balancing** | SMOTEENN (imbalanced-learn) |
| **Explainability** | SHAP (SHapley Additive exPlanations) |
| **Survival Analysis** | Tenure-based probability modeling |
| **Web Framework** | Flask (Python) |
| **Frontend** | HTML, CSS, Chart.js |
| **Language** | Python |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Chart.js |

---

## Model Performance

| Metric | Result |
|---|---|
| Accuracy | **93%** on test records |
| Class Balancing | SMOTEENN applied |
| Explainability | SHAP feature importance per prediction |
| Validation | Train/Test split with stratified evaluation |

---

## Output Screens

The platform provides four core output views:

**1. System Dashboard** — High-level summary of all customer churn metrics and analytics.

**2. Churn Risk Prediction** — Per-customer churn probability score with SHAP-driven explanation of the top 5 contributing factors (Explainable AI panel).

**3. Retention Timeline & Survival Analysis** — Survival curve showing the estimated churn window and recommended outreach timing.

**4. CLV Analysis & Action Recommendations** — Customer Lifetime Value in ₹, priority tier, and suggested retention actions.

---

## Key Benefits

| Benefit | Description |
|---|---|
| ⚡ Proactive Decision-Making | Identify at-risk customers *before* churn happens — enabling early intervention |
| 🔎 Data-Driven Insights | Clear explanations behind every churn prediction for smarter retention strategies |
| 💵 Revenue Protection | Prioritises high-value customers using CLV to minimise revenue loss |
| 💡 Cost Efficiency | Retaining existing customers is 5–7× cheaper than acquiring new ones |

---

## Applications & Future Scope

### Current Applications
- **Telecom** — Reduce churn and improve customer retention strategies
- **Banking** — Predict account closures before they happen
- **E-commerce** — Identify and re-engage inactive customers
- **Subscription Services** — Improve renewal rates

### Future Scope
- 🔗 CRM Integration for automated retention action triggers
- 📱 Mobile-friendly responsive dashboard
- ⚡ Real-time data pipeline for live churn scoring
- 📬 Automated real-time alerts via Email / SMS
- 🌍 Expansion to multi-industry churn datasets

---

## Conclusion

The Telecom Customer Churn Prediction & Retention Analytics System converts raw customer data into actionable retention intelligence. By combining a high-accuracy Random Forest model with SHAP explainability, survival analysis, and CLV scoring, the platform enables retention teams to act *before* churn happens — and focus their efforts where they matter most.

> **Not just a prediction tool — a complete and transparent customer retention system.**
