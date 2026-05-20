# 👥 Employee Attrition Risk Predictor

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![ML](https://img.shields.io/badge/ML-Gradient_Boosting-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange)
![Streamlit](https://img.shields.io/badge/Deployed-Streamlit_Cloud-FF4B4B?logo=streamlit)

> **Predict which employees are at risk of leaving using ML. Enables proactive HR intervention before attrition happens — saving $50,000+ per lost employee.**

---

## 📌 What This Project Does

Employee attrition costs companies 50–200% of an employee's annual salary in recruiting, onboarding, and productivity loss. This ML system predicts attrition risk before it happens — giving HR teams time to intervene.

**Key capabilities:**
- 🔍 **Individual risk prediction** — enter any employee's profile and get instant risk score
- 📊 **Batch analysis** — analyze 500+ employees at once and identify high-risk groups
- ⚠️ **Risk factor identification** — pinpoints exactly what's driving each employee's risk
- 💡 **HR recommendations** — actionable retention strategies based on risk factors
- 📈 **Feature importance** — understand which factors matter most company-wide

---

## 🏗️ ML Pipeline

```
Synthetic HR Dataset (1,000 employees)
         │
         ▼
Feature Engineering
  • Demographics (age, education)
  • Compensation (monthly income, job level)
  • Satisfaction scores (job, work-life balance)
  • Tenure metrics (years at company, in role)
  • Growth signals (promotions, training)
  • Behavioral (overtime, commute distance)
         │
         ▼
Gradient Boosting Classifier
  • 100 estimators
  • 80/20 train/test split
  • ~85%+ test accuracy
         │
         ▼
Risk Score (0–100%) + Risk Level
         │
         ▼
Risk Factors + HR Recommendations
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| ML Model | scikit-learn Gradient Boosting Classifier |
| Data Processing | Pandas + NumPy |
| Encoding | LabelEncoder (categorical features) |
| Evaluation | accuracy_score, classification_report |
| Frontend | Streamlit |
| Language | Python 3.10+ |

---

## ✨ Key Features

- 🎯 **Risk scoring** — 0–100% attrition probability per employee
- 🚦 **4-level risk classification** — Critical / High / Moderate / Low
- 💰 **Replacement cost estimator** — calculates financial impact of losing each employee
- ⚠️ **8 risk factor checks** — satisfaction, overtime, pay, promotion, commute, tenure
- 💡 **6 HR recommendation types** — specific, actionable retention strategies
- 📊 **Batch dashboard** — risk distribution and department-level analysis
- 📈 **Feature importance chart** — which factors drive attrition most

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/madhavi-akella-lab/attrition-predictor
cd attrition-predictor
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
attrition-predictor/
├── app.py          # Streamlit UI — prediction, batch, insights tabs
├── model.py        # ML model, data generation, risk analysis
├── requirements.txt
└── README.md
```

---

## 📊 Risk Factors Modeled

| Factor | Impact Level |
|---|---|
| Job Satisfaction (Low) | 🔴 High |
| Work-Life Balance (Poor) | 🔴 High |
| Overtime | 🔴 High |
| Low Compensation | 🔴 High |
| No Recent Promotion (5+ years) | 🟠 Medium-High |
| New Employee (<2 years) | 🟠 Medium |
| Long Commute | 🟡 Medium |
| High Job-Hopping History | 🟡 Medium |

---

## 💼 Business Impact

```
Average replacement cost = 6 months salary
500-employee company with 15% attrition = 75 employees/year
Average salary = $80,000 → replacement cost = $40,000/employee
Total annual attrition cost = $3,000,000

With ML early intervention (30% reduction in attrition):
Annual savings = $900,000
```

---

## 🔮 Production Enhancement Path

| Current | Production |
|---|---|
| Synthetic data | Real HRIS system data (Workday, SAP) |
| Gradient Boosting | XGBoost / LightGBM with hypertuning |
| Streamlit UI | Internal HR dashboard |
| Manual input | Automated daily batch scoring |
| Local model | AWS SageMaker endpoint |

---

## 👩‍💻 About

**Madhavi Akella** — Data & AI Engineer | Databricks Generative AI Engineer Associate

🔗 [LinkedIn](https://linkedin.com/in/madhavi-akella-2b8213114) · 🌐 [Portfolio](https://madhavi-akella.netlify.app) · ⬡ [GitHub](https://github.com/madhavi-akella-lab)
