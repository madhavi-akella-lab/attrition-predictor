import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


def generate_hr_data(n=1000):
    """Generate realistic synthetic HR dataset."""
    np.random.seed(42)

    age = np.random.randint(22, 60, n)
    years_at_company = np.random.randint(0, 20, n)
    years_in_role = np.clip(np.random.randint(0, years_at_company + 1, n), 0, years_at_company)
    monthly_income = np.random.randint(2000, 20000, n)
    distance_from_home = np.random.randint(1, 30, n)
    job_satisfaction = np.random.randint(1, 5, n)       # 1=Low, 4=High
    work_life_balance = np.random.randint(1, 5, n)
    overtime = np.random.choice([0, 1], n, p=[0.7, 0.3])
    num_companies_worked = np.random.randint(0, 10, n)
    training_times = np.random.randint(0, 6, n)
    job_level = np.random.randint(1, 6, n)
    performance_rating = np.random.randint(1, 5, n)
    last_promotion_years = np.random.randint(0, 15, n)

    departments = np.random.choice(["Sales", "R&D", "HR", "Engineering", "Finance"], n)
    job_roles = np.random.choice(["Manager", "Analyst", "Engineer", "Director", "Executive"], n)
    education = np.random.randint(1, 6, n)

    # Attrition logic — realistic risk factors
    attrition_score = (
        (job_satisfaction < 2).astype(int) * 2
        + (work_life_balance < 2).astype(int) * 1.5
        + (overtime == 1).astype(int) * 1.5
        + (monthly_income < 4000).astype(int) * 1.5
        + (years_at_company < 2).astype(int) * 1
        + (distance_from_home > 20).astype(int) * 0.5
        + (last_promotion_years > 5).astype(int) * 1
        + (num_companies_worked > 5).astype(int) * 0.5
        + np.random.normal(0, 0.5, n)
    )
    attrition = (attrition_score > 3.5).astype(int)

    df = pd.DataFrame({
        "Age": age,
        "MonthlyIncome": monthly_income,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_role,
        "DistanceFromHome": distance_from_home,
        "JobSatisfaction": job_satisfaction,
        "WorkLifeBalance": work_life_balance,
        "OverTime": overtime,
        "NumCompaniesWorked": num_companies_worked,
        "TrainingTimesLastYear": training_times,
        "JobLevel": job_level,
        "PerformanceRating": performance_rating,
        "YearsSinceLastPromotion": last_promotion_years,
        "Education": education,
        "Department": departments,
        "JobRole": job_roles,
        "Attrition": attrition
    })
    return df


FEATURE_COLS = [
    "Age", "MonthlyIncome", "YearsAtCompany", "YearsInCurrentRole",
    "DistanceFromHome", "JobSatisfaction", "WorkLifeBalance", "OverTime",
    "NumCompaniesWorked", "TrainingTimesLastYear", "JobLevel",
    "PerformanceRating", "YearsSinceLastPromotion", "Education",
    "Department_encoded", "JobRole_encoded"
]


def train_model(st):
    @st.cache_resource
    def _train():
        df = generate_hr_data(1000)

        # Encode categoricals
        le_dept = LabelEncoder()
        le_role = LabelEncoder()
        df["Department_encoded"] = le_dept.fit_transform(df["Department"])
        df["JobRole_encoded"] = le_role.fit_transform(df["JobRole"])

        X = df[FEATURE_COLS]
        y = df["Attrition"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        accuracy = round(accuracy_score(y_test, preds) * 100, 1)

        # Feature importance
        importance = dict(zip(FEATURE_COLS, model.feature_importances_))
        importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        return model, accuracy, le_dept, le_role, importance_sorted

    return _train()


def predict_attrition(model, le_dept, le_role, employee_data):
    """Predict attrition risk for a single employee."""
    dept_encoded = le_dept.transform([employee_data["Department"]])[0]
    role_encoded = le_role.transform([employee_data["JobRole"]])[0]

    features = np.array([[
        employee_data["Age"],
        employee_data["MonthlyIncome"],
        employee_data["YearsAtCompany"],
        employee_data["YearsInCurrentRole"],
        employee_data["DistanceFromHome"],
        employee_data["JobSatisfaction"],
        employee_data["WorkLifeBalance"],
        employee_data["OverTime"],
        employee_data["NumCompaniesWorked"],
        employee_data["TrainingTimesLastYear"],
        employee_data["JobLevel"],
        employee_data["PerformanceRating"],
        employee_data["YearsSinceLastPromotion"],
        employee_data["Education"],
        dept_encoded,
        role_encoded,
    ]])

    probability = model.predict_proba(features)[0][1]
    prediction = "High Risk" if probability > 0.5 else "Low Risk"

    risk_level = (
        "🔴 Critical" if probability > 0.75
        else "🟠 High" if probability > 0.5
        else "🟡 Moderate" if probability > 0.3
        else "🟢 Low"
    )

    return prediction, probability, risk_level


def get_risk_factors(employee_data):
    """Identify key risk factors for an employee."""
    factors = []
    if employee_data["JobSatisfaction"] <= 2:
        factors.append("⚠️ Low job satisfaction")
    if employee_data["WorkLifeBalance"] <= 2:
        factors.append("⚠️ Poor work-life balance")
    if employee_data["OverTime"] == 1:
        factors.append("⚠️ Working overtime")
    if employee_data["MonthlyIncome"] < 4000:
        factors.append("⚠️ Below-average compensation")
    if employee_data["YearsSinceLastPromotion"] > 5:
        factors.append("⚠️ No promotion in 5+ years")
    if employee_data["YearsAtCompany"] < 2:
        factors.append("⚠️ Relatively new employee (high early attrition risk)")
    if employee_data["DistanceFromHome"] > 20:
        factors.append("⚠️ Long commute distance")
    if employee_data["NumCompaniesWorked"] > 5:
        factors.append("⚠️ High job-hopping history")
    return factors if factors else ["✅ No significant risk factors detected"]
