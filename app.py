import streamlit as st
import pandas as pd
import numpy as np
from model import train_model, predict_attrition, get_risk_factors, generate_hr_data, FEATURE_COLS

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👥",
    layout="wide",
)

st.markdown("""
<style>
    .risk-critical { background:#fce8e6; border:2px solid #ea4335; border-radius:12px; padding:20px; text-align:center; }
    .risk-high     { background:#fff3e0; border:2px solid #ff9800; border-radius:12px; padding:20px; text-align:center; }
    .risk-moderate { background:#fff8e1; border:2px solid #fbbc04; border-radius:12px; padding:20px; text-align:center; }
    .risk-low      { background:#e6f4ea; border:2px solid #34a853; border-radius:12px; padding:20px; text-align:center; }
    .risk-num { font-size:48px; font-weight:800; line-height:1; }
    .insight-box { background:#f0f7ff; border-left:4px solid #2d7dd2; border-radius:8px; padding:12px 16px; margin:6px 0; font-size:14px; }
    .factor-box  { background:#fff8e1; border-left:4px solid #fbbc04; border-radius:8px; padding:12px 16px; margin:6px 0; font-size:14px; }
</style>
""", unsafe_allow_html=True)

st.title("👥 Employee Attrition Risk Predictor")
st.markdown("**Predict which employees are at risk of leaving using ML. Enables proactive HR intervention.**")
st.markdown("*Built with scikit-learn Gradient Boosting · HR Analytics · No API key needed*")
st.divider()

with st.spinner("Training ML model on HR data..."):
    model, accuracy, le_dept, le_role, importance = train_model(st)
st.success(f"✅ Model ready! Test accuracy: **{accuracy}%**")
st.divider()

tab1, tab2, tab3 = st.tabs(["🔍 Predict Individual Risk", "📊 Batch Analysis", "📈 Model Insights"])

with tab1:
    st.subheader("Enter Employee Details")
    st.markdown("*Adjust the inputs to match an employee's profile*")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Personal Info**")
        age = st.slider("Age", 20, 60, 35)
        distance = st.slider("Distance from Home (km)", 1, 30, 10)
        num_companies = st.slider("Number of Previous Companies", 0, 10, 2)
        education = st.selectbox("Education Level", [1, 2, 3, 4, 5],
                                  format_func=lambda x: {1:"Below College", 2:"College", 3:"Bachelor", 4:"Master", 5:"Doctor"}[x])

    with col2:
        st.markdown("**💼 Job Details**")
        department = st.selectbox("Department", ["Engineering", "Sales", "R&D", "HR", "Finance"])
        job_role = st.selectbox("Job Role", ["Analyst", "Engineer", "Manager", "Director", "Executive"])
        job_level = st.slider("Job Level", 1, 5, 2)
        monthly_income = st.slider("Monthly Income ($)", 2000, 20000, 6000, step=500)
        overtime = st.checkbox("Works Overtime")

    with col3:
        st.markdown("**😊 Satisfaction & Growth**")
        job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3,
                                      help="1=Low, 2=Medium, 3=High, 4=Very High")
        work_life_balance = st.slider("Work-Life Balance", 1, 4, 3,
                                       help="1=Bad, 2=Good, 3=Better, 4=Best")
        years_at_company = st.slider("Years at Company", 0, 20, 5)
        years_in_role = st.slider("Years in Current Role", 0, 15, 3)
        last_promotion = st.slider("Years Since Last Promotion", 0, 15, 2)
        training_times = st.slider("Training Sessions Last Year", 0, 6, 3)
        performance = st.slider("Performance Rating", 1, 4, 3)

    st.markdown("")
    if st.button("🔍 Predict Attrition Risk", type="primary", use_container_width=True):
        employee = {
            "Age": age,
            "MonthlyIncome": monthly_income,
            "YearsAtCompany": years_at_company,
            "YearsInCurrentRole": years_in_role,
            "DistanceFromHome": distance,
            "JobSatisfaction": job_satisfaction,
            "WorkLifeBalance": work_life_balance,
            "OverTime": int(overtime),
            "NumCompaniesWorked": num_companies,
            "TrainingTimesLastYear": training_times,
            "JobLevel": job_level,
            "PerformanceRating": performance,
            "YearsSinceLastPromotion": last_promotion,
            "Education": education,
            "Department": department,
            "JobRole": job_role,
        }

        prediction, probability, risk_level = predict_attrition(model, le_dept, le_role, employee)
        risk_factors = get_risk_factors(employee)

        st.divider()
        r1, r2, r3 = st.columns(3)

        css = "risk-critical" if probability > 0.75 else "risk-high" if probability > 0.5 else "risk-moderate" if probability > 0.3 else "risk-low"
        color = "#c0392b" if probability > 0.75 else "#e65100" if probability > 0.5 else "#7b4f00" if probability > 0.3 else "#1a7340"

        with r1:
            st.markdown(f"""
            <div class="{css}">
                <div class="risk-num" style="color:{color}">{probability:.0%}</div>
                <div style="font-size:14px;margin-top:6px">Attrition Probability</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class="{css}">
                <div style="font-size:28px;font-weight:800;color:{color}">{risk_level}</div>
                <div style="font-size:14px;margin-top:6px">Risk Level</div>
            </div>
            """, unsafe_allow_html=True)

        with r3:
            retention_cost = monthly_income * 6
            st.markdown(f"""
            <div class="risk-{'critical' if probability > 0.5 else 'low'}">
                <div style="font-size:28px;font-weight:800;color:{color}">${retention_cost:,}</div>
                <div style="font-size:14px;margin-top:6px">Estimated Replacement Cost</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### ⚠️ Risk Factors Identified")
        for factor in risk_factors:
            css_class = "factor-box" if "⚠️" in factor else "insight-box"
            st.markdown(f'<div class="{css_class}">{factor}</div>', unsafe_allow_html=True)

        st.markdown("#### 💡 HR Recommendations")
        recommendations = []
        if job_satisfaction <= 2:
            recommendations.append("📋 Schedule a 1:1 career development discussion immediately")
        if work_life_balance <= 2:
            recommendations.append("⏰ Review workload and consider flexible working arrangements")
        if overtime:
            recommendations.append("🏠 Evaluate overtime frequency and consider additional resources")
        if monthly_income < 4000:
            recommendations.append("💰 Review compensation against market benchmarks")
        if last_promotion > 5:
            recommendations.append("🚀 Consider promotion or lateral growth opportunities")
        if training_times < 2:
            recommendations.append("📚 Increase training and development investment")
        if not recommendations:
            recommendations.append("✅ Employee profile looks healthy — maintain current engagement levels")

        for rec in recommendations:
            st.markdown(f'<div class="insight-box">{rec}</div>', unsafe_allow_html=True)

with tab2:
    st.subheader("📊 Company-Wide Attrition Analysis")
    st.markdown("*Analyzing synthetic HR dataset of 1,000 employees*")

    if st.button("🔄 Run Batch Analysis", type="primary"):
        with st.spinner("Analyzing all employees..."):
            df = generate_hr_data(500)
            le_d = le_dept
            le_r = le_role
            df["Department_encoded"] = le_d.transform(df["Department"])
            df["JobRole_encoded"] = le_r.transform(df["JobRole"])

            probs = model.predict_proba(df[FEATURE_COLS])[:, 1]
            df["AttritionRisk"] = probs
            df["RiskCategory"] = pd.cut(probs,
                bins=[0, 0.3, 0.5, 0.75, 1.0],
                labels=["Low", "Moderate", "High", "Critical"])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Risk Distribution**")
            risk_counts = df["RiskCategory"].value_counts()
            st.bar_chart(risk_counts)

        with col2:
            st.markdown("**Attrition Risk by Department**")
            dept_risk = df.groupby("Department")["AttritionRisk"].mean().sort_values(ascending=False)
            st.bar_chart(dept_risk)

        high_risk = df[df["AttritionRisk"] > 0.5].sort_values("AttritionRisk", ascending=False).head(10)
        st.markdown("**🚨 Top 10 High-Risk Employees**")
        st.dataframe(
            high_risk[["Department", "JobRole", "MonthlyIncome", "JobSatisfaction", "YearsAtCompany", "AttritionRisk"]]
            .rename(columns={"AttritionRisk": "Risk Score"})
            .assign(**{"Risk Score": lambda x: x["Risk Score"].map("{:.1%}".format)}),
            use_container_width=True, hide_index=True
        )

with tab3:
    st.subheader("📈 Model Performance & Feature Importance")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Accuracy", f"{accuracy}%")
        st.metric("Algorithm", "Gradient Boosting Classifier")
        st.metric("Training Samples", "800 employees")
        st.metric("Test Samples", "200 employees")

    with col2:
        st.markdown("**Top Feature Importances:**")
        top_features = dict(list(importance.items())[:8])
        feat_df = pd.DataFrame(list(top_features.items()), columns=["Feature", "Importance"])
        feat_df["Importance"] = feat_df["Importance"].map("{:.1%}".format)
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

st.divider()
st.markdown(
    "Built by [Madhavi Akella](https://linkedin.com/in/madhavi-akella-2b8213114) · "
    "[GitHub](https://github.com/madhavi-akella-lab) · "
    "Powered by scikit-learn"
)
