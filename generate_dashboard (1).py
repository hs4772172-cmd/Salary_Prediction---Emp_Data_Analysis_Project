"""
Salary Prediction & Employee Data Analysis - Dashboard Generator
Author: Harsh Soni

Matches the data generation and modeling approach used in
SAL-PRED&EMP-DA.ipynb (synthetic employee data via Faker + Linear
Regression / Random Forest salary prediction).

Generates a single self-contained HTML dashboard (salary_dashboard.html).

Usage:
    python generate_dashboard.py
"""

import pandas as pd
import numpy as np
import random
from faker import Faker
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ────────────────────────────────────────────────────────────
# 1. GENERATE SYNTHETIC DATASET (same logic as the notebook)
# ────────────────────────────────────────────────────────────
random.seed(42)
fake = Faker()
Faker.seed(42)

departments = {
    "HR": ["HR Executive", "Recruiter", "HR Manager"],
    "Finance": ["Accountant", "Finance Analyst", "Finance Manager"],
    "IT": ["Software Engineer", "QA Engineer", "System Administrator"],
    "Marketing": ["Marketing Analyst", "Marketing Manager", "Content Strategist"],
    "Sales": ["Sales Executive", "Sales Manager", "Business Development"],
    "Operations": ["Operations Executive", "Operations Lead", "Logistics Manager"],
}

employees = []
for emp_id in range(1, 1501):
    dept = random.choice(list(departments.keys()))
    job = random.choice(departments[dept])
    employees.append({
        "EmployeeID": f"{emp_id:04d}",
        "Name": fake.name(),
        "Age": random.randint(22, 60),
        "Gender": random.choice(["M", "F"]),
        "Department": dept,
        "Job Title": job,
        "Date of Joining": fake.date_between(start_date="-10y", end_date="today"),
        "Salary": random.randint(30000, 120000),
        "Location": fake.city(),
    })

df = pd.DataFrame(employees)
df["Date of Joining"] = pd.to_datetime(df["Date of Joining"])
df["YearOfJoining"] = df["Date of Joining"].dt.year
df.to_csv("employee_data.csv", index=False)
print(f"Generated {len(df)} employee records")

# ────────────────────────────────────────────────────────────
# 2. MODEL TRAINING (Age, Gender, Department, Location -> Salary)
# ────────────────────────────────────────────────────────────
X = df[["Age", "Gender", "Department", "Location"]]
y = df["Salary"]
X_enc = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2, random_state=42)

lr = LinearRegression().fit(X_train, y_train)
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)

model_results = {
    "Linear Regression": {
        "MSE": round(mean_squared_error(y_test, y_pred_lr), 2),
        "R2": round(r2_score(y_test, y_pred_lr), 3),
    },
    "Random Forest": {
        "MSE": round(mean_squared_error(y_test, y_pred_rf), 2),
        "R2": round(r2_score(y_test, y_pred_rf), 3),
    },
}
print("\nModel performance:")
for name, res in model_results.items():
    print(f"  {name}: R2={res['R2']}, MSE={res['MSE']:,.0f}")

importance = pd.Series(rf.feature_importances_, index=X_enc.columns).sort_values(ascending=False).head(10)

# ────────────────────────────────────────────────────────────
# 3. KPI METRICS
# ────────────────────────────────────────────────────────────
kpis = {
    "Total Employees": f"{len(df):,}",
    "Average Salary": f"₹{df['Salary'].mean():,.0f}",
    "Median Salary": f"₹{df['Salary'].median():,.0f}",
    "Departments": f"{df['Department'].nunique()}",
    "Random Forest R²": f"{model_results['Random Forest']['R2']}",
    "Random Forest RMSE": f"₹{np.sqrt(model_results['Random Forest']['MSE']):,.0f}",
}

# ────────────────────────────────────────────────────────────
# 4. CHARTS
# ────────────────────────────────────────────────────────────
fig_dist = px.histogram(df, x="Salary", nbins=30, title="Salary Distribution",
                         color_discrete_sequence=["#2E75B6"])
fig_dist.update_layout(template="plotly_white", bargap=0.05)

fig_box = px.box(df, x="Salary", title="Salary Spread (Boxplot)",
                  color_discrete_sequence=["#D85A30"])
fig_box.update_layout(template="plotly_white")

dept_avg = df.groupby("Department")["Salary"].mean().sort_values(ascending=False).reset_index()
fig_dept = px.bar(dept_avg, x="Department", y="Salary", title="Average Salary by Department",
                   color="Salary", color_continuous_scale="Blues")
fig_dept.update_layout(template="plotly_white", coloraxis_showscale=False)

gender_dept = df.groupby(["Department", "Gender"]).size().reset_index(name="Count")
fig_gender = px.bar(gender_dept, x="Department", y="Count", color="Gender", barmode="stack",
                     title="Gender Distribution by Department",
                     color_discrete_sequence=["#378ADD", "#D4537E"])
fig_gender.update_layout(template="plotly_white")

join_counts = df["YearOfJoining"].value_counts().sort_index().reset_index()
join_counts.columns = ["Year", "Count"]
join_counts["Cumulative"] = join_counts["Count"].cumsum()
fig_growth = px.area(join_counts, x="Year", y="Cumulative", title="Workforce Growth Over Time",
                      color_discrete_sequence=["#7F77DD"])
fig_growth.update_layout(template="plotly_white")

fig_joins = px.line(join_counts, x="Year", y="Count", markers=True,
                     title="Employees Joined Per Year", color_discrete_sequence=["#1D9E75"])
fig_joins.update_layout(template="plotly_white")

corr = df[["Age", "Salary"]].corr()
fig_corr = px.imshow(corr, text_auto=".2f", title="Correlation Heatmap (Age vs Salary)",
                      color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")

pred_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred_rf})
fig_actual_pred = px.scatter(pred_df, x="Actual", y="Predicted",
                              title="Actual vs Predicted Salary (Random Forest)",
                              opacity=0.5, color_discrete_sequence=["#378ADD"])
min_v, max_v = df["Salary"].min(), df["Salary"].max()
fig_actual_pred.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines",
                                      name="Ideal", line=dict(color="#D85A30", dash="dash")))
fig_actual_pred.update_layout(template="plotly_white")

fig_importance = px.bar(x=importance.values, y=importance.index, orientation="h",
                         title="Top 10 Feature Importances (Random Forest)",
                         labels={"x": "Importance", "y": "Feature"},
                         color=importance.values, color_continuous_scale="Greens")
fig_importance.update_layout(template="plotly_white", coloraxis_showscale=False,
                              yaxis={"categoryorder": "total ascending"})

comp_df = pd.DataFrame(model_results).T.reset_index().rename(columns={"index": "Model"})
fig_comp = make_subplots(rows=1, cols=2, subplot_titles=("R² Score (higher = better, 0 = baseline)", "MSE (lower = better)"))
fig_comp.add_trace(go.Bar(x=comp_df["Model"], y=comp_df["R2"], marker_color="#2E75B6", showlegend=False), row=1, col=1)
fig_comp.add_trace(go.Bar(x=comp_df["Model"], y=comp_df["MSE"], marker_color="#D85A30", showlegend=False), row=1, col=2)
fig_comp.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
fig_comp.update_layout(template="plotly_white", title_text="Model Performance Comparison")

# ────────────────────────────────────────────────────────────
# 5. BUILD HTML DASHBOARD
# ────────────────────────────────────────────────────────────
def fig_to_div(fig, div_id):
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)

charts_html = "\n".join([
    f'<div class="chart-card">{fig_to_div(fig_dist, "dist")}</div>',
    f'<div class="chart-card">{fig_to_div(fig_box, "box")}</div>',
    f'<div class="chart-card">{fig_to_div(fig_dept, "dept")}</div>',
    f'<div class="chart-card">{fig_to_div(fig_gender, "gender")}</div>',
    f'<div class="chart-card">{fig_to_div(fig_growth, "growth")}</div>',
    f'<div class="chart-card">{fig_to_div(fig_joins, "joins")}</div>',
    f'<div class="chart-card">{fig_to_div(fig_corr, "corr")}</div>',
    f'<div class="chart-card">{fig_to_div(fig_actual_pred, "actualpred")}</div>',
    f'<div class="chart-card">{fig_to_div(fig_importance, "importance")}</div>',
    f'<div class="chart-card full-width">{fig_to_div(fig_comp, "comp")}</div>',
])

kpi_html = "\n".join([
    f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>'
    for label, value in kpis.items()
])

html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Salary Prediction & Employee Data Analysis Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6fa; color: #1f2937; padding: 24px; }}
  header {{ text-align: center; margin-bottom: 16px; }}
  header h1 {{ font-size: 28px; color: #1F4E79; margin-bottom: 4px; }}
  header p {{ color: #6b7280; font-size: 14px; }}
  .note {{ background: #fff7e6; border: 1px solid #f5c518; border-radius: 10px; padding: 12px 16px;
    font-size: 13px; color: #6b5a1e; max-width: 900px; margin: 0 auto 24px; }}
  .kpi-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16px; margin-bottom: 24px; }}
  .kpi-card {{ background: white; border-radius: 12px; padding: 16px; text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08); border-left: 4px solid #2E75B6; }}
  .kpi-label {{ font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }}
  .kpi-value {{ font-size: 22px; font-weight: 700; color: #1F4E79; }}
  .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 16px; }}
  .chart-card {{ background: white; border-radius: 12px; padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .full-width {{ grid-column: 1 / -1; }}
  footer {{ text-align: center; margin-top: 32px; color: #9ca3af; font-size: 12px; }}
</style>
</head>
<body>
<header>
  <h1>Salary Prediction &amp; Employee Data Analysis Dashboard</h1>
  <p>Built with Python &middot; Pandas &middot; Scikit-learn &middot; Faker &middot; Plotly</p>
</header>

<div class="note">
  <strong>Note:</strong> This dataset is synthetically generated (via Faker) with salary assigned
  randomly and independently of other features. As a result, the regression models show low/negative
  R&sup2; scores &mdash; this is expected and demonstrates an important EDA finding: <em>the available
  features (Age, Gender, Department, Location) do not explain salary variation in this dataset</em>.
  In a real-world dataset, salary would correlate with experience, role, and performance, and the
  same pipeline (preprocessing &rarr; EDA &rarr; model training &rarr; evaluation) would be applied
  to produce a meaningful predictive model.
</div>

<div class="kpi-row">
{kpi_html}
</div>

<div class="chart-grid">
{charts_html}
</div>

<footer>
  Generated by generate_dashboard.py &middot; Harsh Soni &middot; github.com/hs4772172-cmd
</footer>
</body>
</html>
"""

with open("salary_dashboard.html", "w", encoding="utf-8") as f:
    f.write(html_template)

print("\nDashboard saved as salary_dashboard.html")
