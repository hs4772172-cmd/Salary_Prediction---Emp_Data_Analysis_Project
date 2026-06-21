# 💼 Salary Prediction & Employee Data Analysis

An end-to-end data analysis and machine learning project that generates a synthetic employee dataset, performs **Exploratory Data Analysis (EDA)**, and builds **regression models** to study salary patterns across departments, demographics, and tenure.

---

## 📌 Project Overview

This project simulates an HR dataset of 1,500 employees using the `Faker` library, then applies data cleaning, visualization, and machine learning (Linear Regression & Random Forest) to analyze and attempt to predict salaries based on employee attributes.

---

## 🗂️ Dataset

The dataset is synthetically generated with the following columns:

| Column | Description |
|---|---|
| EmployeeID | Unique 4-digit employee ID |
| Name | Randomly generated employee name (Faker) |
| Age | 22–60 |
| Gender | M / F |
| Department | HR, Finance, IT, Marketing, Sales, Operations |
| Job Title | Role within the department |
| Date of Joining | Random date within the last 10 years |
| Salary | Random value between ₹30,000 – ₹120,000 |
| Location | Randomly generated city (Faker) |

> 📦 1,500 records generated using `random` and `Faker`, with a fixed seed for reproducibility.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Data Generation:** Faker
- **Data Handling:** Pandas
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning:** Scikit-learn (Linear Regression, Random Forest Regressor)
- **Evaluation Metrics:** MSE, R² Score
- **Dashboard:** Python + Plotly (standalone HTML)
- **Environment:** Jupyter Notebook

---

## 🔍 Project Workflow

1. **Synthetic Data Generation**
   - Generated 1,500 employee records with `Faker` (name, location) and `random` (age, gender, department, job title, joining date, salary)

2. **Preprocessing**
   - Selected features: `Age`, `Gender`, `Department`, `Location`
   - One-hot encoded categorical variables using `pd.get_dummies()`
   - Train/test split (80/20)

3. **Model Building**
   - Trained **Linear Regression** as a baseline model
   - Trained **Random Forest Regressor** (100 estimators) for comparison

4. **Model Evaluation**

   | Model | MSE | R² Score |
   |---|---|---|
   | Linear Regression | ~815M | ~ -0.20 |
   | Random Forest | ~769M | ~ -0.13 |

5. **Exploratory Data Analysis & Visualization**
   - Salary distribution (histogram + boxplot)
   - Average salary by department
   - Gender distribution by department
   - Workforce growth over time (cumulative joins by year)
   - Correlation heatmap (Age vs Salary)
   - Actual vs Predicted salary scatter plot
   - Residual plot
   - Feature importance (Random Forest)

---

## 📊 Key Insight

Because `Salary` is assigned **randomly and independently** of `Age`, `Gender`, `Department`, and `Location` in this synthetic dataset, both models produce **negative R² scores** — meaning they perform *worse than simply predicting the average salary* for every employee.

This is a genuine EDA finding, not a failure:

- The chosen features have **no real predictive relationship** with the target in this dataset
- A negative R² is a signal to **revisit feature selection / data quality**, not just tune the model
- In a real-world HR dataset, salary would correlate with factors like `Years of Experience`, `Job Title seniority`, and `Performance Rating` — and the **same pipeline** (EDA → preprocessing → model training → evaluation) would be used to build a meaningful predictive model

### Suggested Next Steps
- Add a realistic dependency between `Salary` and features (e.g., base salary + experience-based increment + department multiplier)
- Engineer `Years of Experience` from `Date of Joining`
- Include `Performance Rating` as a feature
- Try `GridSearchCV` for hyperparameter tuning once a meaningful signal exists

---

## 📈 Interactive Dashboard

A standalone Python-generated HTML dashboard (`generate_dashboard.py` → `salary_dashboard.html`) includes:

- KPI cards (employee count, average/median salary, R², RMSE)
- Salary distribution & spread
- Department-wise salary and gender breakdown
- Workforce growth over time
- Actual vs Predicted salary plot
- Feature importance and model comparison

### Run it

```bash
pip install pandas numpy faker plotly scikit-learn
python generate_dashboard.py
```

Then open `salary_dashboard.html` in your browser.

---

## 📁 Repository Structure

```
Salary_Prediction---Emp_Data_Analysis_Project/
│
├── SAL-PRED&EMP-DA.ipynb     # Main notebook: data generation, EDA, models
├── generate_dashboard.py     # Generates the interactive HTML dashboard
├── employee_data.csv         # Generated dataset (1,500 records)
├── salary_dashboard.html     # Interactive dashboard (open in browser)
└── README.md                 # Project documentation
```

---

## 👤 Author

**Harsh Soni**
BCA Graduate | Data Analyst | Python Developer
📧 hs4772172@gmail.com
🔗 [LinkedIn](https://linkedin.com/in/harshsoni) | [GitHub](https://github.com/hs4772172-cmd)

---

## ⭐ Show Your Support

If you found this project useful, consider giving it a ⭐ on GitHub!
