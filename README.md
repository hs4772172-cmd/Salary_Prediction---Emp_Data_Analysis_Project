💼 Salary Prediction & Employee Data Analysis

An end-to-end data science project that performs Exploratory Data Analysis (EDA) on employee data and builds regression models to predict employee salaries based on factors such as experience, education, job title, and department.


📌 Project Overview

This project analyzes an employee dataset to uncover salary trends and performance patterns, and develops machine learning regression models that accurately predict salaries. The goal is to help HR teams and analysts understand the key drivers of compensation and benchmark fair salary ranges.


🎯 Objectives


Perform in-depth Exploratory Data Analysis (EDA) on employee data
Identify salary trends, performance patterns, and feature correlations
Build and evaluate regression models (Linear Regression, Random Forest) to predict salaries
Identify the top features that influence salary
Visualize insights through charts and an interactive dashboard



🗂️ Dataset

The dataset contains employee-level records with attributes such as:

FeatureDescriptionAgeEmployee ageGenderEmployee genderEducation LevelHighest qualificationJob TitleCurrent role/designationDepartmentDepartment/teamYears of ExperienceTotal years of work experiencePerformance RatingLatest performance scoreSalaryTarget variable (annual salary)


Replace this section with the exact column names and source of your dataset.




🛠️ Tech Stack


Language: Python
Data Handling: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Machine Learning: Scikit-learn (Linear Regression, Random Forest Regressor)
Evaluation Metrics: R² Score, RMSE
Dashboard: Streamlit
Environment: Jupyter Notebook



🔍 Project Workflow


Data Cleaning & Preprocessing

Handled missing values and duplicates
Encoded categorical variables (Label Encoding / One-Hot Encoding)
Scaled numerical features where required



Exploratory Data Analysis (EDA)

Distribution analysis of salary, age, and experience
Correlation heatmaps between features
Department-wise and role-wise salary comparisons
Identified outliers and salary trends across experience levels



Model Building

Trained Linear Regression as a baseline model
Trained Random Forest Regressor for improved accuracy
Performed train-test split and cross-validation



Model Evaluation

Compared models using R² Score and RMSE
Random Forest outperformed Linear Regression in predictive accuracy
Extracted feature importance to identify top salary-influencing factors



Insights & Visualization

Built visual dashboards to communicate findings
Highlighted key drivers: Experience, Department, and Performance Rating






📊 Key Insights


Years of Experience is the strongest predictor of salary
Department and Job Title significantly impact compensation bands
Random Forest achieved a higher R² score and lower RMSE compared to Linear Regression
Salary growth is non-linear — it accelerates after a certain experience threshold



📈 Interactive Dashboard

An interactive Streamlit dashboard (app.py) is included, allowing users to:


Explore salary distributions and trends visually
Filter data by department, education level, and experience
Input employee details and get a predicted salary in real time
View feature importance and model performance metrics


Run the dashboard locally

bash# 1. Clone the repository
git clone https://github.com/hs4772172-cmd/Salary_Prediction---Emp_Data_Analysis_Project.git
cd Salary_Prediction---Emp_Data_Analysis_Project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py


📁 Repository Structure

Salary_Prediction---Emp_Data_Analysis_Project/
│
├── SAL-PRED&EMP-DA.ipynb   # Main notebook: EDA + Model building
├── app.py                  # Streamlit interactive dashboard
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation


🚀 Future Improvements


Add more advanced models (XGBoost, Gradient Boosting)
Hyperparameter tuning with GridSearchCV
Deploy the dashboard on Streamlit Cloud for public access
Add SHAP-based explainability for predictions



👤 Author

Harsh Soni
BCA Graduate | Data Analyst | Python Developer
📧 hs4772172@gmail.com
🔗 LinkedIn | GitHub
