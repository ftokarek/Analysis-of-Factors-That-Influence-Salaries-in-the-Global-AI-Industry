## 🔬 Analysis of Factors That Influence Salaries in the Global AI Industry

This project performs a comprehensive regression analysis to identify the most important factors shaping salaries in the global AI job market. It was developed as the final assignment for the SAD course (2024/2025) at AGH University of Science and Technology.

## 📌 Project Goals

- Identify key drivers of AI industry compensation worldwide.
- Build interpretable and accurate regression models.
- Compare workflows in Python and Julia.
- Present data-driven recommendations for professionals and employers.

## 📊 Dataset

- **Source**: [Kaggle - Global AI Job Market and Salary Trends 2025](https://www.kaggle.com/datasets/bismasajjad/global-ai-job-market-and-salary-trends-2025/data)
- **Cleaned version** used in analysis: `data/ai_job_dataset_clean.csv`

## ⚙️ Technologies Used

- **Python**: `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`
- **Julia**: `DataFrames.jl`, `GLM.jl`, `DecisionTree.jl`
- **Jupyter**, **VS Code**, **Git**, **GitHub**

## 📈 Models Trained

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor

Model performance was evaluated using **R²**, **MAE**, **MSE**, and **RMSE**. Feature importance was extracted for tree-based models.

## 🔍 Key Insights

- Salary is primarily driven by:
  - **Experience level**
  - **Job title**
  - **Country of employment**
- Random Forest produced the most accurate predictions and revealed nonlinear interactions.
- Julia was used to validate results and compare model capabilities across environments.

## 📄 Final Report

- ✅ All analysis steps documented
- ✅ All figures and results exported
- ✅ Includes Python vs Julia comparison

📁 Report files:  
- [`reports/Tokarek_Franciszek_AI_Salary_Regression_Report_2025.docx`](reports/Tokarek_Franciszek_AI_Salary_Regression_Report_2025.docx)  
- [`reports/AI_Salary_Regression_Report.pdf`](reports/AI_Salary_Regression_Report.pdf)

## 👤 Author

**Franciszek Tokarek**  
- GitHub: [@ftokarek](https://github.com/ftokarek)
- LinkedIn: [linkedin.com/in/franciszektokarek](https://www.linkedin.com/in/franciszektokarek/)
