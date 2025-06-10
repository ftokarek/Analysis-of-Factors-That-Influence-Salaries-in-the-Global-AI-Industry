# 02_modeling.jl — Regression Modeling in Julia

using CSV, DataFrames
using Statistics
using GLM

println("Loading dataset...")
df = CSV.read("data/ai_job_dataset_encoded.csv", DataFrame)

# -------------------------------
# Multiple Linear Regression
# -------------------------------

println("\nFitting multiple linear regression model...")
model_formula = @formula(salary_usd ~ years_experience + benefits_score + remote_ratio)
ols_model = lm(model_formula, df)

println(coeftable(ols_model))

# Evaluation metrics
function evaluate_model(y_true, y_pred)
    n = length(y_true)
    mae = mean(abs.(y_true .- y_pred))
    rmse = sqrt(mean((y_true .- y_pred).^2))
    r2 = 1 - sum((y_true .- y_pred).^2) / sum((y_true .- mean(y_true)).^2)
    return mae, rmse, r2
end

println("\nLinear Regression Evaluation:")
y_pred = predict(ols_model)
y_true = df.salary_usd
mae, rmse, r2 = evaluate_model(y_true, y_pred)

println("MAE: $mae")
println("RMSE: $rmse")
println("R^2: $r2")

# -------------------------------
# Note:
# Tree-based models (e.g., DecisionTreeRegressor)
# were omitted in this Julia notebook due to compatibility
# and namespace conflicts in script execution context.
# These models are fully implemented and analyzed in the Python version.
# -------------------------------
