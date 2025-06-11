# Model Evaluation for AI Job Salary Analysis (Julia)
#
# This script analyzes regression coefficients, residuals,
# and exports all results and plots to julia_results/.

using CSV
using DataFrames
using MLJ
using MLJLinearModels
using Statistics
using StatsPlots
using Plots

# Ensure results directory exists
results_dir = "../julia_results"
isdir(results_dir) || mkpath(results_dir)

# Load prepared data
X_train = CSV.read(joinpath(results_dir, "X_train.csv"), DataFrame)
X_test = CSV.read(joinpath(results_dir, "X_test.csv"), DataFrame)
y_train = Float64.(CSV.read(joinpath(results_dir, "y_train.csv"), DataFrame)[:, 1])
y_test = Float64.(CSV.read(joinpath(results_dir, "y_test.csv"), DataFrame)[:, 1])

# Remove non-numeric columns (just in case)
num_cols = findall(c -> eltype(c) <: Number, eachcol(X_train))
X_train = X_train[:, num_cols]
X_test = X_test[:, num_cols]

# 1. Linear Regression - coefficients
lin_model = @load LinearRegressor pkg=MLJLinearModels
lin = machine(lin_model(), X_train, y_train) |> fit!
coefs = fitted_params(lin).coefs
coef_names = names(X_train)
coef_df = DataFrame(feature=coef_names, coefficient=coefs)
CSV.write(joinpath(results_dir, "linear_regression_coefficients.csv"), coef_df)

# 2. Residuals analysis (Linear Regression)
y_pred_lin = collect(predict(lin, X_test))
residuals = y_test .- y_pred_lin
@df DataFrame(residuals=residuals) histogram(
    :residuals,
    bins=50,
    title="Linear Regression: Residuals Distribution",
    xlabel="Residual",
    legend=false,
    dpi=150,
    size=(500,400)
)
savefig(joinpath(results_dir, "lin_residuals_eval.png"))

# 3. Actual vs Predicted (Linear Regression)
@df DataFrame(actual=y_test, predicted=y_pred_lin) scatter(
    :actual, :predicted,
    title="Linear Regression: Actual vs Predicted Salary",
    xlabel="Actual Salary",
    ylabel="Predicted Salary",
    legend=false,
    dpi=150,
    size=(500,400)
)
savefig(joinpath(results_dir, "lin_actual_vs_predicted_eval.png"))

println("Model evaluation completed. Results and plots saved to julia_results/")