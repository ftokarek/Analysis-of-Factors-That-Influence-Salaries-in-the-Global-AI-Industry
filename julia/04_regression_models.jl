# Regression Models for AI Job Salary Analysis (Julia)
#
# This script loads prepared data, fits linear and tree-based regressors,
# evaluates them, and exports results and plots to julia_results/.

using CSV
using DataFrames
using MLJ
using MLJLinearModels
using MLJDecisionTreeInterface
using Statistics
using StatsPlots

# Ensure results directory exists
results_dir = "../julia_results"
isdir(results_dir) || mkpath(results_dir)

# Load prepared data
X_train = CSV.read(joinpath(results_dir, "X_train.csv"), DataFrame)
X_test = CSV.read(joinpath(results_dir, "X_test.csv"), DataFrame)

# Remove non-numeric columns (e.g. strings)
X_train = select(X_train, names(X_train, eltype.(eachcol(X_train)) .<: Real))
X_test = select(X_test, names(X_test, eltype.(eachcol(X_test)) .<: Real))

# Load target values and cast to Float64
y_train = Float64.(CSV.read(joinpath(results_dir, "y_train.csv"), DataFrame)[:, 1])
y_test = Float64.(CSV.read(joinpath(results_dir, "y_test.csv"), DataFrame)[:, 1])

# 1. Linear Regression
lin_model = @load LinearRegressor pkg=MLJLinearModels
lin = machine(lin_model(), X_train, y_train) |> fit!
y_pred_lin = collect(predict(lin, X_test))

# 2. Decision Tree Regressor
tree_model = @load DecisionTreeRegressor pkg=DecisionTree
tree = machine(tree_model(), X_train, y_train) |> fit!
y_pred_tree = collect(predict(tree, X_test))

# 3. Random Forest Regressor
forest_model = @load RandomForestRegressor pkg=DecisionTree
forest = machine(forest_model(), X_train, y_train) |> fit!
y_pred_forest = collect(predict(forest, X_test))

# 4. Evaluation metrics
function regression_metrics(y_true, y_pred)
    r2 = 1 - sum((y_true .- y_pred).^2) / sum((y_true .- mean(y_true)).^2)
    mae = mean(abs.(y_true .- y_pred))
    mse = mean((y_true .- y_pred).^2)
    rmse = sqrt(mse)
    return (r2=r2, mae=mae, mse=mse, rmse=rmse)
end

metrics = [
    regression_metrics(y_test, y_pred_lin),
    regression_metrics(y_test, y_pred_tree),
    regression_metrics(y_test, y_pred_forest)
]

results = DataFrame(
    Model = ["Linear Regression", "Decision Tree", "Random Forest"],
    r2 = [m.r2 for m in metrics],
    mae = [m.mae for m in metrics],
    mse = [m.mse for m in metrics],
    rmse = [m.rmse for m in metrics]
)

CSV.write(joinpath(results_dir, "model_comparison.csv"), results)
println(results)

# 5. Plot R2 comparison
@df results bar(
    :Model, :r2,
    title="R2 Score by Model",
    ylabel="R2 Score",
    legend=false,
    dpi=150,
    size=(600,400)
)
savefig(joinpath(results_dir, "model_r2_comparison.png"))

# 6. Actual vs Predicted for Random Forest
@df DataFrame(actual=y_test, predicted=y_pred_forest) scatter(
    :actual, :predicted,
    title="Random Forest: Actual vs Predicted Salary",
    xlabel="Actual Salary",
    ylabel="Predicted Salary",
    legend=false,
    dpi=150,
    size=(500,400)
)
savefig(joinpath(results_dir, "random_forest_actual_vs_predicted.png"))

# 7. Residuals for Random Forest
residuals = y_test .- y_pred_forest
@df DataFrame(residuals=residuals) histogram(
    :residuals,
    bins=50,
    title="Random Forest: Residuals Distribution",
    xlabel="Residual",
    legend=false,
    dpi=150,
    size=(500,400)
)
savefig(joinpath(results_dir, "random_forest_residuals.png"))

println("Regression modeling completed. Results and plots saved to julia_results/")
