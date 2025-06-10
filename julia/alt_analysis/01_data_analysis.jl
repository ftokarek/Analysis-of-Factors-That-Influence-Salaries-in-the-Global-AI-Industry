# 01_data_analysis.jl
# Author: Franciszek Tokarek
# Purpose: Exploratory Data Analysis and Regression in Julia — Alternative Analysis
# ===============================================

using CSV
using DataFrames
using StatsPlots
using Statistics
using GLM
using HypothesisTests

# Load data
println("Loading dataset...")
df = CSV.read("data/ai_job_dataset_encoded.csv", DataFrame)
# Basic statistics
println("\nDescriptive statistics:")
println(describe(df[:, 1:5]))

# Histogram + Boxplot
@df df begin
    histogram(:salary_usd, bins=50, title="Salary Distribution", xlabel="Salary (USD)", ylabel="Count", legend=false)
    savefig("results/salary_hist_julia.png")
    
    boxplot(:salary_usd, title="Boxplot of Salaries", legend=false)
    savefig("results/salary_boxplot_julia.png")
end

# Correlation matrix
println("\nCorrelation Matrix:")
num_cols = filter(col -> eltype(df[!, col]) <: Real, names(df))
corr_matrix = cor(Matrix(df[:, num_cols]))
println(corr_matrix[1:5, 1:5])

# Linear regression
println("\nRunning linear regression model...")
model = lm(@formula(salary_usd ~ years_experience + benefits_score), df)
println(coeftable(model))

# Normality test
println("Shapiro-Wilk Test: Currently not supported in Julia HypothesisTests.jl.")

# End
println("\nDone.")