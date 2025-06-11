# Statistical Analysis for AI Job Salary Analysis (Julia)
#
# This script computes correlations, creates heatmaps and scatterplots,
# checks normality, and exports all results to julia_results/.

using CSV
using DataFrames
using StatsPlots
using Statistics
using HypothesisTests
using Distributions
using Random

# Ensure results directory exists
results_dir = "../julia_results"
isdir(results_dir) || mkpath(results_dir)

# Load cleaned data
df = CSV.read("../data/ai_job_dataset_clean.csv", DataFrame)

# 1. Correlation matrix for numerical features
num_cols = [:salary_usd, :years_experience, :remote_ratio, :job_description_length, :benefits_score]
corr_matrix = cor(Matrix(df[:, num_cols]))
corr_df = DataFrame(corr_matrix, Symbol.(string.(num_cols)))
corr_df.Row = num_cols
select!(corr_df, [:Row, num_cols...])
CSV.write(joinpath(results_dir, "correlation_matrix.csv"), corr_df)

# Heatmap
heatmap(
    string.(num_cols), string.(num_cols), corr_matrix;
    c=:coolwarm, title="Correlation Matrix", size=(500,400), dpi=150, right_margin=10Plots.mm
)
savefig(joinpath(results_dir, "correlation_matrix.png"))

# 2. Scatter plots: salary vs. years_experience, salary vs. benefits_score
@df df scatter(
    :years_experience, :salary_usd,
    title="Salary vs. Years of Experience",
    xlabel="Years of Experience",
    ylabel="Salary (USD)",
    legend=false,
    dpi=150,
    size=(500,400)
)
savefig(joinpath(results_dir, "salary_vs_years_experience.png"))

@df df scatter(
    :benefits_score, :salary_usd,
    title="Salary vs. Benefits Score",
    xlabel="Benefits Score",
    ylabel="Salary (USD)",
    legend=false,
    dpi=150,
    size=(500,400)
)
savefig(joinpath(results_dir, "salary_vs_benefits_score.png"))

# 3. Normality check for salary_usd (Shapiro-Wilk test)
salary_vec = Float64.(collect(skipmissing(df.salary_usd)))
salary_sample = sample(salary_vec, min(500, length(salary_vec)); replace=false)
shapiro_result = try
    HypothesisTests.shapiro_wilk_test(salary_sample)
catch
    nothing
end

if shapiro_result !== nothing
    println("Shapiro-Wilk W: ", shapiro_result.W, ", p-value: ", shapiro_result.pvalue)
    open(joinpath(results_dir, "shapiro_wilk_salary.txt"), "w") do io
        println(io, "Shapiro-Wilk W: ", shapiro_result.W)
        println(io, "p-value: ", shapiro_result.pvalue)
    end
else
    println("Shapiro-Wilk test could not be performed.")
end

# QQ-plot for salary_usd
qqplot(
    df.salary_usd,
    Normal(mean(df.salary_usd), std(df.salary_usd));
    title="QQ-plot for Salary",
    dpi=150,
    size=(500,400)
)
savefig(joinpath(results_dir, "qqplot_salary.png"))

# 4. Grouped boxplots for categorical features
# Salary by job_title (top 10)
job_counts = combine(groupby(df, :job_title), nrow => :count)
top_jobs = sort(job_counts, :count, rev=true)[1:10, :]
top_job_titles = Vector(top_jobs.job_title)
df_top_jobs = filter(row -> row.job_title in top_job_titles, df)
@df df_top_jobs boxplot(
    :job_title, :salary_usd,
    title="Salary by Top 10 Job Titles",
    xlabel="Job Title",
    ylabel="Salary (USD)",
    legend=false,
    rotation=45,
    dpi=150,
    size=(800,400)
)
savefig(joinpath(results_dir, "salary_by_top10_job_title_stat.png"))

# Salary by company_location (top 10)
country_counts = combine(groupby(df, :company_location), nrow => :count)
top_countries = sort(country_counts, :count, rev=true)[1:10, :]
top_country_codes = Vector(top_countries.company_location)
df_top_countries = filter(row -> row.company_location in top_country_codes, df)
@df df_top_countries boxplot(
    :company_location, :salary_usd,
    title="Salary by Top 10 Company Locations",
    xlabel="Company Location",
    ylabel="Salary (USD)",
    legend=false,
    rotation=45,
    dpi=150,
    size=(800,400)
)
savefig(joinpath(results_dir, "salary_by_top10_company_location_stat.png"))

println("Statistical analysis and plots completed. Results saved to julia_results/")