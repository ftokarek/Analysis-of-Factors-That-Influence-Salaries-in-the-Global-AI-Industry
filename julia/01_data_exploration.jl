# Data Exploration for AI Job Salary Analysis (Julia)
#
# This script loads the cleaned dataset, performs basic exploration,
# and generates key visualizations, saving them to julia_results/.

using CSV
using DataFrames
using StatsPlots
using Statistics

# Ensure results directory exists
results_dir = "../julia_results"
isdir(results_dir) || mkpath(results_dir)

# Load cleaned data
df = CSV.read("../data/ai_job_dataset_clean.csv", DataFrame)

# Basic info
println("Shape: ", size(df))
println("Column names: ", names(df))
describe(df) |> println

# Histogram for salary_usd
@df df histogram(
    :salary_usd,
    bins=50,
    title="Salary Distribution",
    xlabel="Salary (USD)",
    ylabel="Count",
    legend=false,
    dpi=150,
    size=(600,400)
)
savefig(joinpath(results_dir, "salary_distribution.png"))

# Boxplot for salary_usd
@df df boxplot(
    fill(1, nrow(df)), :salary_usd,
    title="Salary Boxplot",
    ylabel="Salary (USD)",
    xticks=false,
    legend=false,
    dpi=150,
    size=(400,400)
)
savefig(joinpath(results_dir, "salary_boxplot.png"))

# Boxplot: Salary by Experience Level
@df df boxplot(
    :experience_level, :salary_usd,
    title="Salary by Experience Level",
    xlabel="Experience Level",
    ylabel="Salary (USD)",
    legend=false,
    dpi=150,
    size=(600,400)
)
savefig(joinpath(results_dir, "salary_by_experience_level.png"))

# Boxplot: Salary by Job Title (top 10)
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
savefig(joinpath(results_dir, "salary_by_top10_job_title.png"))

# Boxplot: Salary by Company Location (top 10)
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
savefig(joinpath(results_dir, "salary_by_top10_company_location.png"))

println("Exploration and plots completed. Results saved to julia_results/")