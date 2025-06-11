# Feature Engineering for AI Job Salary Analysis (Julia)
#
# This script encodes categorical variables, standardizes numerical features,
# splits data into train/test sets, and exports results to julia_results/.

using CSV
using DataFrames
using MLJ
using MLJBase
using Statistics
using Random

# Ensure results directory exists
results_dir = "../julia_results"
isdir(results_dir) || mkpath(results_dir)

# Load cleaned data
df = CSV.read("../data/ai_job_dataset_clean.csv", DataFrame)

# 1. Select features
categorical_features = [
    :job_title, :experience_level, :employment_type,
    :company_location, :company_size, :employee_residence,
    :education_required, :industry
]
numerical_features = [
    :years_experience, :remote_ratio, :job_description_length, :benefits_score
]

# 2. One-hot encode categorical features
X_cat = DataFrames.select(df, categorical_features)
X_num = DataFrames.select(df, numerical_features)
y = df.salary_usd

# One-hot encoding
ohe = OneHotEncoder()
mach_ohe = machine(ohe, X_cat) |> fit!
X_cat_ohe = MLJ.transform(mach_ohe, X_cat)

# 3. Standardize numerical features
stand = Standardizer()
mach_stand = machine(stand, X_num) |> fit!
X_num_std = MLJ.transform(mach_stand, X_num)

# 4. Combine all features
X = hcat(X_cat_ohe, X_num_std)

# 5. Train-test split (80/20)
Random.seed!(42)
n = nrow(X)
idx = shuffle(1:n)
n_train = Int(floor(0.8 * n))
train_idx = idx[1:n_train]
test_idx = idx[n_train+1:end]

X_train = X[train_idx, :]
X_test = X[test_idx, :]
y_train = y[train_idx]
y_test = y[test_idx]

# 6. Export to CSV
CSV.write(joinpath(results_dir, "X_train.csv"), X_train)
CSV.write(joinpath(results_dir, "X_test.csv"), X_test)
CSV.write(joinpath(results_dir, "y_train.csv"), DataFrame(y_train=y_train))
CSV.write(joinpath(results_dir, "y_test.csv"), DataFrame(y_test=y_test))

println("Feature engineering completed. Data exported to julia_results/")