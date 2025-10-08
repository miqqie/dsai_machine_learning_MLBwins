# ===============================
# Baseball Wins Prediction Model
# ===============================

# Imports
import pandas as pd
import numpy as np
import hashlib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

# ===============================
# Data Load
# ===============================
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Map team IDs into test set (merge_asof acts like Excel VLOOKUP <=)
lookup_df = train_df[["ID", "teamID"]].drop_duplicates().sort_values("ID")
test_df = pd.merge_asof(test_df.sort_values("ID"), lookup_df, on="ID", direction="backward")
test_df = test_df.sort_index()  # restore order

# ===============================
# Encode Team IDs
# ===============================
def hash_team_id(team_id, n_bins=10):
    return int(hashlib.sha256(team_id.encode("utf-8")).hexdigest(), 16) % n_bins

train_df["teamID_encoded"] = train_df["teamID"].apply(lambda x: hash_team_id(x, n_bins=10))
test_df["teamID_encoded"] = test_df["teamID"].apply(lambda x: hash_team_id(x, n_bins=10))

# ===============================
# Feature Engineering
# ===============================
EPSILON = 1e-6

# Batting & Pitching Ratios
train_df["OBP"] = (train_df["H"] + train_df["BB"]) / (train_df["AB"] + train_df["BB"] + EPSILON)
test_df["OBP"] = (test_df["H"] + test_df["BB"]) / (test_df["AB"] + test_df["BB"] + EPSILON)

train_df["SLG"] = (train_df["H"] + 2*train_df["2B"] + 3*train_df["3B"] + 4*train_df["HR"]) / (train_df["AB"] + EPSILON)
test_df["SLG"] = (test_df["H"] + 2*test_df["2B"] + 3*test_df["3B"] + 4*test_df["HR"]) / (test_df["AB"] + EPSILON)

train_df["K_9"] = (train_df["SOA"] / (train_df["IPouts"]/3 + EPSILON)) * 9
test_df["K_9"] = (test_df["SOA"] / (test_df["IPouts"]/3 + EPSILON)) * 9

train_df["ERA_9"] = train_df["ERA"] * 9
test_df["ERA_9"] = test_df["ERA"] * 9

# Extra Stats
train_df["BA"] = train_df["H"] / (train_df["AB"] + EPSILON)
test_df["BA"] = test_df["H"] / (test_df["AB"] + EPSILON)

train_df["Pitching_to_Hitting_Ratio"] = train_df["ERA_9"] / (train_df["BA"] + EPSILON)
test_df["Pitching_to_Hitting_Ratio"] = test_df["ERA_9"] / (test_df["BA"] + EPSILON)

train_df["OPS"] = train_df["OBP"] + train_df["SLG"]
test_df["OPS"] = test_df["OBP"] + test_df["SLG"]

train_df["Run_Diff"] = train_df["R"] - train_df["RA"]
test_df["Run_Diff"] = test_df["R"] - test_df["RA"]

train_df["WHIP"] = (train_df["BBA"] + train_df["HA"]) / (train_df["IPouts"]/3 + EPSILON)
test_df["WHIP"] = (test_df["BBA"] + test_df["HA"]) / (test_df["IPouts"]/3 + EPSILON)

train_df["K_BB_Ratio"] = train_df["SOA"] / (train_df["BBA"] + EPSILON)
test_df["K_BB_Ratio"] = test_df["SOA"] / (test_df["BBA"] + EPSILON)

train_df["FER"] = 1 - (train_df["E"] / (train_df["G"]*9 + EPSILON))
test_df["FER"] = 1 - (test_df["E"] / (test_df["G"]*9 + EPSILON))

# Per-game features
per_game_cols = ["AB","H","2B","3B","SO","SB","ER","HA","HRA","BBA","SOA","E","DP","IPouts","CG","SV","SHO"]
for col in per_game_cols:
    train_df[f"{col}_per_G"] = train_df[col] / (train_df["G"] + EPSILON)
    test_df[f"{col}_per_G"] = test_df[col] / (test_df["G"] + EPSILON)

# Interaction Features
train_df["OPS_x_Run_Diff"] = train_df["OPS"] * train_df["Run_Diff"]
test_df["OPS_x_Run_Diff"] = test_df["OPS"] * test_df["Run_Diff"]

train_df["K_BB_Ratio_x_WHIP"] = train_df["K_BB_Ratio"] * train_df["WHIP"]
test_df["K_BB_Ratio_x_WHIP"] = test_df["K_BB_Ratio"] * test_df["WHIP"]

train_df["HR_per_G_x_BB_per_G"] = (train_df["HR"]/ (train_df["G"]+EPSILON)) * (train_df["BB"]/(train_df["G"]+EPSILON))
test_df["HR_per_G_x_BB_per_G"] = (test_df["HR"]/ (test_df["G"]+EPSILON)) * (test_df["BB"]/(test_df["G"]+EPSILON))

# ===============================
# Lag Features (Previous Season)
# ===============================
train_df.sort_values(by=["teamID", "yearID"], inplace=True)
features_to_lag = ["W", "Run_Diff"]

# Add lag-1 stats per team
for feature in features_to_lag:
    train_df[f"{feature}_lag1"] = train_df.groupby("teamID")[feature].shift(1)

# Impute missing lags with mean
for feature in features_to_lag:
    train_df[f"{feature}_lag1"].fillna(train_df[f"{feature}_lag1"].mean(), inplace=True)

# Apply to test set (using last season values if available, else mean)
latest_season = train_df[train_df["yearID"] == train_df["yearID"].max()]
lag_map = {f"{f}_lag1": latest_season.set_index("teamID")[f].to_dict() for f in features_to_lag}
for feature in features_to_lag:
    test_df[f"{feature}_lag1"] = test_df["ID"].map(lag_map[f"{feature}_lag1"]).fillna(train_df[f"{feature}_lag1"].mean())

# ===============================
# Model Training (Kernel Ridge)
# ===============================
# Features + Target
selected_features = [col for col in train_df.columns if col not in ["W", "yearID", "teamID"]]
X_train, y_train = train_df[selected_features].fillna(train_df.median()), train_df["W"]
X_test = test_df[selected_features].fillna(train_df.median())

# Pipeline: scaling + Kernel Ridge
pipeline_krr = Pipeline([
    ("scaler", StandardScaler()),
    ("krr", KernelRidge(kernel="rbf"))
])

# Hyperparameter grid
param_grid = {
    "krr__alpha": [0.005, 0.01, 0.05, 0.1],
    "krr__gamma": [0.001, 0.01, 0.1, 1.0]
}

grid_search = GridSearchCV(
    pipeline_krr,
    param_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best CV MAE:", -grid_search.best_score_)

# Train error
train_pred = grid_search.predict(X_train)
print("Train MAE:", mean_absolute_error(y_train, train_pred))

# ===============================
# Submission
# ===============================
test_pred = np.clip(np.round(grid_search.predict(X_test)).astype(int), 0, test_df["G"].max())
submission = pd.DataFrame({"ID": test_df["ID"], "Predicted_Wins": test_pred})
submission.to_csv("submission_krr.csv", index=False)
print("Submission saved: submission_krr.csv")
