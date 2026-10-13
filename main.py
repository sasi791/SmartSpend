
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project folders
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(42)

print("=" * 50)
print(" SMARTSPEND - PERSONAL EXPENSE ANALYSIS ")
print("=" * 50)

# 1. Create sample expense dataset
print("\n1. Creating sample expense data...")
dates = pd.date_range(start="2024-01-01", periods=730, freq="D")
categories = ["Food", "Travel", "Shopping", "Bills", "Education", "Entertainment"]
payments = ["Cash", "UPI", "Card"]

category_amounts = {
    "Food": (100, 700),
    "Travel": (50, 500),
    "Shopping": (200, 2500),
    "Bills": (300, 3000),
    "Education": (100, 1500),
    "Entertainment": (100, 1000),
}

data = []

for date in dates:
    daily_count = np.random.randint(1, 4)
    for _ in range(daily_count):
        category = np.random.choice(categories)
        low, high = category_amounts[category]
        amount = np.random.randint(low, high + 1)

        data.append({
            "Date": date,
            "Category": category,
            "Amount": amount,
            "Payment_Mode": np.random.choice(payments)
        })

df = pd.DataFrame(data)

# Save original dataset
df.to_csv(BASE_DIR / "expenses.csv", index=False)
print(f"Dataset created: {len(df)} expense records")

# 2. Data cleaning
print("\n2. Cleaning data...")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
df.dropna(subset=["Date", "Category", "Amount"], inplace=True)
df = df[df["Amount"] > 0].copy()
df.drop_duplicates(inplace=True)

df.to_csv(OUTPUT_DIR / "cleaned_expenses.csv", index=False)
print("Data cleaning completed.")

# 3. Exploratory Data Analysis
print("\n3. Analyzing expenses...")

total_spending = df["Amount"].sum()
average_expense = df["Amount"].mean()
category_spending = df.groupby("Category")["Amount"].sum().sort_values(ascending=False)

monthly = (
    df.set_index("Date")
      .resample("MS")["Amount"]
      .sum()
      .reset_index()
)
monthly.columns = ["Month_Start", "Monthly_Spending"]
monthly["Month"] = monthly["Month_Start"].dt.strftime("%Y-%m")

print(f"\nTotal spending: Rs. {total_spending:,.2f}")
print(f"Average transaction: Rs. {average_expense:,.2f}")
print("\nSpending by category:")
print(category_spending.to_string())
print("\nMonthly spending:")
print(monthly[["Month", "Monthly_Spending"]].to_string(index=False))

category_spending.to_csv(OUTPUT_DIR / "category_spending.csv")
monthly.to_csv(OUTPUT_DIR / "monthly_spending.csv", index=False)

# 4. Visualizations
print("\n4. Creating graphs...")

sns.set_theme(style="whitegrid")

plt.figure(figsize=(9, 5))
category_spending.sort_values().plot(kind="barh")
plt.title("Total Spending by Category")
plt.xlabel("Amount (Rs.)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "category_spending.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(monthly["Month"], monthly["Monthly_Spending"], marker="o")
plt.title("Monthly Spending Trend")
plt.xlabel("Month")
plt.ylabel("Amount (Rs.)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "monthly_trend.png")
plt.close()

payment_spending = df.groupby("Payment_Mode")["Amount"].sum()
plt.figure(figsize=(7, 5))
payment_spending.plot(kind="pie", autopct="%1.1f%%")
plt.title("Spending by Payment Mode")
plt.ylabel("")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "payment_modes.png")
plt.close()

print("Graphs saved in the outputs folder.")

# 5. Prepare Machine Learning dataset
print("\n5. Preparing Machine Learning model...")

ml_data = monthly[["Monthly_Spending"]].copy()
ml_data["Month_Number"] = np.arange(1, len(ml_data) + 1)
ml_data["Previous_Month"] = ml_data["Monthly_Spending"].shift(1)
ml_data["Previous_2_Months"] = ml_data["Monthly_Spending"].shift(2)
ml_data["Previous_3_Months"] = ml_data["Monthly_Spending"].shift(3)
ml_data["Rolling_Average_3M"] = (
    ml_data["Monthly_Spending"].shift(1).rolling(3).mean()
)

ml_data.dropna(inplace=True)

features = [
    "Month_Number",
    "Previous_Month",
    "Previous_2_Months",
    "Previous_3_Months",
    "Rolling_Average_3M"
]

X = ml_data[features]
y = ml_data["Monthly_Spending"]

# Keep chronological order: earlier months for training
split_index = int(len(ml_data) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# 6. Train and compare models
print("\n6. Training models...")

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42
    )
}

results = []
trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions) if len(y_test) > 1 else np.nan

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2_Score": r2
    })
    trained_models[name] = model

    print(f"\n{name}")
    print(f"MAE: Rs. {mae:,.2f}")
    print(f"RMSE: Rs. {rmse:,.2f}")
    print(f"R2 Score: {r2:.3f}")

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

# Plot actual vs predicted values
best_name = results_df.sort_values("MAE").iloc[0]["Model"]
best_model = trained_models[best_name]
test_predictions = best_model.predict(X_test)

plt.figure(figsize=(9, 5))
plt.plot(y_test.to_numpy(), marker="o", label="Actual Spending")
plt.plot(test_predictions, marker="x", label="Predicted Spending")
plt.title(f"Actual vs Predicted Spending ({best_name})")
plt.xlabel("Test Month")
plt.ylabel("Amount (Rs.)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png")
plt.close()

# 7. Predict the next month
last_row = ml_data.iloc[-1]

next_month_features = pd.DataFrame([{
    "Month_Number": len(monthly) + 1,
    "Previous_Month": monthly["Monthly_Spending"].iloc[-1],
    "Previous_2_Months": monthly["Monthly_Spending"].iloc[-2],
    "Previous_3_Months": monthly["Monthly_Spending"].iloc[-3],
    "Rolling_Average_3M": monthly["Monthly_Spending"].iloc[-3:].mean()
}])[features]

next_prediction = best_model.predict(next_month_features)[0]
next_month_name = (
    monthly["Month_Start"].iloc[-1] + pd.offsets.MonthBegin(1)
).strftime("%B %Y")

print("\n" + "=" * 50)
print(" PROJECT RESULTS ")
print("=" * 50)
print(f"Best model by test MAE: {best_name}")
print(f"Predicted spending for {next_month_name}: Rs. {next_prediction:,.2f}")
print("\nFiles saved in:", OUTPUT_DIR)
print("Project completed successfully!")
print("=" * 50)
