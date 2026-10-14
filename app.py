import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="SmartSpend", page_icon="💸", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "expenses.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

st.markdown("<h1 style='text-align:center;'>💸 SmartSpend</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Personal Expense Analysis & Spending Prediction</p>", unsafe_allow_html=True)
st.divider()

@st.cache_data
def load_data(file_path, modified_time):
    df = pd.read_csv(file_path)
    # Accept common column-name variations without changing the original file.
    rename = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "_")
        if key in {"date", "expense_date", "transaction_date"}:
            rename[col] = "Date"
        elif key in {"category", "expense_category"}:
            rename[col] = "Category"
        elif key in {"amount", "expense_amount", "cost"}:
            rename[col] = "Amount"
        elif key in {"payment_mode", "payment_method", "mode_of_payment"}:
            rename[col] = "Payment_Mode"
    df = df.rename(columns=rename)
    required = {"Date", "Category", "Amount", "Payment_Mode"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df["Category"] = df["Category"].fillna("Other").astype(str)
    df["Payment_Mode"] = df["Payment_Mode"].fillna("Unknown").astype(str)
    df = df.dropna(subset=["Date", "Amount"])
    return df

if not DATA_FILE.exists():
    st.error("I could not find expenses.csv. Place this app.py in the same folder as expenses.csv.")
    st.stop()

try:
    df = load_data(str(DATA_FILE), DATA_FILE.stat().st_mtime)
except Exception as error:
    st.error(f"Could not read expenses.csv: {error}")
    st.info("Check that the CSV has Date, Category, Amount, and Payment_Mode columns.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filter Expenses")
all_categories = sorted(df["Category"].unique().tolist())
selected_categories = st.sidebar.multiselect("Categories", all_categories, default=all_categories)
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

filtered = df[df["Category"].isin(selected_categories)].copy()
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered = filtered[(filtered["Date"] >= start_date) & (filtered["Date"] <= end_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))]

# Metric cards
col1, col2, col3, col4 = st.columns(4)
total = filtered["Amount"].sum() if not filtered.empty else 0
average = filtered["Amount"].mean() if not filtered.empty else 0
transactions = len(filtered)
if not filtered.empty:
    highest_category = filtered.groupby("Category")["Amount"].sum().idxmax()
else:
    highest_category = "—"
col1.metric("Total Spending", f"₹{total:,.2f}")
col2.metric("Average Transaction", f"₹{average:,.2f}")
col3.metric("Transactions", f"{transactions:,}")
col4.metric("Top Category", highest_category)

st.subheader("Spending Analysis")
left, right = st.columns(2)
with left:
    st.markdown("**Category-wise Spending**")
    if filtered.empty:
        st.info("No expenses match these filters.")
    else:
        category_totals = filtered.groupby("Category")["Amount"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots()
        category_totals.plot(kind="bar", ax=ax)
        ax.set_xlabel("Category")
        ax.set_ylabel("Amount (₹)")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
with right:
    st.markdown("**Monthly Spending Trend**")
    if filtered.empty:
        st.info("No expenses match these filters.")
    else:
        monthly = filtered.set_index("Date").resample("MS")["Amount"].sum()
        fig, ax = plt.subplots()
        monthly.plot(kind="line", marker="o", ax=ax)
        ax.set_xlabel("Month")
        ax.set_ylabel("Amount (₹)")
        fig.autofmt_xdate()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

left2, right2 = st.columns(2)
with left2:
    st.markdown("**Payment Mode Breakdown**")
    if not filtered.empty:
        modes = filtered.groupby("Payment_Mode")["Amount"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots()
        modes.plot(kind="bar", ax=ax)
        ax.set_xlabel("Payment Mode")
        ax.set_ylabel("Amount (₹)")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
with right2:
    st.markdown("**Model Evaluation**")
    comparison_file = OUTPUT_DIR / "model_comparison.csv"
    if comparison_file.exists():
        try:
            model_results = pd.read_csv(comparison_file)
            st.dataframe(model_results, use_container_width=True, hide_index=True)
            st.caption("These are the saved evaluation results from the existing project.")
        except Exception:
            st.warning("The model comparison file exists but could not be displayed.")
    else:
        st.info("Model comparison results will appear here when outputs/model_comparison.csv is available.")
    st.caption("Note: prediction quality depends on the model evaluation results; this dashboard does not claim that forecasts are accurate.")

st.subheader("Add a New Expense")
with st.form("add_expense_form", clear_on_submit=True):
    f1, f2 = st.columns(2)
    with f1:
        expense_date = st.date_input("Date", value=pd.Timestamp.today().date())
        category = st.selectbox("Category", options=all_categories + (["Other"] if "Other" not in all_categories else []))
    with f2:
        amount = st.number_input("Amount (₹)", min_value=0.01, value=100.00, step=10.00)
        payment_mode = st.selectbox("Payment Mode", sorted(df["Payment_Mode"].unique().tolist()) + (["Other"] if "Other" not in df["Payment_Mode"].unique() else []))
    submitted = st.form_submit_button("Save Expense", type="primary")
    if submitted:
        new_row = pd.DataFrame([{
            "Date": pd.to_datetime(expense_date).strftime("%Y-%m-%d"),
            "Category": category,
            "Amount": float(amount),
            "Payment_Mode": payment_mode,
        }])
        # Write using the app's standard column names. Preserve any existing extra columns with blanks.
        original = pd.read_csv(DATA_FILE)
        original_cols = list(original.columns)
        existing_lower = {c.strip().lower().replace(" ", "_"): c for c in original_cols}
        output_row = {}
        for col in original_cols:
            key = col.strip().lower().replace(" ", "_")
            if key in {"date", "expense_date", "transaction_date"}:
                output_row[col] = new_row.loc[0, "Date"]
            elif key in {"category", "expense_category"}:
                output_row[col] = category
            elif key in {"amount", "expense_amount", "cost"}:
                output_row[col] = float(amount)
            elif key in {"payment_mode", "payment_method", "mode_of_payment"}:
                output_row[col] = payment_mode
            else:
                output_row[col] = ""
        updated = pd.concat([original, pd.DataFrame([output_row], columns=original_cols)], ignore_index=True)
        updated.to_csv(DATA_FILE, index=False)
        st.cache_data.clear()
        st.success("Expense saved to expenses.csv. Refresh the page to update the dashboard.")

st.subheader("Recent Expenses")
if filtered.empty:
    st.info("No expenses match these filters.")
else:
    recent = filtered.sort_values("Date", ascending=False).head(20).copy()
    recent["Date"] = pd.to_datetime(recent["Date"], errors="coerce")
    recent["Date"] = recent["Date"].dt.strftime("%Y-%m-%d")
    recent["Amount"] = recent["Amount"].map(lambda value: f"₹{value:,.2f}")
    st.dataframe(recent[["Date", "Category", "Amount", "Payment_Mode"]], use_container_width=True, hide_index=True)

st.caption("SmartSpend is an academic demo using sample expense data. Review the model metrics before relying on any prediction.")
