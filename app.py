import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression


st.set_page_config(
    page_title="AI Business Intelligence Copilot",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
/* Arsha-inspired Streamlit theme */
.stApp {
    background: linear-gradient(135deg, #f5f9ff 0%, #eef5ff 45%, #ffffff 100%);
    color: #1f2937;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}

.hero-card {
    background: linear-gradient(135deg, #37517e 0%, #47b2e4 100%);
    padding: 38px 40px;
    border-radius: 24px;
    color: white;
    margin-bottom: 28px;
    box-shadow: 0 18px 45px rgba(55, 81, 126, 0.25);
    border: 1px solid rgba(255,255,255,0.18);
}

.hero-card h1 {
    font-size: 43px;
    line-height: 1.15;
    font-weight: 850;
    margin-bottom: 12px;
    color: white;
}

.hero-card p {
    font-size: 18px;
    line-height: 1.7;
    margin-bottom: 0;
    opacity: 0.95;
    max-width: 980px;
}

.hero-badges {
    margin-top: 20px;
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}

.hero-badge {
    background: rgba(255, 255, 255, 0.16);
    border: 1px solid rgba(255, 255, 255, 0.24);
    padding: 8px 13px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 13px;
    color: #ffffff;
}

h1, h2, h3 {
    color: #37517e;
    font-weight: 800;
}

h4 {
    color: #37517e;
    font-weight: 750;
}

[data-testid="stMetric"] {
    background: #ffffff;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #e6eef8;
    box-shadow: 0 10px 28px rgba(55, 81, 126, 0.10);
}

[data-testid="stMetricLabel"] {
    color: #37517e;
    font-weight: 750;
}

[data-testid="stMetricValue"] {
    color: #111827;
    font-weight: 850;
}

.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(135deg, #47b2e4, #37517e);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.65rem 1.2rem;
    font-weight: 750;
    box-shadow: 0 8px 20px rgba(71, 178, 228, 0.25);
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #37517e, #243b63);
    color: white;
    border: none;
}

[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 2px dashed #47b2e4;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(55, 81, 126, 0.08);
}

.stSelectbox,
.stTextInput {
    background: rgba(255,255,255,0.55);
    border-radius: 14px;
}

[data-testid="stDataFrame"],
[data-testid="stPlotlyChart"] {
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 8px 22px rgba(55, 81, 126, 0.08);
}

[data-testid="stAlert"] {
    border-radius: 14px;
    border: 1px solid #b9dcff;
    background: #e8f4ff !important;
    box-shadow: 0 6px 18px rgba(55, 81, 126, 0.06);
}

[data-testid="stAlert"] *,
[data-testid="stAlert"] p,
[data-testid="stAlert"] div,
[data-testid="stAlert"] span {
    color: #1f2937 !important;
    font-weight: 600;
}

[data-testid="stAlert"] svg {
    color: #37517e !important;
    fill: #37517e !important;
}

.footer-box {
    margin-top: 40px;
    padding: 22px;
    text-align: center;
    background: #ffffff;
    border-radius: 18px;
    border: 1px solid #e6eef8;
    color: #37517e;
    box-shadow: 0 8px 20px rgba(55, 81, 126, 0.08);
}

.footer-box strong {
    color: #37517e;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-card">
    <h1>📊 AI-Powered Business Intelligence Copilot for SMEs</h1>
    <p>
        Upload any sales, retail, finance, order, transaction, or business dataset. Map the columns once, and the app will generate interactive dashboards, machine learning insights, business recommendations, and downloadable reports.
    </p>
    <div class="hero-badges">
        <span class="hero-badge">Dashboard</span>
        <span class="hero-badge">Machine Learning</span>
        <span class="hero-badge">Business Copilot</span>
        <span class="hero-badge">Report Export</span>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# Helper Functions
# -----------------------------

def make_unique_columns(columns):
    clean_cols = []
    seen = {}

    for i, col in enumerate(columns):
        col = str(col).strip()

        if col == "" or col.lower() in ["nan", "none", "unnamed: 0"]:
            col = f"Column_{i+1}"

        if col in seen:
            seen[col] += 1
            col = f"{col}_{seen[col]}"
        else:
            seen[col] = 0

        clean_cols.append(col)

    return clean_cols


def load_data(file):
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    df.columns = make_unique_columns(df.columns)
    return df


def smart_numeric(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.replace("Rs.", "", regex=False)
        .str.replace("Rs", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip(),
        errors="coerce"
    )


def auto_select_column(columns, keywords):
    columns_lower = [str(c).lower() for c in columns]

    for keyword in keywords:
        for i, col in enumerate(columns_lower):
            if keyword.lower() in col:
                return i + 1

    return 0


def clean_selected_data(df, value_col, date_col):
    df = df.copy()

    df[value_col] = smart_numeric(df[value_col])
    df = df.dropna(subset=[value_col])

    if date_col != "None":
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)

    df = df.drop_duplicates()

    return df


def show_kpis(df, value_col, order_col, customer_col):
    total_value = df[value_col].sum()
    total_records = len(df)

    if order_col != "None":
        total_orders = df[order_col].nunique()
    else:
        total_orders = total_records

    if customer_col != "None":
        total_customers = df[customer_col].nunique()
    else:
        total_customers = "N/A"

    avg_value = total_value / total_orders if total_orders > 0 else 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Value", f"{total_value:,.2f}")
    col2.metric("Total Orders / Records", f"{total_orders:,}")
    col3.metric("Total Customers", f"{total_customers}")
    col4.metric("Average Value", f"{avg_value:,.2f}")


def group_chart(df, group_col, value_col, chart_title):
    if group_col == "None":
        st.info(f"Select a column for {chart_title}.")
        return

    chart_df = (
        df.groupby(group_col)[value_col]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .head(20)
    )

    fig = px.bar(
        chart_df,
        x=group_col,
        y=value_col,
        title=chart_title,
        text_auto=True
    )

    st.plotly_chart(fig, use_container_width=True)


def top_items_chart(df, item_col, value_col, title):
    if item_col == "None":
        st.info(f"Select a column for {title}.")
        return

    top_df = (
        df.groupby(item_col)[value_col]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        top_df,
        x=value_col,
        y=item_col,
        orientation="h",
        title=title,
        text_auto=True
    )

    st.plotly_chart(fig, use_container_width=True)


def automatic_insights(df, value_col, date_col, category_col, region_col, product_col, customer_col):
    insights = []

    total_value = df[value_col].sum()
    insights.append(f"Total business value in this dataset is **{total_value:,.2f}**.")

    if region_col != "None":
        region_sales = df.groupby(region_col)[value_col].sum().sort_values(ascending=False)
        insights.append(
            f"Best performing location/region is **{region_sales.index[0]}** with value **{region_sales.iloc[0]:,.2f}**."
        )
        insights.append(
            f"Lowest performing location/region is **{region_sales.index[-1]}** with value **{region_sales.iloc[-1]:,.2f}**."
        )

    if category_col != "None":
        category_sales = df.groupby(category_col)[value_col].sum().sort_values(ascending=False)
        insights.append(
            f"Top category is **{category_sales.index[0]}** with value **{category_sales.iloc[0]:,.2f}**."
        )

    if product_col != "None":
        product_sales = df.groupby(product_col)[value_col].sum().sort_values(ascending=False)
        insights.append(
            f"Top product/item is **{product_sales.index[0]}** with value **{product_sales.iloc[0]:,.2f}**."
        )

    if customer_col != "None":
        customer_sales = df.groupby(customer_col)[value_col].sum().sort_values(ascending=False)
        insights.append(
            f"Highest value customer is **{customer_sales.index[0]}** with value **{customer_sales.iloc[0]:,.2f}**."
        )

    if date_col != "None":
        date_df = df.dropna(subset=[date_col]).copy()

        if len(date_df) > 0:
            monthly = (
                date_df.groupby(pd.Grouper(key=date_col, freq="M"))[value_col]
                .sum()
                .dropna()
            )

            if len(monthly) >= 2:
                if monthly.iloc[-1] > monthly.iloc[-2]:
                    insights.append("Latest monthly trend shows **growth** compared with the previous month.")
                else:
                    insights.append("Latest monthly trend shows **decline** compared with the previous month.")

    return insights


def sales_forecasting(df, value_col, date_col):
    if date_col == "None":
        st.warning("Date column is required for forecasting.")
        return

    forecast_df = df.dropna(subset=[date_col]).copy()

    if len(forecast_df) < 10:
        st.warning("Not enough date-based data for forecasting.")
        return

    monthly_sales = (
        forecast_df.groupby(pd.Grouper(key=date_col, freq="M"))[value_col]
        .sum()
        .reset_index()
        .dropna()
    )

    if len(monthly_sales) < 4:
        st.warning("At least 4 months of data are needed for forecasting.")
        return

    monthly_sales["Month_Number"] = np.arange(len(monthly_sales))

    X = monthly_sales[["Month_Number"]]
    y = monthly_sales[value_col]

    model = LinearRegression()
    model.fit(X, y)

    future_months = 6
    future_numbers = np.arange(
        len(monthly_sales),
        len(monthly_sales) + future_months
    ).reshape(-1, 1)

    predictions = model.predict(future_numbers)

    last_date = monthly_sales[date_col].max()
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=future_months,
        freq="M"
    )

    result = pd.DataFrame({
        "Forecast Date": future_dates,
        "Forecasted Value": predictions
    })

    fig = px.line(
        result,
        x="Forecast Date",
        y="Forecasted Value",
        markers=True,
        title="Next 6 Months Forecast"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(result, use_container_width=True)


def sales_prediction_model(df, value_col, date_col, feature_cols):
    model_df = df.copy()

    if date_col != "None":
        model_df["Year"] = model_df[date_col].dt.year
        model_df["Month"] = model_df[date_col].dt.month
        model_df["DayOfWeek"] = model_df[date_col].dt.dayofweek
        feature_cols = feature_cols + ["Year", "Month", "DayOfWeek"]

    feature_cols = [col for col in feature_cols if col != "None" and col in model_df.columns]

    if len(feature_cols) == 0:
        st.warning("Select at least one feature column for ML prediction.")
        return

    model_data = model_df[feature_cols + [value_col]].dropna()

    if len(model_data) < 50:
        st.warning("At least 50 clean rows are recommended for sales/value prediction model.")
        return

    X = model_data[feature_cols]
    y = model_data[value_col]

    X_encoded = pd.get_dummies(X, drop_first=True)

    if X_encoded.shape[1] == 0:
        st.warning("Not enough usable feature columns after encoding.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:,.2f}")
    col2.metric("RMSE", f"{rmse:,.2f}")
    col3.metric("R² Score", f"{r2:.2f}")

    importance = pd.DataFrame({
        "Feature": X_encoded.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False).head(15)

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Important Features for Prediction"
    )

    st.plotly_chart(fig, use_container_width=True)


def customer_segmentation(df, value_col, customer_col, order_col):
    if customer_col == "None":
        st.warning("Customer column is required for customer segmentation.")
        return

    if order_col != "None":
        customer_df = df.groupby(customer_col).agg(
            Total_Value=(value_col, "sum"),
            Total_Orders=(order_col, "nunique"),
            Average_Value=(value_col, "mean")
        ).reset_index()
    else:
        customer_df = df.groupby(customer_col).agg(
            Total_Value=(value_col, "sum"),
            Total_Orders=(value_col, "count"),
            Average_Value=(value_col, "mean")
        ).reset_index()

    if len(customer_df) < 3:
        st.warning("At least 3 customers are required for clustering.")
        return

    features = customer_df[["Total_Value", "Total_Orders", "Average_Value"]]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    clusters = min(3, len(customer_df))

    kmeans = KMeans(
        n_clusters=clusters,
        random_state=42,
        n_init=10
    )

    customer_df["Cluster"] = kmeans.fit_predict(scaled)

    cluster_summary = customer_df.groupby("Cluster").agg(
        Customers=(customer_col, "count"),
        Avg_Total_Value=("Total_Value", "mean"),
        Avg_Orders=("Total_Orders", "mean"),
        Avg_Value=("Average_Value", "mean")
    ).reset_index()

    st.dataframe(cluster_summary, use_container_width=True)

    fig = px.scatter(
        customer_df,
        x="Total_Orders",
        y="Total_Value",
        color="Cluster",
        size="Average_Value",
        hover_data=[customer_col],
        title="Customer Segmentation using K-Means"
    )

    st.plotly_chart(fig, use_container_width=True)


def anomaly_detection(df, value_col, order_col, customer_col, category_col, product_col, region_col):
    anomaly_df = df.dropna(subset=[value_col]).copy()

    if len(anomaly_df) < 30:
        st.warning("At least 30 rows are recommended for anomaly detection.")
        return

    model = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    anomaly_df["Anomaly"] = model.fit_predict(anomaly_df[[value_col]])

    anomaly_df["Anomaly_Label"] = anomaly_df["Anomaly"].map({
        1: "Normal",
        -1: "Unusual"
    })

    count_df = anomaly_df["Anomaly_Label"].value_counts().reset_index()
    count_df.columns = ["Order Type", "Count"]

    fig = px.bar(
        count_df,
        x="Order Type",
        y="Count",
        title="Normal vs Unusual Records",
        text_auto=True
    )

    st.plotly_chart(fig, use_container_width=True)

    display_cols = []

    for col in [order_col, customer_col, region_col, category_col, product_col, value_col, "Anomaly_Label"]:
        if col != "None" and col in anomaly_df.columns and col not in display_cols:
            display_cols.append(col)

    unusual = anomaly_df[anomaly_df["Anomaly_Label"] == "Unusual"].sort_values(
        by=value_col,
        ascending=False
    )

    st.write("Top unusual records:")
    st.dataframe(unusual[display_cols].head(20), use_container_width=True)


def simple_copilot_answer(question, df, value_col, date_col, category_col, region_col, product_col, customer_col):
    q = question.lower()

    if "summary" in q or "summarize" in q or "overall" in q:
        insights = automatic_insights(df, value_col, date_col, category_col, region_col, product_col, customer_col)
        return "\n\n".join(insights)

    if "top" in q and region_col != "None" and ("region" in q or "city" in q or "location" in q):
        result = df.groupby(region_col)[value_col].sum().sort_values(ascending=False).head(5)
        return result.to_string()

    if "top" in q and category_col != "None" and "category" in q:
        result = df.groupby(category_col)[value_col].sum().sort_values(ascending=False).head(5)
        return result.to_string()

    if "top" in q and product_col != "None" and ("product" in q or "item" in q):
        result = df.groupby(product_col)[value_col].sum().sort_values(ascending=False).head(5)
        return result.to_string()

    if "customer" in q and customer_col != "None":
        result = df.groupby(customer_col)[value_col].sum().sort_values(ascending=False).head(5)
        return result.to_string()

    if "low" in q or "lowest" in q or "weak" in q:
        if region_col != "None":
            result = df.groupby(region_col)[value_col].sum().sort_values(ascending=True).head(5)
            return result.to_string()

    return (
        "I can answer basic questions like: summary, top region, top category, "
        "top product, top customer, or lowest region. For advanced natural language answers, "
        "we can later connect Gemini/OpenAI API."
    )


# -----------------------------
# Main App
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    df_original = load_data(uploaded_file)

    st.subheader("1. Raw Dataset Preview")
    st.dataframe(df_original.head(20), use_container_width=True)

    st.subheader("2. Column Mapping")

    columns = list(df_original.columns)
    options = ["None"] + columns

    value_default = auto_select_column(
        columns,
        ["sales", "amount", "revenue", "total", "value", "price", "unitprice", "unit price"]
    )

    date_default = auto_select_column(
        columns,
        ["order date", "invoice date", "date", "month", "ship date"]
    )

    order_default = auto_select_column(
        columns,
        ["order id", "order", "invoice", "transaction", "bill"]
    )

    customer_default = auto_select_column(
        columns,
        ["customer id", "customer", "client", "buyer", "name"]
    )

    category_default = auto_select_column(
        columns,
        ["category", "segment", "department", "type"]
    )

    region_default = auto_select_column(
        columns,
        ["region", "city", "state", "country", "location", "market"]
    )

    product_default = auto_select_column(
        columns,
        ["product name", "product", "item", "description", "sku"]
    )

    col1, col2 = st.columns(2)

    with col1:
        value_col = st.selectbox(
            "Select Sales / Amount / Revenue / Value column",
            options,
            index=value_default
        )

        date_col = st.selectbox(
            "Select Date column",
            options,
            index=date_default
        )

        order_col = st.selectbox(
            "Select Order ID / Invoice / Transaction column",
            options,
            index=order_default
        )

        customer_col = st.selectbox(
            "Select Customer column",
            options,
            index=customer_default
        )

    with col2:
        category_col = st.selectbox(
            "Select Category / Segment column",
            options,
            index=category_default
        )

        region_col = st.selectbox(
            "Select Region / City / Location column",
            options,
            index=region_default
        )

        product_col = st.selectbox(
            "Select Product / Item column",
            options,
            index=product_default
        )

    if value_col == "None":
        st.error("Please select a Sales / Amount / Revenue / Value column to continue.")
        st.stop()

    df = clean_selected_data(df_original, value_col, date_col)

    if df[value_col].isna().all() or len(df) == 0:
        st.error("Selected value column could not be converted into numbers. Please select another numeric column.")
        st.stop()

    st.subheader("3. Cleaned Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("4. Dataset Quality Check")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows After Cleaning", f"{df.shape[0]:,}")
    col2.metric("Columns", f"{df.shape[1]:,}")
    col3.metric("Duplicate Rows Removed", "Yes")

    missing_values = df.isnull().sum().reset_index()
    missing_values.columns = ["Column", "Missing Values"]
    st.dataframe(missing_values, use_container_width=True)

    st.subheader("5. Key Business KPIs")
    show_kpis(df, value_col, order_col, customer_col)

    st.subheader("6. Business Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        group_chart(df, region_col, value_col, f"{value_col} by {region_col}")

    with col2:
        group_chart(df, category_col, value_col, f"{value_col} by {category_col}")

    col3, col4 = st.columns(2)

    with col3:
        top_items_chart(df, product_col, value_col, f"Top 10 {product_col} by {value_col}")

    with col4:
        if customer_col != "None":
            top_items_chart(df, customer_col, value_col, f"Top 10 Customers by {value_col}")
        else:
            st.info("Select customer column to show customer analysis.")

    st.subheader("7. Time Series Trend")

    if date_col != "None":
        time_df = df.dropna(subset=[date_col]).copy()

        monthly = (
            time_df.groupby(pd.Grouper(key=date_col, freq="M"))[value_col]
            .sum()
            .reset_index()
        )

        fig = px.line(
            monthly,
            x=date_col,
            y=value_col,
            markers=True,
            title=f"Monthly Trend of {value_col}"
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select date column to show time series trend.")

    st.subheader("8. Automatic Business Insights")

    insights = automatic_insights(
        df,
        value_col,
        date_col,
        category_col,
        region_col,
        product_col,
        customer_col
    )

    for insight in insights:
        st.info(insight)

    st.subheader("9. Forecasting Model")
    sales_forecasting(df, value_col, date_col)

    st.subheader("10. Machine Learning: Value Prediction Model")

    feature_columns = [
        col for col in [category_col, region_col, product_col, customer_col, order_col]
        if col != "None"
    ]

    sales_prediction_model(df, value_col, date_col, feature_columns)

    st.subheader("11. Machine Learning: Customer Segmentation")
    customer_segmentation(df, value_col, customer_col, order_col)

    st.subheader("12. Machine Learning: Anomaly Detection")
    anomaly_detection(df, value_col, order_col, customer_col, category_col, product_col, region_col)

    def valid_column(df, col):
        return col != "None" and col in df.columns


    def get_top_table(df, group_col, sales_col, n=10, ascending=False):
        result = (
            df.groupby(group_col)[sales_col]
            .sum()
            .sort_values(ascending=ascending)
            .head(n)
            .reset_index()
        )
        return result


    def business_copilot_answer(
        question,
        df,
        sales_col,
        date_col,
        region_col,
        category_col,
        product_col,
        customer_col,
        order_col
    ):
        q = question.lower().strip()

        if q == "":
            return "Please ask a business question.", None

        total_sales = df[sales_col].sum()
        total_records = len(df)

        if valid_column(df, order_col):
            total_orders = df[order_col].nunique()
        else:
            total_orders = total_records

        answer = []
        table = None

        if "summary" in q or "overview" in q or "summarize" in q:
            answer.append("### Business Summary")
            answer.append(f"- Total sales/value is **{total_sales:,.2f}**.")
            answer.append(f"- Total records are **{total_records:,}**.")
            answer.append(f"- Total orders/transactions are **{total_orders:,}**.")

            if valid_column(df, date_col):
                min_date = df[date_col].min()
                max_date = df[date_col].max()
                if pd.notna(min_date) and pd.notna(max_date):
                    answer.append(f"- Data period is from **{min_date.date()}** to **{max_date.date()}**.")

            if valid_column(df, region_col):
                top_region = df.groupby(region_col)[sales_col].sum().sort_values(ascending=False)
                answer.append(f"- Top {region_col} is **{top_region.index[0]}** with **{top_region.iloc[0]:,.2f}**.")

            if valid_column(df, category_col):
                top_category = df.groupby(category_col)[sales_col].sum().sort_values(ascending=False)
                answer.append(f"- Top {category_col} is **{top_category.index[0]}** with **{top_category.iloc[0]:,.2f}**.")

            if valid_column(df, product_col):
                top_product = df.groupby(product_col)[sales_col].sum().sort_values(ascending=False)
                answer.append(f"- Top product/item is **{top_product.index[0]}** with **{top_product.iloc[0]:,.2f}**.")

            if valid_column(df, customer_col):
                top_customer = df.groupby(customer_col)[sales_col].sum().sort_values(ascending=False)
                answer.append(f"- Top customer is **{top_customer.index[0]}** with **{top_customer.iloc[0]:,.2f}**.")

            return "\n".join(answer), table

        elif "top region" in q or "best region" in q or ("highest" in q and "region" in q):
            if valid_column(df, region_col):
                table = get_top_table(df, region_col, sales_col, n=10)
                top_row = table.iloc[0]
                answer_text = f"Top {region_col} is **{top_row[region_col]}** with total value **{top_row[sales_col]:,.2f}**."
                return answer_text, table
            return "Region/location column is not selected.", None

        elif "lowest region" in q or "low region" in q or ("worst" in q and "region" in q):
            if valid_column(df, region_col):
                table = get_top_table(df, region_col, sales_col, n=10, ascending=True)
                low_row = table.iloc[0]
                answer_text = f"Lowest {region_col} is **{low_row[region_col]}** with total value **{low_row[sales_col]:,.2f}**."
                return answer_text, table
            return "Region/location column is not selected.", None

        elif "top category" in q or "best category" in q or ("highest" in q and "category" in q):
            if valid_column(df, category_col):
                table = get_top_table(df, category_col, sales_col, n=10)
                top_row = table.iloc[0]
                answer_text = f"Top {category_col} is **{top_row[category_col]}** with total value **{top_row[sales_col]:,.2f}**."
                return answer_text, table
            return "Category column is not selected.", None

        elif "lowest category" in q or "low category" in q or ("worst" in q and "category" in q):
            if valid_column(df, category_col):
                table = get_top_table(df, category_col, sales_col, n=10, ascending=True)
                low_row = table.iloc[0]
                answer_text = f"Lowest {category_col} is **{low_row[category_col]}** with total value **{low_row[sales_col]:,.2f}**."
                return answer_text, table
            return "Category column is not selected.", None

        elif "top product" in q or "best product" in q or "top item" in q:
            if valid_column(df, product_col):
                table = get_top_table(df, product_col, sales_col, n=10)
                top_row = table.iloc[0]
                answer_text = f"Top product/item is **{top_row[product_col]}** with total value **{top_row[sales_col]:,.2f}**."
                return answer_text, table
            return "Product/item column is not selected.", None

        elif "top customer" in q or "best customer" in q or "highest customer" in q:
            if valid_column(df, customer_col):
                table = get_top_table(df, customer_col, sales_col, n=10)
                top_row = table.iloc[0]
                answer_text = f"Top customer is **{top_row[customer_col]}** with total value **{top_row[sales_col]:,.2f}**."
                return answer_text, table
            return "Customer column is not selected.", None

        elif "month" in q or "monthly" in q or "trend" in q:
            if valid_column(df, date_col):
                monthly = (
                    df.dropna(subset=[date_col])
                    .groupby(pd.Grouper(key=date_col, freq="M"))[sales_col]
                    .sum()
                    .reset_index()
                )

                if monthly.empty:
                    return "No valid monthly data found.", None

                table = monthly.sort_values(by=sales_col, ascending=False)
                best_month = table.iloc[0]
                answer_text = f"Highest monthly sales/value was in **{best_month[date_col].strftime('%B %Y')}** with **{best_month[sales_col]:,.2f}**."
                return answer_text, table

            return "Date column is not selected.", None

        elif "recommend" in q or "suggest" in q or "improve" in q or "advice" in q:
            answer.append("### Business Recommendations")

            if valid_column(df, region_col):
                region_sales = df.groupby(region_col)[sales_col].sum().sort_values()
                answer.append(f"- Focus on improving **{region_col}: {region_sales.index[0]}**, because it has the lowest value of **{region_sales.iloc[0]:,.2f}**.")

            if valid_column(df, category_col):
                category_sales = df.groupby(category_col)[sales_col].sum().sort_values(ascending=False)
                answer.append(f"- Increase marketing around top-performing **{category_col}: {category_sales.index[0]}**.")

            if valid_column(df, product_col):
                product_sales = df.groupby(product_col)[sales_col].sum().sort_values(ascending=False)
                answer.append(f"- Keep enough stock for top product/item: **{product_sales.index[0]}**.")

            if valid_column(df, customer_col):
                customer_sales = df.groupby(customer_col)[sales_col].sum().sort_values(ascending=False)
                answer.append(f"- Give loyalty offers to high-value customers such as **{customer_sales.index[0]}**.")

            answer.append("- Review low-performing areas and compare them with best-performing areas to find gaps in pricing, marketing, stock, or delivery.")

            return "\n".join(answer), None

        else:
            return (
                "I can answer questions like: **summary**, **top region**, **lowest region**, "
                "**top category**, **lowest category**, **top product**, **top customer**, "
                "**monthly trend**, and **recommendations**."
            ), None


    st.subheader("13. Ask Business Copilot")

    st.write(
        "Ask a question, for example: summary, top region, top category, top product, "
        "top customer, lowest region, monthly trend, recommendations."
    )

    question = st.text_input("Ask your business question")

    if st.button("Ask Copilot"):
        answer, result_table = business_copilot_answer(
            question,
            df,
            value_col,
            date_col,
            region_col,
            category_col,
            product_col,
            customer_col,
            order_col
        )

        st.markdown(answer)

        if result_table is not None:
            st.dataframe(result_table, use_container_width=True)

    st.subheader("14. Download Business Report")

    def safe_col(col):
        return col != "None" and col in df.columns

    def make_business_report():
        report = []

        report.append("AI-Powered Business Intelligence Copilot for SMEs")
        report.append("=" * 55)
        report.append("")
        report.append("BUSINESS REPORT")
        report.append("-" * 20)
        report.append("")

        total_value = df[value_col].sum()
        total_records = len(df)

        if safe_col(order_col):
            total_orders = df[order_col].nunique()
        else:
            total_orders = total_records

        report.append(f"Total Sales / Value: {total_value:,.2f}")
        report.append(f"Total Records: {total_records:,}")
        report.append(f"Total Orders / Transactions: {total_orders:,}")

        if total_orders > 0:
            report.append(f"Average Value per Order: {total_value / total_orders:,.2f}")

        report.append("")

        if safe_col(date_col):
            min_date = df[date_col].min()
            max_date = df[date_col].max()

            if pd.notna(min_date) and pd.notna(max_date):
                report.append(f"Data Period: {min_date.date()} to {max_date.date()}")
                report.append("")

        if safe_col(region_col):
            region_sales = df.groupby(region_col)[value_col].sum().sort_values(ascending=False)
            report.append("REGION / LOCATION ANALYSIS")
            report.append("-" * 30)
            report.append(f"Top {region_col}: {region_sales.index[0]} ({region_sales.iloc[0]:,.2f})")
            report.append(f"Lowest {region_col}: {region_sales.index[-1]} ({region_sales.iloc[-1]:,.2f})")
            report.append("")

        if safe_col(category_col):
            category_sales = df.groupby(category_col)[value_col].sum().sort_values(ascending=False)
            report.append("CATEGORY ANALYSIS")
            report.append("-" * 30)
            report.append(f"Top {category_col}: {category_sales.index[0]} ({category_sales.iloc[0]:,.2f})")
            report.append(f"Lowest {category_col}: {category_sales.index[-1]} ({category_sales.iloc[-1]:,.2f})")
            report.append("")

        if safe_col(product_col):
            product_sales = df.groupby(product_col)[value_col].sum().sort_values(ascending=False)
            report.append("PRODUCT / ITEM ANALYSIS")
            report.append("-" * 30)
            report.append(f"Top Product / Item: {product_sales.index[0]} ({product_sales.iloc[0]:,.2f})")
            report.append("")

        if safe_col(customer_col):
            customer_sales = df.groupby(customer_col)[value_col].sum().sort_values(ascending=False)
            report.append("CUSTOMER ANALYSIS")
            report.append("-" * 30)
            report.append(f"Top Customer: {customer_sales.index[0]} ({customer_sales.iloc[0]:,.2f})")
            report.append("")

        report.append("BUSINESS RECOMMENDATIONS")
        report.append("-" * 30)

        if safe_col(region_col):
            region_sales = df.groupby(region_col)[value_col].sum().sort_values()
            report.append(
                f"1. Focus on improving {region_col}: {region_sales.index[0]}, "
                f"because it has the lowest value of {region_sales.iloc[0]:,.2f}."
            )

        if safe_col(category_col):
            category_sales = df.groupby(category_col)[value_col].sum().sort_values(ascending=False)
            report.append(
                f"2. Increase marketing and stock planning for top-performing "
                f"{category_col}: {category_sales.index[0]}."
            )

        if safe_col(product_col):
            product_sales = df.groupby(product_col)[value_col].sum().sort_values(ascending=False)
            report.append(
                f"3. Keep enough inventory for high-performing product/item: {product_sales.index[0]}."
            )

        if safe_col(customer_col):
            customer_sales = df.groupby(customer_col)[value_col].sum().sort_values(ascending=False)
            report.append(
                f"4. Offer loyalty benefits to high-value customers such as {customer_sales.index[0]}."
            )

        report.append(
            "5. Compare low-performing and high-performing areas to identify gaps in pricing, "
            "marketing, delivery, product demand, or customer targeting."
        )

        report.append("")
        report.append("Generated by AI-Powered Business Intelligence Copilot for SMEs.")

        return "\n".join(report)

    business_report = make_business_report()

    st.download_button(
        label="Download Business Report",
        data=business_report,
        file_name="business_report.txt",
        mime="text/plain"
    )

    st.subheader("15. Download Cleaned Data")

    cleaned_csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Cleaned CSV",
        data=cleaned_csv,
        file_name="cleaned_sales_data.csv",
        mime="text/csv"
    )

    st.markdown("""
    <div class="footer-box">
        <strong>AI-Powered Business Intelligence Copilot for SMEs</strong><br>
        Built with Python, Streamlit, Plotly and Scikit-learn.<br>
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("Upload any sales/business CSV or Excel file to start.")