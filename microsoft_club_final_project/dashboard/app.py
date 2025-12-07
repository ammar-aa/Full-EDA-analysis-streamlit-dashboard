import warnings 
warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Car EDA Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("clean_data.csv")

df = load_data()



st.title("Car Data EDA Dashboard")
st.subheader("Dataset Preview")
st.dataframe(df.head())

col1, col2, col3 = st.columns(3)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing Values", df.isnull().sum().sum())


st.header("Data Overview")
st.subheader("Statistical Summary")
st.write(df.describe().T)
st.subheader("Column Types")
st.write(df.dtypes)

numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()


st.header("Filtering")

selected_col = st.selectbox("Select column to filter", df.columns)

if selected_col in numeric_cols:
    min_val = float(df[selected_col].min())
    max_val = float(df[selected_col].max())

    min_input, max_input = st.slider(
        "Value range",
        min_val, max_val, (min_val, max_val)
    )

    df_filtered = df[(df[selected_col] >= min_input) & (df[selected_col] <= max_input)]

else:
    unique_values = df[selected_col].unique().tolist()
    selected_values = st.multiselect(
        "Choose values", unique_values, default=unique_values[:5]
    )
    df_filtered = df[df[selected_col].isin(selected_values)]

st.write("Filtered Data")
st.dataframe(df_filtered.head())


st.header("Visualizations")

plot_type = st.selectbox(
    "Select Plot Type",
    ["Histogram", "Boxplot", "Scatter Plot", "Correlation Heatmap", "Brand Price Distribution"]
)

if plot_type == "Histogram":
    col = st.selectbox("Select numeric column", numeric_cols)
    fig, ax = plt.subplots(figsize=(9,4))
    sns.histplot(df[col], bins=30, ax=ax)
    ax.set_title(f"Histogram of {col}")
    st.pyplot(fig)

elif plot_type == "Boxplot":
    num_col = st.selectbox("Numeric column", numeric_cols)
    cat_col = st.selectbox("Categorical column", cat_cols)
    fig, ax = plt.subplots(figsize=(9,4))
    sns.boxplot(x=df[cat_col], y=df[num_col], ax=ax)
    plt.xticks(rotation=90)
    ax.set_title(f"{num_col} Distribution by {cat_col}")
    plt.xticks(rotation=0)
    st.pyplot(fig)

elif plot_type == "Scatter Plot":
    x_col = st.selectbox("X-axis", numeric_cols)
    y_col = st.selectbox("Y-axis", numeric_cols)
    fig, ax = plt.subplots(figsize=(9,4))
    sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
    plt.xticks(rotation=0)
    ax.set_title(f"{y_col} vs. {x_col}")
    st.pyplot(fig)

elif plot_type == "Correlation Heatmap":
    fig, ax = plt.subplots(figsize=(9,4))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    plt.xticks(rotation=0)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

elif plot_type == "Brand Price Distribution":
    fig, ax = plt.subplots(figsize=(9, 4))
    brand_avg = df.groupby("car_name")["selling_price"].mean().sort_values(ascending=False).head(20)
    sns.barplot(x=brand_avg.index, y=brand_avg.values, ax=ax)
    plt.xticks(rotation=75)
    ax.set_ylabel("Average Selling Price")
    ax.set_title("Top 20 Car Brands by Average Price")
    st.pyplot(fig)
