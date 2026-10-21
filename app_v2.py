import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

# ======================================================
# PAGE CONFIGURATION
# ======================================================

st.set_page_config(
    page_title="Market Basket Analysis System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# DATA FUNCTIONS
# ======================================================

DATASET_PATH = Path("data/raw/online+retail/Online Retail.xlsx")
CACHE_PATH = Path("data/processed/transactions_clean.csv")


# Full cleaned data (Dataset, Dashboard, Visualization pages)
@st.cache_data(show_spinner="Loading dataset (first time only)...")
def load_clean_data():
    df = pd.read_excel(DATASET_PATH)
    df = df.dropna(subset=["Description"])
    df["Description"] = df["Description"].astype(str).str.strip()
    df = df[df["Quantity"] > 0]
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    return df


# Light data for Analysis and Export: 3 columns, saved to a small CSV
# so the big Excel file is read only ONE time ever.
@st.cache_data(show_spinner="Preparing analysis data (first time only)...")
def load_analysis_data():
    if CACHE_PATH.exists():
        cached = pd.read_csv(CACHE_PATH)
        required = {"InvoiceNo", "Description", "Country"}
        if required.issubset(cached.columns):
            cached["InvoiceNo"] = cached["InvoiceNo"].astype(str)
            cached["Description"] = cached["Description"].astype(str).str.strip()
            cached = cached.dropna(subset=["InvoiceNo", "Description"])
            return cached[["InvoiceNo", "Description", "Country"]].drop_duplicates()
        # Rebuild an old/incomplete cache from the original Excel dataset.
        CACHE_PATH.unlink()

    df = pd.read_excel(
        DATASET_PATH,
        usecols=["InvoiceNo", "Description", "Quantity", "Country"]
    )
    df = df.dropna(subset=["Description"])
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df = df[~df["InvoiceNo"].str.startswith("C")]
    df = df[df["Quantity"] > 0]
    df["Description"] = df["Description"].astype(str).str.strip()
    df = df[["InvoiceNo", "Description", "Country"]].drop_duplicates()

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    return df


def apriori_pairs(df, top_n, country, min_support, min_confidence):
    """Level-wise Apriori for itemsets of size 1 and 2 (memory safe).

    Step 1: count single items, keep those with support >= min_support.
    Step 2: build candidate pairs ONLY from frequent items (Apriori pruning)
            and count them with one matrix product.
    Step 3: generate rules and compute support, confidence and lift.
    Support is the share of ALL invoices (in the selected country).
    """
    if country != "All":
        df = df[df["Country"] == country]

    n_total = df["InvoiceNo"].nunique()
    if n_total == 0 or df.empty:
        return pd.DataFrame(), pd.DataFrame(), 0, 0

    top_items = (
        df.groupby("Description")["InvoiceNo"]
        .nunique()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    d = df[df["Description"].isin(top_items)][["InvoiceNo", "Description"]]

    inv_codes, inv_uniques = pd.factorize(d["InvoiceNo"])
    item_codes, item_uniques = pd.factorize(d["Description"])
    names = np.array(item_uniques)

    B = np.zeros((len(inv_uniques), len(names)), dtype=np.float32)
    B[inv_codes, item_codes] = 1.0

    # Level 1
    item_support = B.sum(axis=0) / n_total
    keep = np.where(item_support >= min_support)[0]
    if len(keep) == 0:
        return pd.DataFrame(), pd.DataFrame(), n_total, len(names)

    # Level 2
    Bk = B[:, keep]
    sup1 = item_support[keep]
    k_names = names[keep]
    co = (Bk.T @ Bk) / n_total

    iu, ju = np.triu_indices(len(keep), k=1)
    pair_sup = co[iu, ju]
    ok = pair_sup >= min_support
    iu, ju, pair_sup = iu[ok], ju[ok], pair_sup[ok]

    rows = [{"support": float(s), "itemsets": k_names[i]}
            for i, s in enumerate(sup1)]
    rows += [{"support": float(s), "itemsets": f"{k_names[i]}, {k_names[j]}"}
             for i, j, s in zip(iu, ju, pair_sup)]
    itemsets = (pd.DataFrame(rows)
                .sort_values("support", ascending=False)
                .reset_index(drop=True))

    # Rules (both directions of every frequent pair)
    ant = np.concatenate([iu, ju])
    con = np.concatenate([ju, iu])
    sup = np.concatenate([pair_sup, pair_sup])
    conf = sup / sup1[ant]
    lift = conf / sup1[con]
    good = conf >= min_confidence

    rules = pd.DataFrame({
        "antecedents": k_names[ant[good]],
        "consequents": k_names[con[good]],
        "support": sup[good],
        "confidence": conf[good],
        "lift": lift[good],
    }).sort_values("lift", ascending=False).reset_index(drop=True)

    return itemsets, rules, n_total, len(names)


# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("🛒 Market Basket Analysis")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📂 Dataset",
        "📊 Dashboard",
        "📈 Analysis",
        "📉 Visualization",
        "📥 Export",
        "ℹ️ About"
    ]
)

# ======================================================
# HOME PAGE
# ======================================================

if page == "🏠 Home":

    st.title("🛒 Market Basket Analysis System")

    st.markdown("---")

    st.header("Welcome")

    st.write("""
This software performs Market Basket Analysis using the Apriori Algorithm.

### Features

- Upload transaction dataset
- Preview dataset
- Clean data
- Prepare transactions
- Create basket matrix
- Discover frequent itemsets
- Generate association rules
- Calculate Support
- Calculate Confidence
- Calculate Lift
- Visualize results
- Export analysis
""")

    st.info("👈 Use the navigation menu on the left to begin.")

# ======================================================
# DATASET PAGE
# ======================================================

elif page == "📂 Dataset":

    st.title("📂 Dataset Management")

    st.markdown("---")

    if DATASET_PATH.exists():

        st.success("✅ Dataset found successfully!")

        df = load_clean_data()

        st.subheader("Dataset Preview (after cleaning)")
        st.dataframe(df.head(), use_container_width=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Rows", f"{df.shape[0]:,}")

        with col2:
            st.metric("Columns", df.shape[1])

        with col3:
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")

        st.subheader("Column Names")
        st.write(df.columns.tolist())

    else:

        st.error("❌ Dataset not found!")

        st.write("Expected location:")

        st.code("data/raw/online+retail/Online Retail.xlsx")

# ======================================================
# DASHBOARD PAGE
# ======================================================

elif page == "📊 Dashboard":

    st.title("📊 Dashboard")

    st.markdown("---")

    if DATASET_PATH.exists():

        df = load_clean_data()

        total_transactions = df["InvoiceNo"].nunique()
        total_products = df["Description"].nunique()
        total_records = len(df)
        total_customers = df["CustomerID"].nunique()

        st.subheader("📈 Key Performance Indicators")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(label="📄 Transactions", value=f"{total_transactions:,}")

        with col2:
            st.metric(label="📦 Products", value=f"{total_products:,}")

        with col3:
            st.metric(label="🛍️ Records", value=f"{total_records:,}")

        with col4:
            st.metric(label="👥 Customers", value=f"{total_customers:,}")

        st.markdown("---")

        st.subheader("Dataset Summary")

        st.write(df.describe())

    else:

        st.error("Dataset not found!")

# ======================================================
# ANALYSIS PAGE
# ======================================================

elif page == "📈 Analysis":

    st.title("📈 Market Basket Analysis")
    st.markdown("---")

    if not DATASET_PATH.exists():
        st.error("Dataset not found!")

    else:
        data = load_analysis_data()

        st.subheader("Analysis Settings")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            min_support = st.slider(
                "Minimum Support",
                min_value=0.005,
                max_value=0.10,
                value=0.005,
                step=0.005,
                format="%.3f"
            )
        with col2:
            min_confidence = st.slider(
                "Minimum Confidence",
                min_value=0.10,
                max_value=1.00,
                value=0.20,
                step=0.05,
                format="%.2f"
            )
        with col3:
            top_n = st.slider(
                "Top N products",
                min_value=20,
                max_value=150,
                value=50,
                step=10
            )
        with col4:
            countries = ["All"] + sorted(data["Country"].unique().tolist())
            country = st.selectbox("Country", countries)

        st.caption(
            "Apriori is applied to the top N most frequent products, "
            "with itemsets of up to 2 products."
        )

        if st.button("▶ Run Analysis", type="primary"):

            with st.spinner("Running Apriori..."):
                frequent_itemsets, rules, n_total, n_items = apriori_pairs(
                    data, top_n, country, min_support, min_confidence
                )

            st.info(f"Transactions analysed: {n_total:,} | "
                    f"Products considered: {n_items:,}")

            if frequent_itemsets.empty:
                st.warning("No frequent itemsets found. Lower the minimum support.")

            else:
                st.success(f"✅ {len(frequent_itemsets):,} frequent itemsets generated.")
                st.subheader("Frequent Itemsets")
                st.dataframe(frequent_itemsets.head(20), use_container_width=True)

                st.download_button(
                    "📥 Download Frequent Itemsets",
                    frequent_itemsets.to_csv(index=False).encode("utf-8"),
                    "frequent_itemsets.csv", "text/csv"
                )

                pair_count = int(
                    frequent_itemsets["itemsets"]
                    .astype(str)
                    .str.contains(", ", regex=False)
                    .sum()
                )
                st.info(f"Frequent product pairs found: {pair_count:,}")

                if rules.empty:
                    st.warning(
                        "Frequent itemsets were found, but no association rules "
                        "meet the current thresholds. Try Minimum Support = 0.005 "
                        "and Minimum Confidence = 0.10."
                    )
                else:
                    st.success(f"✅ {len(rules):,} association rules generated.")
                    st.subheader("Association Rules")
                    st.dataframe(rules.head(20), use_container_width=True)

                    st.download_button(
                        "📥 Download Association Rules",
                        rules.to_csv(index=False).encode("utf-8"),
                        "association_rules.csv", "text/csv"
                    )

# ======================================================
# VISUALIZATION PAGE
# ======================================================

elif page == "📉 Visualization":

    st.title("📉 Data Visualization")

    st.markdown("---")

    if DATASET_PATH.exists():

        df = load_clean_data()

        st.subheader("Top 10 Best Selling Products")

        top_products = (
            df.groupby("Description")["Quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig = px.bar(
            top_products,
            x="Quantity",
            y="Description",
            orientation="h",
            title="Top 10 Best Selling Products"
        )

        fig.update_layout(yaxis={"categoryorder": "total ascending"})

        st.plotly_chart(fig, use_container_width=True)

    else:

        st.error("Dataset not found!")

# ======================================================
# EXPORT PAGE
# ======================================================

elif page == "📥 Export":

    st.title("📥 Export Results")

    st.markdown("---")

    st.write(
        "Use this page to download the results generated by the "
        "Market Basket Analysis system."
    )

    if DATASET_PATH.exists():

        data = load_analysis_data()

        st.subheader("Analysis Settings")

        col1, col2, col3 = st.columns(3)

        with col1:
            min_support = st.slider(
                "Minimum Support",
                min_value=0.005,
                max_value=0.10,
                value=0.005,
                step=0.005,
                format="%.3f",
                key="exp_support"
            )

        with col2:
            min_confidence = st.slider(
                "Minimum Confidence",
                min_value=0.10,
                max_value=1.00,
                value=0.20,
                step=0.05,
                format="%.2f",
                key="exp_conf"
            )

        with col3:
            top_n = st.slider(
                "Top N products",
                min_value=20,
                max_value=150,
                value=50,
                step=10,
                key="exp_topn"
            )

        if st.button("🔄 Generate Export Files"):

            with st.spinner("Generating export files..."):
                frequent_itemsets, rules, n_total, n_items = apriori_pairs(
                    data, top_n, "All", min_support, min_confidence
                )

            if frequent_itemsets.empty:

                st.warning("No frequent itemsets found. Lower the minimum support.")

            else:

                st.success("✅ Export files generated successfully!")

                st.download_button(
                    "📥 Download Frequent Itemsets",
                    frequent_itemsets.to_csv(index=False).encode("utf-8"),
                    "frequent_itemsets.csv",
                    "text/csv"
                )

                pair_count = int(
                    frequent_itemsets["itemsets"]
                    .astype(str)
                    .str.contains(", ", regex=False)
                    .sum()
                )
                st.info(f"Frequent product pairs found: {pair_count:,}")

                if rules.empty:
                    st.warning(
                        "Frequent itemsets were found, but no association rules "
                        "meet the current thresholds. Try Minimum Support = 0.005 "
                        "and Minimum Confidence = 0.10."
                    )
                else:
                    st.download_button(
                        "📥 Download Association Rules",
                        rules.to_csv(index=False).encode("utf-8"),
                        "association_rules.csv",
                        "text/csv"
                    )

    else:

        st.error("❌ Dataset not found!")

# ======================================================
# ABOUT PAGE
# ======================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About the System")

    st.markdown("---")

    st.header("Market Basket Analysis System")

    st.write("""
This system is a software application developed to perform
Market Basket Analysis using the Apriori Algorithm.

The system analyzes transaction data to discover relationships
between products that are frequently purchased together.
""")

    st.subheader("🎯 Purpose of the System")

    st.write("""
The purpose of this system is to help businesses understand
customer purchasing patterns and identify products that are
frequently bought together.
""")

    st.subheader("⚙️ Technologies Used")

    technologies = [
        "Python",
        "Streamlit",
        "Pandas",
        "mlxtend",
        "Plotly",
        "Apriori Algorithm",
        "Association Rule Mining"
    ]

    for technology in technologies:
        st.write(f"• {technology}")

    st.subheader("📊 Analysis Measures")

    st.write("""
The system uses the following measures to evaluate association rules:
""")

    st.write("• Support – shows how frequently items occur together.")
    st.write("• Confidence – shows how often the consequent occurs when the antecedent occurs.")
    st.write("• Lift – measures the strength of the relationship between items.")

    st.subheader("🛒 System Features")

    st.write("""
• Dataset management  
• Data cleaning  
• Transaction processing  
• Frequent itemset generation  
• Association rule generation  
• Data visualization  
• Results export
""")

    st.info(
        "This system was developed as a final-year Computer Science project."
    )