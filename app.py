"""
Marketing Analytics Dashboard
Enhanced for Infosys Data Science Analyst JD Requirements

Author: [Your Name]
Date: November 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
import os
from scipy import stats as scipy_stats

# Import custom modules
from modules.sql_loader import DataLoader
from modules.churn_predictor import ChurnPredictor
from modules.statistical_analyzer import StatisticalAnalyzer

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Marketing Analytics Dashboard", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📊 Marketing Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown("**Comprehensive CRM Analytics with SQL Integration, Predictive Modeling & Statistical Analysis**")

# Initialize session state
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False

@st.cache_data
def load_data_from_csv():
    """Load data from CSV file"""
    try:
        csv_path = "data/Churn_pred.csv"
        if not os.path.exists(csv_path):
            st.error(f"❌ File not found: {csv_path}")
            st.info("📁 Please ensure 'Churn_pred.csv' is in the 'data/' folder")
            return None
        
        df = pd.read_csv(csv_path)
        
        # Data preprocessing
        df_original = df.copy()
        df.drop(columns=['customerID'], errors='ignore', inplace=True)
        df = df.dropna()
        
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df = df.dropna(subset=['TotalCharges'])
        
        # Encode categorical variables
        categorical_cols = df.select_dtypes(include='object').columns
        df_encoded = df.copy()
        
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        # Scale numerical variables
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        scaler = StandardScaler()
        df_scaled = df_encoded.copy()
        df_scaled[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
        
        st.success(f"✅ Data loaded successfully! {len(df)} records")
        
        return df_encoded, df_scaled, df, scaler, label_encoders, df_original
    
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

# Load data
data_result = load_data_from_csv()
if data_result is None:
    st.stop()

df_encoded, df_scaled, df_original, scaler, label_encoders, df_raw = data_result

# Sidebar Navigation
st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("---")

analysis_type = st.sidebar.radio(
    "Select Analysis Module:",
    [
        "🏠 Dashboard Overview",
        "🗄️ SQL Data Extraction",
        "🔮 Churn Prediction",
        "📈 Statistical Analysis",
        "💰 CLV Analysis",
        "🎯 Customer Segmentation",
        "🛒 Market Basket Analysis",
        "🧪 A/B Testing"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Project Features:**
- SQL Database Integration ✅
- Multiple ML Models ✅
- Statistical Hypothesis Testing ✅
- Business Rule Translation ✅
- Data Visualization ✅
""")

# ========================================
# 1. DASHBOARD OVERVIEW
# ========================================
if analysis_type == "🏠 Dashboard Overview":
    st.header("📋 Executive Dashboard Overview")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = len(df_encoded)
        st.metric("👥 Total Customers", f"{total_customers:,}")
    
    with col2:
        avg_tenure = df_encoded['tenure'].mean()
        st.metric("📅 Avg Tenure", f"{avg_tenure:.1f} months")
    
    with col3:
        avg_monthly = df_encoded['MonthlyCharges'].mean()
        st.metric("💳 Avg Monthly", f"${avg_monthly:.2f}")
    
    with col4:
        if 'Churn' in df_encoded.columns:
            churn_rate = (df_original['Churn'] == 'Yes').mean() * 100
            st.metric("⚠️ Churn Rate", f"{churn_rate:.1f}%", delta=f"-{churn_rate:.1f}%", delta_color="inverse")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Customer Distribution by Tenure")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df_encoded['tenure'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Tenure (months)')
        ax.set_ylabel('Number of Customers')
        ax.set_title('Customer Tenure Distribution')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader("💵 Monthly Charges Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df_encoded['MonthlyCharges'], bins=30, color='coral', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Monthly Charges ($)')
        ax.set_ylabel('Number of Customers')
        ax.set_title('Monthly Charges Distribution')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("🎯 Key Business Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📌 Customer Retention:**
        - Focus on customers with <12 months tenure
        - Implement early engagement programs
        - Offer contract incentives
        """)
    
    with col2:
        st.markdown("""
        **📌 Revenue Optimization:**
        - Identify high-value customer segments
        - Cross-sell opportunities
        - Personalized pricing strategies
        """)

# ========================================
# 2. SQL DATA EXTRACTION
# ========================================
elif analysis_type == "🗄️ SQL Data Extraction":
    st.header("🗄️ SQL Data Extraction Module")
    st.markdown("**Demonstrates: SQL coding skills as per JD requirements**")
    
    # Initialize DataLoader
    loader = DataLoader()
    
    # Database initialization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📁 Database Setup")
        
        if not st.session_state.db_initialized:
            if st.button("🔄 Initialize Database from CSV", type="primary"):
                with st.spinner("Creating database..."):
                    success = loader.create_database_from_csv("data/Churn_pred.csv")
                    if success:
                        st.session_state.db_initialized = True
                        st.success("✅ Database created successfully!")
                        st.rerun()
        else:
            st.success("✅ Database is ready!")
            if st.button("🔄 Recreate Database"):
                success = loader.create_database_from_csv("data/Churn_pred.csv")
                if success:
                    st.success("✅ Database recreated!")
    
    with col2:
        st.info("""
        **Database Info:**
        - Type: SQLite
        - Table: customers
        - Location: data/
        """)
    
    if st.session_state.db_initialized:
        st.markdown("---")
        st.subheader("📊 SQL Query Examples")
        
        query_option = st.selectbox(
            "Select SQL Query to Execute:",
            [
                "Customer Summary Statistics",
                "High-Value Customers (>$100/month)",
                "Churn Analysis by Contract Type",
                "Revenue Analysis by Segment",
                "Data Extraction for ML"
            ]
        )
        
        if st.button("▶️ Execute Query", type="primary"):
            
            if query_option == "Customer Summary Statistics":
                st.code("""
SELECT 
    COUNT(*) as total_customers,
    ROUND(AVG(tenure), 2) as avg_tenure,
    ROUND(AVG(MonthlyCharges), 2) as avg_monthly_charges,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned_customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) as churn_rate
FROM customers
                """, language="sql")
                
                result = loader.get_customer_summary()
                if result is not None:
                    st.dataframe(result, use_container_width=True)
                
            elif query_option == "High-Value Customers (>$100/month)":
                st.code("""
SELECT customerID, tenure, MonthlyCharges, TotalCharges, Contract
FROM customers
WHERE MonthlyCharges >= 100
ORDER BY TotalCharges DESC
LIMIT 100
                """, language="sql")
                
                result = loader.get_high_value_customers()
                if result is not None:
                    st.write(f"**Found {len(result)} high-value customers**")
                    st.dataframe(result.head(20), use_container_width=True)
                
            elif query_option == "Churn Analysis by Contract Type":
                st.code("""
SELECT 
    Contract,
    COUNT(*) as customer_count,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) as churn_rate
FROM customers
GROUP BY Contract
                """, language="sql")
                
                result = loader.get_churn_analysis_by_contract()
                if result is not None:
                    st.dataframe(result, use_container_width=True)
                    
                    # Visualization
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.bar(result['Contract'], result['churn_rate_percent'], color='coral')
                    ax.set_xlabel('Contract Type')
                    ax.set_ylabel('Churn Rate (%)')
                    ax.set_title('Churn Rate by Contract Type')
                    ax.grid(axis='y', alpha=0.3)
                    st.pyplot(fig)
                
            elif query_option == "Revenue Analysis by Segment":
                result = loader.get_revenue_by_segment()
                if result is not None:
                    st.dataframe(result, use_container_width=True)
                
            elif query_option == "Data Extraction for ML":
                st.code("""
SELECT *
FROM customers
WHERE TotalCharges IS NOT NULL 
    AND tenure > 0
                """, language="sql")
                
                result = loader.get_customers_for_modeling()
                if result is not None:
                    st.write(f"**Extracted {len(result)} records for ML modeling**")
                    st.dataframe(result.head(10), use_container_width=True)
        
        loader.close_connection()

# ========================================
# 3. CHURN PREDICTION
# ========================================
elif analysis_type == "🔮 Churn Prediction":
    st.header("🔮 Customer Churn Prediction Module")
    st.markdown("**Demonstrates: Classification, Multiple ML Models, Feature Importance**")
    
    # Initialize predictor
    predictor = ChurnPredictor(df_raw)
    
    # Prepare data button
    if st.button("🔄 Prepare Data & Train Models", type="primary"):
        with st.spinner("Preparing data and training models... This may take a minute."):
            predictor.prepare_data()
            predictor.train_models()
            
            # Store in session state
            st.session_state.churn_predictor = predictor
            st.success("✅ Models trained successfully!")
    
    # Check if models are trained
    if 'churn_predictor' in st.session_state:
        predictor = st.session_state.churn_predictor
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Comparison", "🎯 Feature Importance", 
                                          "💼 Business Rules", "🔮 Individual Prediction"])
        
        with tab1:
            st.subheader("📊 Model Performance Comparison")
            
            comparison = predictor.get_model_comparison()
            st.dataframe(comparison, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = predictor.plot_model_comparison()
                st.pyplot(fig1)
            
            with col2:
                fig2 = predictor.plot_confusion_matrices()
                st.pyplot(fig2)
            
            # Best model highlight
            best_model = comparison.loc[comparison['ROC-AUC'].idxmax(), 'Model']
            best_auc = comparison.loc[comparison['ROC-AUC'].idxmax(), 'ROC-AUC']
            st.success(f"🏆 **Best Model**: {best_model} (ROC-AUC: {best_auc:.4f})")
        
        with tab2:
            st.subheader("🎯 Feature Importance Analysis")
            
            model_choice = st.selectbox("Select Model", list(predictor.models.keys()))
            
            fig3 = predictor.plot_feature_importance(model_choice)
            if fig3:
                st.pyplot(fig3)
                
                # Show feature importance table
                feature_imp = predictor.get_feature_importance(model_choice)
                st.dataframe(feature_imp, use_container_width=True)
            else:
                st.warning("Feature importance not available for this model.")
        
        with tab3:
            st.subheader("💼 Business Rules Translation")
            st.markdown("**KEY JD REQUIREMENT: Translate model parameters into implementable business rules**")
            
            model_choice2 = st.selectbox("Select Model for Business Rules", 
                                        list(predictor.models.keys()), key="business_rules")
            
            business_rules = predictor.translate_to_business_rules(model_choice2)
            st.markdown(business_rules)
        
        with tab4:
            st.subheader("🔮 Individual Customer Churn Prediction")
            st.markdown("**Enter customer details to predict churn risk:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                tenure_pred = st.slider("Tenure (months)", 0, 72, 12, key="pred_tenure")
                monthly_pred = st.slider("Monthly Charges ($)", 18, 120, 65, key="pred_monthly")
            
            with col2:
                total_pred = st.slider("Total Charges ($)", 18, 8500, 780, key="pred_total")
                contract_pred = st.selectbox("Contract Type", [0, 1, 2], 
                                            format_func=lambda x: ["Month-to-month", "One year", "Two year"][x])
            
            with col3:
                internet_pred = st.selectbox("Internet Service", [0, 1, 2],
                                            format_func=lambda x: ["No", "DSL", "Fiber optic"][x])
                support_pred = st.selectbox("Tech Support", [0, 1],
                                           format_func=lambda x: ["No", "Yes"][x])
            
            st.info("ℹ️ Note: This is a simplified prediction. Full implementation would include all features.")

# ========================================
# 4. STATISTICAL ANALYSIS
# ========================================
elif analysis_type == "📈 Statistical Analysis":
    st.header("📈 Statistical Analysis Module")
    st.markdown("**Demonstrates: Hypothesis testing, statistical concepts (JD requirement)**")
    
    analyzer = StatisticalAnalyzer(df_raw)
    
    analysis_subtype = st.selectbox(
        "Select Analysis Type:",
        ["Hypothesis Testing (Numeric)", "Hypothesis Testing (Categorical)", 
         "Correlation Analysis"]
    )
    
    if analysis_subtype == "Hypothesis Testing (Numeric)":
        st.subheader("🔬 T-Test for Numeric Features")
        
        numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_raw.select_dtypes(include='object').columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            feature = st.selectbox("Select Numeric Feature", numeric_cols)
            group_column = st.selectbox("Group By", categorical_cols)
        
        with col2:
            unique_values = df_raw[group_column].unique()[:10]  # Limit to 10 for performance
            group1 = st.selectbox("Group 1", unique_values)
            group2 = st.selectbox("Group 2", [v for v in unique_values if v != group1])
        
        if st.button("▶️ Run Hypothesis Test"):
            results = analyzer.hypothesis_test_numeric(feature, group_column, group1, group2)
            
            st.markdown("### 📋 Test Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Test Statistic", f"{results['statistic']:.4f}")
            with col2:
                st.metric("P-value", f"{results['p_value']:.4f}")
            with col3:
                st.metric("Effect Size", f"{results['cohens_d']:.3f} ({results['effect_size']})")
            
            if results['significant']:
                st.success(f"✅ **Statistically Significant** (p < {analyzer.alpha})")
                st.write("**Conclusion**: There IS a significant difference between the groups.")
            else:
                st.info(f"❌ **Not Significant** (p >= {analyzer.alpha})")
                st.write("**Conclusion**: NO significant difference between the groups.")
            
            # Group statistics
            st.markdown("### 📊 Group Statistics")
            stats_df = pd.DataFrame({
                'Group': [group1, group2],
                'Mean': [results['group1_mean'], results['group2_mean']],
                'Std Dev': [results['group1_std'], results['group2_std']],
                'Sample Size': [results['group1_size'], results['group2_size']]
            })
            st.dataframe(stats_df, use_container_width=True)
            
            # Visualization
            fig = analyzer.plot_hypothesis_test(feature, group_column, group1, group2)
            st.pyplot(fig)
    
    elif analysis_subtype == "Hypothesis Testing (Categorical)":
        st.subheader("🔬 Chi-Square Test for Categorical Features")
        
        categorical_cols = df_raw.select_dtypes(include='object').columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            feature1 = st.selectbox("Feature 1", categorical_cols)
        with col2:
            feature2 = st.selectbox("Feature 2", [c for c in categorical_cols if c != feature1])
        
        if st.button("▶️ Run Chi-Square Test"):
            results = analyzer.hypothesis_test_categorical(feature1, feature2)
            
            st.markdown("### 📋 Test Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("χ² Statistic", f"{results['chi2_statistic']:.4f}")
            with col2:
                st.metric("P-value", f"{results['p_value']:.4f}")
            with col3:
                st.metric("Cramér's V", f"{results['cramers_v']:.3f} ({results['effect_size']})")
            
            if results['significant']:
                st.success(f"✅ **Statistically Significant** (p < {analyzer.alpha})")
                st.write(f"**Conclusion**: {feature1} and {feature2} ARE associated.")
            else:
                st.info(f"❌ **Not Significant** (p >= {analyzer.alpha})")
                st.write(f"**Conclusion**: {feature1} and {feature2} are INDEPENDENT.")
            
            st.markdown("### 📊 Contingency Table")
            st.dataframe(results['contingency_table'], use_container_width=True)
    
    elif analysis_subtype == "Correlation Analysis":
        st.subheader("🔗 Correlation Analysis")
        
        numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        
        selected_features = st.multiselect("Select Features (leave empty for all)", 
                                          numeric_cols, default=numeric_cols[:5])
        
        if not selected_features:
            selected_features = numeric_cols
        
        if st.button("📊 Calculate Correlations"):
            corr_matrix, corr_pairs = analyzer.correlation_analysis(selected_features)
            
            fig = analyzer.plot_correlation_heatmap(selected_features)
            st.pyplot(fig)
            
            st.markdown("### 🔝 Strongest Correlations")
            st.dataframe(corr_pairs.head(10), use_container_width=True)

# ========================================
# 5. CLV ANALYSIS (Original)
# ========================================
elif analysis_type == "💰 CLV Analysis":
    st.header("💰 Customer Lifetime Value (CLV) Analysis")
    
    df_encoded['CLV'] = (df_encoded['MonthlyCharges'] * df_encoded['tenure'] * 
                        (1 - df_encoded.get('Churn', 0) * 0.5))
    
    X_clv = df_encoded[['tenure', 'MonthlyCharges', 'TotalCharges']]
    y_clv = df_encoded['CLV']
    
    clv_model = LinearRegression()
    clv_model.fit(X_clv, y_clv)
    clv_r2 = r2_score(y_clv, clv_model.predict(X_clv))
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎯 CLV Predictor")
        st.write(f"**Model Accuracy (R²): {clv_r2:.3f}**")
        
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
        total_charges = st.slider("Total Charges ($)", 18.0, 8500.0, float(tenure * monthly_charges))
        
        input_data = pd.DataFrame([[tenure, monthly_charges, total_charges]],
                                  columns=['tenure', 'MonthlyCharges', 'TotalCharges'])
        clv_prediction = clv_model.predict(input_data)[0]
        
        st.metric("💰 Estimated CLV", f"${clv_prediction:.2f}")
        
        if clv_prediction > df_encoded['CLV'].quantile(0.75):
            st.success("🌟 High Value Customer")
        elif clv_prediction > df_encoded['CLV'].quantile(0.25):
            st.info("📈 Medium Value Customer")
        else:
            st.warning("⚠️ Low Value Customer")
    
    with col2:
        st.subheader("📊 CLV Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df_encoded['CLV'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(clv_prediction, color='red', linestyle='--', linewidth=2, 
                  label=f'Predicted: ${clv_prediction:.2f}')
        ax.set_xlabel('Customer Lifetime Value ($)')
        ax.set_ylabel('Count')
        ax.set_title('CLV Distribution')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)

# ========================================
# 6. CUSTOMER SEGMENTATION
# ========================================
elif analysis_type == "🎯 Customer Segmentation":
    st.header("🎯 Customer Segmentation Analysis")
    st.markdown("**Demonstrates: K-Means Clustering, Customer Profiling, Marketing Strategy Development**")
    
    # Perform clustering
    features_for_clustering = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Number of clusters selection
    col1, col2 = st.columns([1, 3])
    with col1:
        n_clusters = st.selectbox("Number of Segments", [3, 4, 5], index=1)
    
    with col2:
        st.info(f"💡 Segmenting customers into {n_clusters} groups based on tenure and spending patterns")
    
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_encoded['Segment'] = kmeans.fit_predict(df_scaled[features_for_clustering])
    
    # Create meaningful segment labels based on characteristics
    segment_profiles = df_encoded.groupby('Segment')[features_for_clustering].mean()
    
    # Create better segment names
    segment_names = {}
    for i in range(n_clusters):
        tenure_avg = segment_profiles.loc[i, 'tenure']
        charges_avg = segment_profiles.loc[i, 'MonthlyCharges']
        
        if tenure_avg > df_encoded['tenure'].mean() and charges_avg > df_encoded['MonthlyCharges'].mean():
            segment_names[i] = 'Loyal High-Spenders'
        elif tenure_avg > df_encoded['tenure'].mean():
            segment_names[i] = 'Loyal Budget-Conscious'
        elif charges_avg > df_encoded['MonthlyCharges'].mean():
            segment_names[i] = 'New High-Spenders'
        else:
            segment_names[i] = 'New Budget-Conscious'
    
    # Handle duplicate names
    name_counts = {}
    for key, name in segment_names.items():
        if name in name_counts:
            name_counts[name] += 1
            segment_names[key] = f"{name} {name_counts[name]}"
        else:
            name_counts[name] = 1
    
    df_encoded['Segment_Name'] = df_encoded['Segment'].map(segment_names)
    
    # Visualization
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📈 Customer Segment Distribution")
        seg_counts = df_encoded['Segment_Name'].value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC'][:n_clusters]
        wedges, texts, autotexts = ax.pie(
            seg_counts.values, 
            labels=seg_counts.index, 
            autopct='%1.1f%%', 
            colors=colors, 
            startangle=90,
            textprops={'fontsize': 10}
        )
        ax.set_title('Customer Segment Distribution', fontsize=14, pad=20)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("🎯 Segment Characteristics")
        segment_details = df_encoded.groupby('Segment_Name').agg({
            'tenure': 'mean',
            'MonthlyCharges': 'mean',
            'TotalCharges': 'mean',
            'Segment': 'count'
        }).round(2)
        segment_details.columns = ['Avg Tenure (months)', 'Avg Monthly ($)', 'Avg Total ($)', 'Count']
        st.dataframe(segment_details, use_container_width=True)
    
    # Scatter plot visualization
    st.subheader("📊 Customer Segmentation Visualization")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Tenure vs Monthly Charges
    for segment in df_encoded['Segment'].unique():
        mask = df_encoded['Segment'] == segment
        axes[0].scatter(
            df_encoded[mask]['tenure'],
            df_encoded[mask]['MonthlyCharges'],
            label=segment_names[segment],
            alpha=0.6,
            s=50
        )
    axes[0].set_xlabel('Tenure (months)')
    axes[0].set_ylabel('Monthly Charges ($)')
    axes[0].set_title('Segmentation: Tenure vs Monthly Charges')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Monthly Charges vs Total Charges
    for segment in df_encoded['Segment'].unique():
        mask = df_encoded['Segment'] == segment
        axes[1].scatter(
            df_encoded[mask]['MonthlyCharges'],
            df_encoded[mask]['TotalCharges'],
            label=segment_names[segment],
            alpha=0.6,
            s=50
        )
    axes[1].set_xlabel('Monthly Charges ($)')
    axes[1].set_ylabel('Total Charges ($)')
    axes[1].set_title('Segmentation: Monthly vs Total Charges')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Marketing strategies
    st.markdown("---")
    st.subheader("💡 Recommended Marketing Strategies")
    
    strategies = {
        'Loyal High-Spenders': {
            'icon': '🌟',
            'strategy': 'VIP treatment, exclusive offers, premium support, loyalty rewards',
            'tactics': [
                '• Dedicated account manager',
                '• Early access to new products',
                '• Premium customer service line',
                '• Annual appreciation events'
            ]
        },
        'Loyal Budget-Conscious': {
            'icon': '🎁',
            'strategy': 'Volume discounts, referral bonuses, appreciation programs',
            'tactics': [
                '• Long-term contract discounts',
                '• Referral reward programs',
                '• Loyalty points system',
                '• Bundle package offers'
            ]
        },
        'New High-Spenders': {
            'icon': '🚀',
            'strategy': 'Premium service upgrades, early access to new features',
            'tactics': [
                '• Onboarding assistance',
                '• Premium feature trials',
                '• Upgrade incentives',
                '• Quick win demonstrations'
            ]
        },
        'New Budget-Conscious': {
            'icon': '💰',
            'strategy': 'Welcome discounts, free trials, gradual upselling',
            'tactics': [
                '• Welcome bonus offers',
                '• Free trial periods',
                '• Educational content',
                '• Gradual feature introduction'
            ]
        }
    }
    
    for segment_name in seg_counts.index:
        base_name = segment_name.split(' ')[0] + ' ' + segment_name.split(' ')[1] if len(segment_name.split()) > 2 else segment_name
        
        for key, value in strategies.items():
            if key in segment_name or base_name == key:
                with st.expander(f"{value['icon']} **{segment_name}** ({seg_counts[segment_name]} customers)"):
                    st.write(f"**Strategy**: {value['strategy']}")
                    st.write("**Tactics**:")
                    for tactic in value['tactics']:
                        st.write(tactic)
                break

# ========================================
# 7. MARKET BASKET ANALYSIS
# ========================================
elif analysis_type == "🛒 Market Basket Analysis":
    st.header("🛒 Market Basket Analysis (Cross-Sell Opportunities)")
    st.markdown("**Demonstrates: Apriori Algorithm, Association Rules, Cross-Selling Strategy**")
    
    # Select boolean/binary service columns
    service_cols = [col for col in df_original.columns if df_original[col].nunique() == 2 and col != 'Churn']
    
    if len(service_cols) > 0:
        st.subheader("📊 Service Adoption Analysis")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(f"**Analyzing {len(service_cols)} services** for cross-selling opportunities")
        with col2:
            min_support = st.slider("Minimum Support", 0.05, 0.3, 0.1, 0.05)
        
        # Convert to boolean for market basket analysis
        df_basket = df_original[service_cols].copy()
        
        # Handle different encoding formats (Yes/No, 1/0)
        for col in service_cols:
            unique_vals = df_basket[col].unique()
            if 'Yes' in unique_vals:
                df_basket[col] = (df_basket[col] == 'Yes')
            elif 'No' in unique_vals:
                df_basket[col] = (df_basket[col] != 'No')
            else:
                df_basket[col] = df_basket[col].astype(bool)
        
        # Service adoption rates
        st.subheader("📈 Individual Service Adoption Rates")
        adoption_rates = df_basket.mean().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(adoption_rates.index, adoption_rates.values * 100, color='steelblue')
        ax.set_xlabel('Adoption Rate (%)')
        ax.set_title('Service Adoption Rates')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        for i, v in enumerate(adoption_rates.values * 100):
            ax.text(v + 1, i, f'{v:.1f}%', va='center')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Generate frequent itemsets
        st.markdown("---")
        st.subheader("🔗 Association Rules Mining")
        
        try:
            with st.spinner("Mining association rules..."):
                frequent_itemsets = apriori(df_basket, min_support=min_support, use_colnames=True)
            
            if len(frequent_itemsets) > 0:
                st.success(f"✅ Found {len(frequent_itemsets)} frequent itemsets")
                
                # Generate association rules
                rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.1)
                
                if len(rules) > 0:
                    st.success(f"✅ Generated {len(rules)} association rules")
                    
                    # Convert frozensets to strings
                    rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                    rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                    
                    # Display top rules
                    st.subheader("🎯 Top Cross-Sell Opportunities")
                    
                    # Filter and sort rules
                    top_rules = rules.nlargest(10, 'lift')[
                        ['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']
                    ]
                    top_rules.columns = ['If Customer Has', 'Recommend', 'Support', 'Confidence', 'Lift']
                    
                    # Style the dataframe
                    st.dataframe(
                        top_rules.style.format({
                            'Support': '{:.3f}',
                            'Confidence': '{:.3f}',
                            'Lift': '{:.2f}'
                        }).background_gradient(subset=['Lift'], cmap='RdYlGn'),
                        use_container_width=True
                    )
                    
                    # Explanation
                    st.info("""
                    **Metric Definitions:**
                    - **Support**: How frequently the itemset appears in the dataset
                    - **Confidence**: Likelihood of buying consequent if antecedent is purchased
                    - **Lift**: How much more likely the consequent is purchased when antecedent is purchased (>1 is good)
                    """)
                    
                    # Service adoption heatmap
                    st.markdown("---")
                    st.subheader("🔥 Service Correlation Heatmap")
                    
                    corr_matrix = df_basket.corr()
                    fig, ax = plt.subplots(figsize=(12, 10))
                    sns.heatmap(
                        corr_matrix, 
                        annot=True, 
                        cmap='coolwarm', 
                        center=0, 
                        ax=ax,
                        fmt='.2f',
                        square=True,
                        linewidths=0.5,
                        cbar_kws={"shrink": 0.8}
                    )
                    ax.set_title('Service Correlation Matrix', fontsize=14, pad=20)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Marketing insights
                    st.markdown("---")
                    st.subheader("💡 Key Marketing Insights & Action Items")
                    
                    if len(rules) > 0:
                        best_rule = rules.loc[rules['lift'].idxmax()]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Best Lift", f"{best_rule['lift']:.2f}")
                        with col2:
                            st.metric("Confidence", f"{best_rule['confidence']:.1%}")
                        with col3:
                            st.metric("Support", f"{best_rule['support']:.1%}")
                        
                        st.success(f"""
                        🎯 **Best Cross-sell Opportunity**: 
                        
                        If customer has **{list(best_rule['antecedents'])[0]}**, 
                        recommend **{list(best_rule['consequents'])[0]}** 
                        
                        (They are {best_rule['lift']:.2f}x more likely to adopt it)
                        """)
                        
                        st.write("**📋 Recommended Actions:**")
                        
                        for idx, rule in rules.head(5).iterrows():
                            antecedent = list(rule['antecedents'])[0]
                            consequent = list(rule['consequents'])[0]
                            
                            with st.expander(f"Rule {idx + 1}: {antecedent} → {consequent}"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Marketing Campaign:**")
                                    st.write(f"• Target customers with **{antecedent}**")
                                    st.write(f"• Promote **{consequent}** bundle")
                                    st.write(f"• Expected success rate: {rule['confidence']:.1%}")
                                with col2:
                                    st.write("**Business Impact:**")
                                    st.write(f"• Lift factor: {rule['lift']:.2f}x")
                                    st.write(f"• Potential reach: {rule['support']:.1%} of customers")
                                    st.write(f"• High-confidence opportunity")
                else:
                    st.warning("⚠️ No significant association rules found. Try lowering the minimum support threshold.")
            else:
                st.warning("⚠️ No frequent itemsets found. Try lowering the minimum support threshold.")
        
        except Exception as e:
            st.error(f"❌ Error in market basket analysis: {str(e)}")
            st.info("💡 Try adjusting the minimum support threshold")
    
    else:
        st.warning("⚠️ No suitable service columns found for market basket analysis in this dataset.")

# ========================================
# 8. A/B TESTING
# ========================================
elif analysis_type == "🧪 A/B Testing":
    st.header("🧪 A/B Testing Simulation")
    st.markdown("**Demonstrates: Statistical Testing, Campaign Optimization, Data-Driven Decision Making**")
    
    st.info("💡 Simulate marketing campaigns to determine which strategy performs better")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Test Configuration")
        
        # Test parameters
        group_size = st.number_input(
            "Sample Size per Group", 
            min_value=100, 
            max_value=10000, 
            value=1000, 
            step=100,
            help="Number of customers in each test group"
        )
        
        st.markdown("**Strategy A (Control)**")
        strategy_A = st.text_input("Strategy A Name", value="Discount Offer", key="strat_a")
        conversion_rate_A = st.slider(
            "Strategy A Conversion Rate", 
            0.01, 0.50, 0.15, 0.01,
            help="Expected conversion rate for Strategy A",
            key="conv_a"
        )
        
        st.markdown("**Strategy B (Variant)**")
        strategy_B = st.text_input("Strategy B Name", value="Premium Support", key="strat_b")
        conversion_rate_B = st.slider(
            "Strategy B Conversion Rate", 
            0.01, 0.50, 0.20, 0.01,
            help="Expected conversion rate for Strategy B",
            key="conv_b"
        )
        
        # Run simulation button
        run_test = st.button("🚀 Run A/B Test", type="primary")
    
    with col2:
        st.subheader("📈 Results & Analysis")
        
        if run_test:
            # Run simulation
            np.random.seed(42)  # For reproducible results
            conversions_A = np.random.binomial(group_size, conversion_rate_A)
            conversions_B = np.random.binomial(group_size, conversion_rate_B)
            
            # Calculate metrics
            actual_rate_A = conversions_A / group_size
            actual_rate_B = conversions_B / group_size
            difference = abs(conversions_B - conversions_A)
            relative_improvement = (conversions_B - conversions_A) / conversions_A * 100 if conversions_A > 0 else 0
            
            # Display metrics
            col2_1, col2_2 = st.columns(2)
            
            with col2_1:
                st.metric(
                    label=f"🅰️ {strategy_A}", 
                    value=f"{conversions_A:,}",
                    delta=f"{actual_rate_A*100:.1f}% conversion"
                )
            
            with col2_2:
                st.metric(
                    label=f"🅱️ {strategy_B}", 
                    value=f"{conversions_B:,}",
                    delta=f"{actual_rate_B*100:.1f}% conversion",
                    delta_color="normal"
                )
            
            # Statistical significance (Z-test for proportions)
            from scipy import stats as scipy_stats
            
            # Pooled proportion
            p_pool = (conversions_A + conversions_B) / (2 * group_size)
            se = np.sqrt(p_pool * (1 - p_pool) * (2 / group_size))
            z_score = (actual_rate_B - actual_rate_A) / se if se > 0 else 0
            p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_score)))
            
            # Visualization
            st.markdown("---")
            fig, ax = plt.subplots(figsize=(10, 6))
            strategies = [strategy_A, strategy_B]
            conversions = [conversions_A, conversions_B]
            colors = ['#FF6B6B', '#4ECDC4']
            
            bars = ax.bar(strategies, conversions, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            ax.set_ylabel("Number of Conversions", fontsize=12)
            ax.set_title("A/B Test Results Comparison", fontsize=14, pad=20)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, conv, rate in zip(bars, conversions, [actual_rate_A, actual_rate_B]):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2., 
                    height + 5,
                    f'{conv:,}\n({rate*100:.1f}%)', 
                    ha='center', 
                    va='bottom',
                    fontsize=11,
                    fontweight='bold'
                )
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Statistical conclusion
            st.markdown("---")
            st.subheader("📊 Statistical Analysis")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Absolute Difference", f"{difference:,} conversions")
            with col2:
                st.metric("Relative Improvement", f"{relative_improvement:+.1f}%")
            with col3:
                st.metric("P-value", f"{p_value:.4f}")
            
            # Winner determination
            st.markdown("---")
            if p_value < 0.05:
                if conversions_B > conversions_A:
                    st.success(f"""
                    ✅ **{strategy_B} WINS!** (Statistically Significant)
                    
                    - **Improvement**: +{relative_improvement:.1f}%
                    - **Additional Conversions**: {difference:,}
                    - **Confidence**: 95%+
                    - **Recommendation**: Implement Strategy B
                    """)
                else:
                    st.success(f"""
                    ✅ **{strategy_A} WINS!** (Statistically Significant)
                    
                    - **Improvement**: +{abs(relative_improvement):.1f}%
                    - **Additional Conversions**: {difference:,}
                    - **Confidence**: 95%+
                    - **Recommendation**: Continue with Strategy A
                    """)
            else:
                st.info(f"""
                🤝 **NO CLEAR WINNER** (Not Statistically Significant)
                
                - **P-value**: {p_value:.4f} (>0.05)
                - **Conclusion**: Results are inconclusive
                - **Recommendation**: 
                  - Increase sample size
                  - Run test longer
                  - Consider other metrics
                """)
            
            # Detailed summary
            st.markdown("---")
            st.subheader("📋 Detailed Test Summary")
            
            summary_df = pd.DataFrame({
                'Strategy': [strategy_A, strategy_B],
                'Sample Size': [group_size, group_size],
                'Conversions': [conversions_A, conversions_B],
                'Conversion Rate': [f"{actual_rate_A*100:.2f}%", f"{actual_rate_B*100:.2f}%"],
                'Difference': ['-', f"{relative_improvement:+.1f}%"]
            })
            
            st.dataframe(summary_df, use_container_width=True)
            
            st.write("**Statistical Metrics:**")
            st.write(f"• Z-score: {z_score:.4f}")
            st.write(f"• P-value: {p_value:.4f}")
            st.write(f"• Significance Level (α): 0.05")
            st.write(f"• Statistical Power: {1 - p_value:.2%}" if p_value < 1 else "• Statistical Power: N/A")
            
            # Recommendations
            st.markdown("---")
            st.subheader("💡 Marketing Recommendations")
            
            if p_value < 0.05 and conversions_B > conversions_A:
                st.write(f"""
                **Action Plan:**
                1. ✅ Roll out **{strategy_B}** to all customers
                2. 📊 Monitor performance metrics closely
                3. 💰 Estimate ROI: {relative_improvement:.1f}% increase in conversions
                4. 📈 Scale gradually to validate results
                5. 🔄 Continue testing variations for further optimization
                """)
            elif p_value < 0.05:
                st.write(f"""
                **Action Plan:**
                1. ✅ Continue with **{strategy_A}** as the winning strategy
                2. 🔍 Analyze why {strategy_B} underperformed
                3. 🧪 Test alternative variations
                4. 📊 Document learnings for future campaigns
                """)
            else:
                st.write(f"""
                **Action Plan:**
                1. ⏳ **Extend Test Duration**: Collect more data
                2. 📈 **Increase Sample Size**: Target {group_size * 2:,}+ per group
                3. 🎯 **Refine Strategies**: Optimize messaging/targeting
                4. 🔍 **Analyze Segments**: Different strategies may work for different customer groups
                5. 📊 **Monitor Secondary Metrics**: Revenue, retention, satisfaction
                """)
        
        else:
            st.info("👆 Configure your test parameters and click 'Run A/B Test' to see results")
            
            st.markdown("---")
            st.subheader("📖 How to Use This Tool")
            st.write("""
            **Steps:**
            1. Set your sample size (customers per group)
            2. Define Strategy A (control/baseline)
            3. Define Strategy B (new variant to test)
            4. Set expected conversion rates
            5. Click 'Run A/B Test' to simulate
            
            **Interpreting Results:**
            - **P-value < 0.05**: Statistically significant difference
            - **Lift**: Percentage improvement over control
            - **Winner**: The strategy with higher conversions (if significant)
            """)


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📊 Marketing Analytics Dashboard | Built with Streamlit & Python</p>
    <p>Demonstrates: SQL Skills • Machine Learning • Statistical Analysis • Business Intelligence</p>
</div>
""", unsafe_allow_html=True)