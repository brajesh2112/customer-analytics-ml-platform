import sqlite3
import pandas as pd
import streamlit as st

class DataLoader:
    """
    Data loading module with SQL database integration
    Demonstrates SQL skills as per JD requirements
    """
    
    def __init__(self, db_name="customer_data.db"):
        self.db_name = db_name
        self.conn = None
    
    def create_database_from_csv(self, csv_path):
        """
        Create SQLite database from CSV file
        Simulates extracting data from RDBMS
        """
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Create SQLite connection
        self.conn = sqlite3.connect(self.db_name)
        
        # Load data into SQL table
        df.to_sql('customers', self.conn, if_exists='replace', index=False)
        
        print(f"✅ Database created: {self.db_name}")
        return self.conn
    
    def execute_sql_query(self, query):
        """
        Execute SQL query and return DataFrame
        Demonstrates SQL coding skills
        """
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_name)
        
        return pd.read_sql_query(query, self.conn)
    
    def get_customer_summary(self):
        """
        SQL: Basic aggregation query
        """
        query = """
        SELECT 
            COUNT(*) as total_customers,
            AVG(tenure) as avg_tenure,
            AVG(MonthlyCharges) as avg_monthly_charges,
            AVG(TotalCharges) as avg_total_charges,
            SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned_customers,
            ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) as churn_rate
        FROM customers
        """
        return self.execute_sql_query(query)
    
    def get_high_value_customers(self, min_charges=100):
        """
        SQL: Filter high-value customers
        """
        query = f"""
        SELECT 
            customerID,
            tenure,
            MonthlyCharges,
            TotalCharges,
            Contract,
            PaymentMethod
        FROM customers
        WHERE MonthlyCharges >= {min_charges}
        ORDER BY TotalCharges DESC
        LIMIT 100
        """
        return self.execute_sql_query(query)
    
    def get_churn_analysis_by_contract(self):
        """
        SQL: GROUP BY analysis
        """
        query = """
        SELECT 
            Contract,
            COUNT(*) as customer_count,
            SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned,
            ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) as churn_rate,
            AVG(tenure) as avg_tenure,
            AVG(MonthlyCharges) as avg_monthly_charges
        FROM customers
        GROUP BY Contract
        ORDER BY churn_rate DESC
        """
        return self.execute_sql_query(query)
    
    def get_customers_for_modeling(self):
        """
        SQL: Complex query for ML pipeline
        Extract and prepare data for predictive modeling
        """
        query = """
        SELECT 
            tenure,
            MonthlyCharges,
            TotalCharges,
            Contract,
            PaymentMethod,
            InternetService,
            OnlineSecurity,
            OnlineBackup,
            TechSupport,
            StreamingTV,
            StreamingMovies,
            CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END as Churn
        FROM customers
        WHERE TotalCharges IS NOT NULL 
            AND TotalCharges != ''
            AND tenure > 0
        """
        return self.execute_sql_query(query)
    
    def get_revenue_by_segment(self):
        """
        SQL: Advanced analytics query
        """
        query = """
        SELECT 
            CASE 
                WHEN tenure < 12 THEN 'New (0-12 months)'
                WHEN tenure < 36 THEN 'Medium (12-36 months)'
                ELSE 'Long-term (36+ months)'
            END as tenure_segment,
            CASE 
                WHEN MonthlyCharges < 50 THEN 'Low Spender'
                WHEN MonthlyCharges < 80 THEN 'Medium Spender'
                ELSE 'High Spender'
            END as spending_segment,
            COUNT(*) as customer_count,
            SUM(TotalCharges) as total_revenue,
            AVG(TotalCharges) as avg_revenue,
            SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned_count
        FROM customers
        WHERE TotalCharges IS NOT NULL AND TotalCharges != ''
        GROUP BY tenure_segment, spending_segment
        ORDER BY total_revenue DESC
        """
        return self.execute_sql_query(query)
    
    def close_connection(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


# Example Usage in Streamlit
def demonstrate_sql_integration():
    """
    Add this to your Streamlit app to showcase SQL skills
    """
    st.header("🗄️ SQL Data Extraction (JD Requirement)")
    
    # Initialize data loader
    loader = DataLoader()
    
    # Create database from CSV (one-time setup)
    if st.button("Initialize Database from CSV"):
        loader.create_database_from_csv("Churn_pred.csv")
        st.success("✅ Database created successfully!")
    
    # Show SQL queries
    st.subheader("📊 SQL Query Examples")
    
    query_option = st.selectbox(
        "Select SQL Query to Execute:",
        ["Customer Summary", "High-Value Customers", "Churn by Contract", 
         "Revenue by Segment", "Data for Modeling"]
    )
    
    if st.button("Execute Query"):
        if query_option == "Customer Summary":
            result = loader.get_customer_summary()
            st.write("**SQL Query:**")
            st.code("""
SELECT 
    COUNT(*) as total_customers,
    AVG(tenure) as avg_tenure,
    AVG(MonthlyCharges) as avg_monthly_charges,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned_customers
FROM customers
            """)
            st.dataframe(result)
            
        elif query_option == "High-Value Customers":
            result = loader.get_high_value_customers()
            st.write(f"**Found {len(result)} high-value customers**")
            st.dataframe(result.head(10))
            
        elif query_option == "Churn by Contract":
            result = loader.get_churn_analysis_by_contract()
            st.dataframe(result)
            
        elif query_option == "Revenue by Segment":
            result = loader.get_revenue_by_segment()
            st.dataframe(result)
            
        elif query_option == "Data for Modeling":
            result = loader.get_customers_for_modeling()
            st.write(f"**Extracted {len(result)} records for ML modeling**")
            st.dataframe(result.head())
    
    loader.close_connection()