# 📊 Marketing Analytics Dashboard - Enhanced Edition

An advanced Streamlit-based analytics dashboard designed for the **Infosys Data Science Analyst role**, featuring SQL integration, multiple ML models, statistical analysis, and comprehensive business intelligence capabilities.

---

## 🎯 Project Alignment with Infosys JD

This project demonstrates **ALL key requirements** from the Infosys Data Science Analyst Job Description:

### ✅ Technical Skills Covered

| JD Requirement | Implementation |
|---------------|----------------|
| **SQL Coding** | SQLite database with complex queries (GROUP BY, CASE, JOINs) |
| **Python/R** | Python with pandas, scikit-learn, scipy |
| **Machine Learning** | 3 classification models (Logistic Regression, Random Forest, Gradient Boosting) |
| **Statistical Concepts** | Hypothesis testing (t-test, chi-square), effect sizes, correlation analysis |
| **Predictive Modeling** | CLV prediction, Churn prediction with model comparison |
| **Data Visualization** | Matplotlib, Seaborn, interactive Streamlit dashboards |
| **Business Rule Translation** | Model parameters translated to actionable business strategies |
| **CRM Analytics** | Customer segmentation, churn analysis, CLV modeling |

---

## 🚀 Features

1. **🗄️ SQL Data Extraction**
   - SQLite database integration
   - Complex SQL queries (aggregations, window functions, CASE statements)
   - Data extraction for ML pipelines

2. **🔮 Churn Prediction**
   - Multiple classification models
   - Model performance comparison
   - Feature importance analysis
   - Business rule translation

3. **📈 Statistical Analysis**
   - Hypothesis testing (T-test, Chi-square)
   - Effect size calculations
   - Correlation analysis
   - P-value interpretation

4. **💰 CLV Prediction**
   - Linear regression model
   - Customer value forecasting
   - Interactive predictions

5. **🎯 Customer Segmentation**
   - K-Means clustering
   - Segment profiling
   - Marketing strategy recommendations

6. **🛒 Market Basket Analysis**
   - Apriori algorithm
   - Association rules
   - Cross-sell opportunities

7. **🧪 A/B Testing**
   - Statistical significance testing
   - Conversion rate analysis
   - Confidence intervals

---

## 📁 Project Structure

```
Marketing-Analytics-Dashboard/
│
├── data/
│   ├── Churn_pred.csv              # Dataset
│   └── customer_data.db            # SQLite database (auto-generated)
│
├── modules/
│   ├── __init__.py                 # Package initializer
│   ├── sql_loader.py               # SQL integration module
│   ├── churn_predictor.py          # ML prediction module
│   └── statistical_analyzer.py     # Statistical analysis module
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── .gitignore                      # Git ignore rules
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone/Download the Repository

```bash
# If using Git
git clone https://github.com/your-username/Marketing-Analytics-Dashboard.git
cd Marketing-Analytics-Dashboard

# Or download and extract the ZIP file
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare Data

1. Create a `data/` folder in the project root
2. Place your `Churn_pred.csv` file inside the `data/` folder
3. The database will be auto-generated when you run the app

### Step 5: Run the Application

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

---

## 📊 Dataset Requirements

The application expects a CSV file with the following columns:
- `customerID` - Unique customer identifier
- `tenure` - Number of months customer has stayed
- `MonthlyCharges` - Monthly subscription fee
- `TotalCharges` - Total amount charged
- `Churn` - Yes/No indicating if customer churned
- Other categorical features (Contract, InternetService, etc.)

**Sample datasets**: Kaggle Telco Customer Churn dataset works perfectly!

---

## 🎓 Usage Guide

### 1. Dashboard Overview
- View key metrics and KPIs
- Explore customer distributions
- Get high-level insights

### 2. SQL Data Extraction
- Initialize database from CSV
- Execute pre-built SQL queries
- Demonstrates SQL proficiency

### 3. Churn Prediction
- Train multiple ML models
- Compare model performance
- Analyze feature importance
- Get business recommendations

### 4. Statistical Analysis
- Perform hypothesis tests
- Calculate statistical significance
- Visualize distributions

### 5. Other Modules
- CLV Analysis: Predict customer value
- Segmentation: Group similar customers
- Market Basket: Find cross-sell opportunities
- A/B Testing: Compare strategies

---

## 📈 Key Technical Highlights

### Machine Learning Models
```python
- Logistic Regression (baseline)
- Random Forest Classifier (ensemble)
- Gradient Boosting (advanced ensemble)
```

### Statistical Tests
```python
- Independent T-test
- Chi-Square test
- Effect size calculations (Cohen's d, Cramér's V)
- Correlation analysis
```

### SQL Queries
```sql
-- Complex aggregations
-- CASE statements for segmentation
-- GROUP BY for analytics
-- Window functions for advanced metrics
```

---

## 💡 Business Value

This dashboard enables businesses to:

1. **Reduce Churn**: Identify at-risk customers early
2. **Increase Revenue**: Target high-value customers
3. **Optimize Marketing**: Data-driven campaign decisions
4. **Cross-Sell**: Discover product associations
5. **Segment Customers**: Personalized strategies

---

## 🐛 Troubleshooting

### Common Issues

**1. Module Import Errors**
```bash
# Ensure you're in the project root directory
cd Marketing-Analytics-Dashboard

# Verify modules folder exists
ls modules/
```

**2. CSV File Not Found**
```bash
# Check file location
ls data/Churn_pred.csv

# Ensure correct path in app.py (line 37)
csv_path = "data/Churn_pred.csv"
```

**3. Database Errors**
```bash
# Delete and recreate database
rm data/customer_data.db

# Then reinitialize in the app
```

**4. Streamlit Port Already in Use**
```bash
# Use different port
streamlit run app.py --server.port 8502
```

---

## 📚 Learning Resources

- **SQL**: [W3Schools SQL Tutorial](https://www.w3schools.com/sql/)
- **Machine Learning**: [Scikit-learn Documentation](https://scikit-learn.org/)
- **Statistics**: [Khan Academy Statistics](https://www.khanacademy.org/math/statistics-probability)
- **Streamlit**: [Streamlit Docs](https://docs.streamlit.io/)

---

## 🎯 Interview Preparation

When presenting this project:

1. **SQL Skills**: Demonstrate queries in the SQL module
2. **ML Understanding**: Explain model selection rationale
3. **Business Acumen**: Discuss how analytics drives decisions
4. **Statistical Knowledge**: Explain hypothesis testing results
5. **End-to-End Thinking**: From data extraction to deployment

---

## 📝 Future Enhancements

- [ ] Deploy on Streamlit Cloud
- [ ] Add more ML models (XGBoost, Neural Networks)
- [ ] Implement real-time predictions via API
- [ ] Add automated reporting
- [ ] Integrate with Tableau/Power BI
- [ ] Add PySpark for big data processing

---

## 👤 Author

**[Your Name]**
- LinkedIn: [Your Profile]
- GitHub: [Your Username]
- Email: [Your Email]

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- Dataset: [Kaggle Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- Built for: Infosys Data Science Analyst Position
- Technologies: Python, Streamlit, Scikit-learn, SQL

---

## ⭐ Show Your Support

If this project helped you, please give it a ⭐ on GitHub!

**Ready for Infosys Data Science Analyst interviews!** 🚀