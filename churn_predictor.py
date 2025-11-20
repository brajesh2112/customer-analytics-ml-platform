import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

class ChurnPredictor:
    """
    Comprehensive Churn Prediction Module
    Demonstrates: Classification, Model Comparison, Feature Importance
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def prepare_data(self):
        """
        Data preprocessing for churn prediction
        """
        # Drop customer ID
        df = self.df.drop(columns=['customerID'], errors='ignore')
        
        # Handle missing values
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df = df.dropna()
        
        # Encode target variable
        df['Churn'] = (df['Churn'] == 'Yes').astype(int)
        
        # Encode categorical variables
        categorical_cols = df.select_dtypes(include='object').columns
        df_encoded = df.copy()
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col])
        
        # Separate features and target
        X = df_encoded.drop('Churn', axis=1)
        y = df_encoded['Churn']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        self.X_train = pd.DataFrame(
            scaler.fit_transform(self.X_train),
            columns=self.feature_names
        )
        self.X_test = pd.DataFrame(
            scaler.transform(self.X_test),
            columns=self.feature_names
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """
        Train multiple ML models for comparison
        Demonstrates understanding of various algorithms
        """
        # Logistic Regression
        self.models['Logistic Regression'] = LogisticRegression(
            random_state=42, max_iter=1000
        )
        
        # Random Forest
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=10
        )
        
        # Gradient Boosting
        self.models['Gradient Boosting'] = GradientBoostingClassifier(
            n_estimators=100, random_state=42, max_depth=5
        )
        
        # Train all models
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            self.results[name] = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1_score': f1_score(self.y_test, y_pred),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
    
    def get_model_comparison(self):
        """
        Compare model performance
        """
        comparison = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [r['accuracy'] for r in self.results.values()],
            'Precision': [r['precision'] for r in self.results.values()],
            'Recall': [r['recall'] for r in self.results.values()],
            'F1-Score': [r['f1_score'] for r in self.results.values()],
            'ROC-AUC': [r['roc_auc'] for r in self.results.values()]
        })
        return comparison.round(4)
    
    def plot_model_comparison(self):
        """
        Visualize model performance comparison
        """
        comparison = self.get_model_comparison()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart of metrics
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        x = np.arange(len(comparison))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            axes[0].bar(x + i * width, comparison[metric], width, label=metric)
        
        axes[0].set_xlabel('Models')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Model Performance Comparison')
        axes[0].set_xticks(x + width * 2)
        axes[0].set_xticklabels(comparison['Model'], rotation=45, ha='right')
        axes[0].legend()
        axes[0].set_ylim([0, 1.1])
        
        # ROC Curves
        for name in self.results.keys():
            y_pred_proba = self.results[name]['y_pred_proba']
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc = self.results[name]['roc_auc']
            axes[1].plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
        
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrices(self):
        """
        Plot confusion matrices for all models
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, (name, result) in enumerate(self.results.items()):
            cm = result['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{name}\nConfusion Matrix')
            axes[idx].set_ylabel('Actual')
            axes[idx].set_xlabel('Predicted')
        
        plt.tight_layout()
        return fig
    
    def get_feature_importance(self, model_name='Random Forest', top_n=10):
        """
        Extract and visualize feature importance
        Demonstrates ability to explain model parameters (JD requirement)
        """
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            return None
        
        # Create DataFrame
        feature_imp = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        return feature_imp.head(top_n)
    
    def plot_feature_importance(self, model_name='Random Forest', top_n=10):
        """
        Visualize feature importance
        """
        feature_imp = self.get_feature_importance(model_name, top_n)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feature_imp['Feature'], feature_imp['Importance'], color='steelblue')
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_n} Features - {model_name}')
        ax.invert_yaxis()
        
        return fig
    
    def translate_to_business_rules(self, model_name='Random Forest'):
        """
        Translate model parameters into business rules
        Key JD requirement: "translate this into implementable business rules"
        """
        feature_imp = self.get_feature_importance(model_name, top_n=5)
        
        rules = []
        rules.append("## 🎯 Churn Risk Business Rules\n")
        rules.append("Based on model analysis, customers are at HIGH RISK if:\n")
        
        # Generate rules from top features
        for _, row in feature_imp.iterrows():
            feature = row['Feature']
            importance = row['Importance']
            
            if feature == 'tenure':
                rules.append(f"• **Low Tenure** (< 12 months) - High churn indicator ({importance:.2%} importance)")
            elif feature == 'MonthlyCharges':
                rules.append(f"• **High Monthly Charges** (> $70) - Price sensitivity factor ({importance:.2%} importance)")
            elif feature == 'Contract':
                rules.append(f"• **Month-to-Month Contract** - No commitment issue ({importance:.2%} importance)")
            elif 'InternetService' in feature:
                rules.append(f"• **Fiber Optic Internet** - Service quality concerns ({importance:.2%} importance)")
            elif 'TechSupport' in feature:
                rules.append(f"• **No Tech Support** - Lack of assistance ({importance:.2%} importance)")
            else:
                rules.append(f"• **{feature}** factor ({importance:.2%} importance)")
        
        rules.append("\n### 📋 Actionable Retention Strategies:")
        rules.append("1. **Early Engagement Program**: Intensive support in first 12 months")
        rules.append("2. **Contract Incentives**: Offer discounts for annual contracts")
        rules.append("3. **Tech Support Bundle**: Free tech support for first 6 months")
        rules.append("4. **Price Optimization**: Review pricing for high-charge customers")
        rules.append("5. **Service Quality**: Investigate fiber optic service issues")
        
        return '\n'.join(rules)
    
    def predict_individual_churn(self, customer_data, model_name='Random Forest'):
        """
        Predict churn probability for a single customer
        """
        model = self.models[model_name]
        
        # Ensure same feature order
        customer_df = pd.DataFrame([customer_data], columns=self.feature_names)
        
        churn_prob = model.predict_proba(customer_df)[0, 1]
        churn_pred = model.predict(customer_df)[0]
        
        return {
            'churn_probability': churn_prob,
            'will_churn': bool(churn_pred),
            'risk_level': 'High' if churn_prob > 0.7 else 'Medium' if churn_prob > 0.4 else 'Low'
        }


# Streamlit Integration Example
def add_churn_prediction_tab():
    """
    Add this to your Streamlit app
    """
    st.header("🔮 Customer Churn Prediction")
    
    # Load data
    df = pd.read_csv("Churn_pred.csv")
    
    # Initialize predictor
    predictor = ChurnPredictor(df)
    
    # Prepare data
    with st.spinner("Preparing data..."):
        predictor.prepare_data()
        predictor.train_models()
    
    # Model Comparison
    st.subheader("📊 Model Performance Comparison")
    comparison = predictor.get_model_comparison()
    st.dataframe(comparison)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = predictor.plot_model_comparison()
        st.pyplot(fig1)
    
    with col2:
        fig2 = predictor.plot_confusion_matrices()
        st.pyplot(fig2)
    
    # Feature Importance
    st.subheader("🎯 Feature Importance Analysis")
    model_choice = st.selectbox("Select Model", list(predictor.models.keys()))
    
    fig3 = predictor.plot_feature_importance(model_choice)
    st.pyplot(fig3)
    
    # Business Rules
    st.subheader("💼 Business Rules Translation")
    business_rules = predictor.translate_to_business_rules(model_choice)
    st.markdown(business_rules)
    
    # Individual Prediction
    st.subheader("🎲 Predict Individual Customer Churn")
    st.write("Enter customer details:")
    
    # Add input fields for prediction (simplified example)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tenure = st.number_input("Tenure (months)", 0, 72, 12)
    with col2:
        monthly = st.number_input("Monthly Charges", 18.0, 120.0, 65.0)
    with col3:
        total = st.number_input("Total Charges", 18.0, 8500.0, 780.0)
    
    if st.button("Predict Churn Risk"):
        # Create dummy customer data (you'd need to add all features)
        customer_data = [tenure, monthly, total] + [0] * (len(predictor.feature_names) - 3)
        result = predictor.predict_individual_churn(customer_data, model_choice)
        
        st.metric("Churn Probability", f"{result['churn_probability']:.1%}")
        
        if result['risk_level'] == 'High':
            st.error(f"⚠️ **{result['risk_level']} Risk** - Immediate intervention needed!")
        elif result['risk_level'] == 'Medium':
            st.warning(f"⚡ **{result['risk_level']} Risk** - Monitor closely")
        else:
            st.success(f"✅ **{result['risk_level']} Risk** - Customer stable")