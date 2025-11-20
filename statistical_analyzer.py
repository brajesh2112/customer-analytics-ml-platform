import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

class StatisticalAnalyzer:
    """
    Statistical Analysis Module
    Demonstrates: Hypothesis testing, statistical concepts, A/B testing
    JD Requirement: "Strong knowledge of statistical concepts"
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.alpha = 0.05  # Significance level
    
    def descriptive_statistics(self, feature, group_by=None):
        """
        Comprehensive descriptive statistics
        """
        if group_by:
            stats_df = self.df.groupby(group_by)[feature].describe()
        else:
            stats_df = self.df[feature].describe().to_frame().T
        
        return stats_df
    
    def hypothesis_test_numeric(self, feature, group_column, group1_value, group2_value):
        """
        T-test or Mann-Whitney U test for numeric features
        Tests if two groups have significantly different means
        """
        # Extract groups
        group1 = self.df[self.df[group_column] == group1_value][feature].dropna()
        group2 = self.df[self.df[group_column] == group2_value][feature].dropna()
        
        # Check normality (Shapiro-Wilk test)
        _, p_normal_1 = stats.shapiro(group1.sample(min(5000, len(group1))))
        _, p_normal_2 = stats.shapiro(group2.sample(min(5000, len(group2))))
        
        # Choose appropriate test
        if p_normal_1 > 0.05 and p_normal_2 > 0.05:
            # Both normal: use t-test
            statistic, p_value = ttest_ind(group1, group2)
            test_name = "Independent T-Test"
        else:
            # Non-normal: use Mann-Whitney U test
            statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
            test_name = "Mann-Whitney U Test"
        
        # Calculate effect size (Cohen's d)
        mean1, mean2 = group1.mean(), group2.mean()
        std_pooled = np.sqrt((group1.var() + group2.var()) / 2)
        cohens_d = (mean1 - mean2) / std_pooled if std_pooled != 0 else 0
        
        results = {
            'test_name': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'group1_mean': mean1,
            'group2_mean': mean2,
            'group1_std': group1.std(),
            'group2_std': group2.std(),
            'group1_size': len(group1),
            'group2_size': len(group2),
            'cohens_d': cohens_d,
            'effect_size': self._interpret_effect_size(abs(cohens_d))
        }
        
        return results
    
    def hypothesis_test_categorical(self, feature1, feature2):
        """
        Chi-square test for categorical features
        Tests if two categorical variables are independent
        """
        # Create contingency table
        contingency_table = pd.crosstab(self.df[feature1], self.df[feature2])
        
        # Perform chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Calculate Cramér's V (effect size for chi-square)
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        
        results = {
            'test_name': 'Chi-Square Test of Independence',
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'significant': p_value < self.alpha,
            'cramers_v': cramers_v,
            'effect_size': self._interpret_cramers_v(cramers_v),
            'contingency_table': contingency_table,
            'expected_frequencies': expected
        }
        
        return results
    
    def correlation_analysis(self, features=None):
        """
        Correlation analysis between numerical features
        """
        if features is None:
            # Select all numeric columns
            features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        corr_matrix = self.df[features].corr()
        
        # Find strongest correlations
        corr_pairs = []
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                corr_pairs.append({
                    'feature1': features[i],
                    'feature2': features[j],
                    'correlation': corr_matrix.iloc[i, j],
                    'abs_correlation': abs(corr_matrix.iloc[i, j])
                })
        
        corr_df = pd.DataFrame(corr_pairs).sort_values('abs_correlation', ascending=False)
        
        return corr_matrix, corr_df
    
    def ab_test_analysis(self, metric, variant_column, variant_a, variant_b):
        """
        Comprehensive A/B test statistical analysis
        """
        # Extract groups
        group_a = self.df[self.df[variant_column] == variant_a][metric].dropna()
        group_b = self.df[self.df[variant_column] == variant_b][metric].dropna()
        
        # Perform statistical test
        test_results = self.hypothesis_test_numeric(
            metric, variant_column, variant_a, variant_b
        )
        
        # Calculate confidence intervals
        ci_a = stats.t.interval(
            0.95, len(group_a)-1, 
            loc=group_a.mean(), 
            scale=stats.sem(group_a)
        )
        ci_b = stats.t.interval(
            0.95, len(group_b)-1, 
            loc=group_b.mean(), 
            scale=stats.sem(group_b)
        )
        
        # Calculate relative improvement
        relative_improvement = ((group_b.mean() - group_a.mean()) / group_a.mean() * 100 
                               if group_a.mean() != 0 else 0)
        
        # Power analysis (simplified)
        pooled_std = np.sqrt((group_a.var() + group_b.var()) / 2)
        effect_size = (group_b.mean() - group_a.mean()) / pooled_std if pooled_std != 0 else 0
        
        results = {
            **test_results,
            'variant_a': variant_a,
            'variant_b': variant_b,
            'ci_a_lower': ci_a[0],
            'ci_a_upper': ci_a[1],
            'ci_b_lower': ci_b[0],
            'ci_b_upper': ci_b[1],
            'relative_improvement': relative_improvement,
            'absolute_improvement': group_b.mean() - group_a.mean()
        }
        
        return results
    
    def _interpret_effect_size(self, cohens_d):
        """Interpret Cohen's d effect size"""
        if cohens_d < 0.2:
            return "Negligible"
        elif cohens_d < 0.5:
            return "Small"
        elif cohens_d < 0.8:
            return "Medium"
        else:
            return "Large"
    
    def _interpret_cramers_v(self, cramers_v):
        """Interpret Cramér's V effect size"""
        if cramers_v < 0.1:
            return "Negligible"
        elif cramers_v < 0.3:
            return "Small"
        elif cramers_v < 0.5:
            return "Medium"
        else:
            return "Large"
    
    def plot_hypothesis_test(self, feature, group_column, group1_value, group2_value):
        """
        Visualize hypothesis test results
        """
        group1 = self.df[self.df[group_column] == group1_value][feature].dropna()
        group2 = self.df[self.df[group_column] == group2_value][feature].dropna()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Distribution plot
        axes[0].hist(group1, bins=30, alpha=0.6, label=f'{group1_value}', color='blue', density=True)
        axes[0].hist(group2, bins=30, alpha=0.6, label=f'{group2_value}', color='red', density=True)
        axes[0].axvline(group1.mean(), color='blue', linestyle='--', linewidth=2)
        axes[0].axvline(group2.mean(), color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel(feature)
        axes[0].set_ylabel('Density')
        axes[0].set_title(f'Distribution Comparison: {feature}')
        axes[0].legend()
        
        # Box plot
        data_for_boxplot = pd.DataFrame({
            feature: np.concatenate([group1, group2]),
            group_column: [group1_value]*len(group1) + [group2_value]*len(group2)
        })
        sns.boxplot(data=data_for_boxplot, x=group_column, y=feature, ax=axes[1])
        axes[1].set_title(f'Box Plot Comparison: {feature}')
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(self, features=None):
        """
        Visualize correlation matrix
        """
        corr_matrix, _ = self.correlation_analysis(features)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Feature Correlation Matrix', fontsize=16, pad=20)
        
        return fig


# Streamlit Integration
def add_statistical_analysis_tab():
    """
    Add statistical analysis to Streamlit app
    """
    st.header("📈 Statistical Analysis")
    
    # Load data
    df = pd.read_csv("Churn_pred.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    
    analyzer = StatisticalAnalyzer(df)
    
    # Analysis type selection
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Descriptive Statistics", "Hypothesis Testing (Numeric)", 
         "Hypothesis Testing (Categorical)", "Correlation Analysis", 
         "A/B Test Analysis"]
    )
    
    if analysis_type == "Descriptive Statistics":
        st.subheader("📊 Descriptive Statistics")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature = st.selectbox("Select Feature", numeric_cols)
        
        group_by = st.selectbox("Group By (Optional)", 
                               ['None'] + df.select_dtypes(include='object').columns.tolist())
        
        if group_by == 'None':
            stats_df = analyzer.descriptive_statistics(feature)
        else:
            stats_df = analyzer.descriptive_statistics(feature, group_by)
        
        st.dataframe(stats_df)
    
    elif analysis_type == "Hypothesis Testing (Numeric)":
        st.subheader("🔬 Hypothesis Testing for Numeric Features")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            feature = st.selectbox("Select Numeric Feature", numeric_cols)
            group_column = st.selectbox("Group By", categorical_cols)
        
        with col2:
            unique_values = df[group_column].unique()
            group1 = st.selectbox("Group 1", unique_values)
            group2 = st.selectbox("Group 2", [v for v in unique_values if v != group1])
        
        if st.button("Run Test"):
            results = analyzer.hypothesis_test_numeric(feature, group_column, group1, group2)
            
            st.write(f"**Test Used**: {results['test_name']}")
            st.write(f"**Test Statistic**: {results['statistic']:.4f}")
            st.write(f"**P-value**: {results['p_value']:.4f}")
            
            if results['significant']:
                st.success(f"✅ **Statistically Significant** (p < {analyzer.alpha})")
                st.write("**Conclusion**: There IS a significant difference between the groups.")
            else:
                st.info(f"❌ **Not Significant** (p >= {analyzer.alpha})")
                st.write("**Conclusion**: NO significant difference between the groups.")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"{group1} Mean", f"{results['group1_mean']:.2f}")
            with col2:
                st.metric(f"{group2} Mean", f"{results['group2_mean']:.2f}")
            with col3:
                st.metric("Effect Size", f"{results['cohens_d']:.3f} ({results['effect_size']})")
            
            # Visualization
            fig = analyzer.plot_hypothesis_test(feature, group_column, group1, group2)
            st.pyplot(fig)
    
    elif analysis_type == "Hypothesis Testing (Categorical)":
        st.subheader("🔬 Chi-Square Test for Categorical Features")
        
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            feature1 = st.selectbox("Feature 1", categorical_cols)
        with col2:
            feature2 = st.selectbox("Feature 2", [c for c in categorical_cols if c != feature1])
        
        if st.button("Run Chi-Square Test"):
            results = analyzer.hypothesis_test_categorical(feature1, feature2)
            
            st.write(f"**Chi-Square Statistic**: {results['chi2_statistic']:.4f}")
            st.write(f"**P-value**: {results['p_value']:.4f}")
            st.write(f"**Degrees of Freedom**: {results['degrees_of_freedom']}")
            
            if results['significant']:
                st.success(f"✅ **Statistically Significant** (p < {analyzer.alpha})")
                st.write(f"**Conclusion**: {feature1} and {feature2} ARE associated.")
            else:
                st.info(f"❌ **Not Significant** (p >= {analyzer.alpha})")
                st.write(f"**Conclusion**: {feature1} and {feature2} are INDEPENDENT.")
            
            st.metric("Cramér's V", f"{results['cramers_v']:.3f} ({results['effect_size']})")
            
            st.subheader("📊 Contingency Table")
            st.dataframe(results['contingency_table'])
    
    elif analysis_type == "Correlation Analysis":
        st.subheader("🔗 Correlation Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if st.checkbox("Select specific features", value=False):
            selected_features = st.multiselect("Features", numeric_cols, default=numeric_cols[:5])
        else:
            selected_features = numeric_cols
        
        if st.button("Calculate Correlations"):
            corr_matrix, corr_pairs = analyzer.correlation_analysis(selected_features)
            
            st.subheader("📈 Correlation Matrix")
            fig = analyzer.plot_correlation_heatmap(selected_features)
            st.pyplot(fig)
            
            st.subheader("🔝 Strongest Correlations")
            st.dataframe(corr_pairs.head(10))