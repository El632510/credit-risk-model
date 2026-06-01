import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from xverse.transformer import WOE
from sklearn.base import BaseEstimator, TransformerMixin

class DataAggregator(BaseEstimator, TransformerMixin):
    """Aggregates transaction data to customer level before modeling."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Ensure timestamp is datetime for potential extraction if needed
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        
        # Aggregate to customer level
        agg = X.groupby('CustomerId').agg(
            Total_Amount=('Amount', 'sum'),
            Average_Amount=('Amount', 'mean'),
            Transaction_Count=('TransactionId', 'count'),
            Std_Amount=('Amount', 'std'),
            # Take the first occurrence of static categorical info
            CurrencyCode=('CurrencyCode', 'first'),
            CountryCode=('CountryCode', 'first'),
            PricingStrategy=('PricingStrategy', 'first')
        ).reset_index()
        
        return agg.fillna(0)

def get_full_pipeline():
    """
    Returns a unified pipeline that aggregates raw transaction logs
    into a customer-level, model-ready feature set.
    """
    # 1. Define feature sets based on the aggregated schema
    numeric_features = ['Total_Amount', 'Average_Amount', 'Transaction_Count', 'Std_Amount', 'PricingStrategy']
    categorical_features = ['CurrencyCode', 'CountryCode']

    # 2. Preprocessing steps
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 3. Final Pipeline: Aggregate -> Preprocess -> WoE
    # Note: WOE() requires y at fit time.
    pipeline = Pipeline(steps=[
        ('aggregator', DataAggregator()),
        ('preprocessor', preprocessor),
        ('woe', WOE())
    ])
    
    return pipeline