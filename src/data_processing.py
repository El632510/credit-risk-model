import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin

# ── Feature lists ──────────────────────────────────────────────────
NUMERIC_FEATURES = ['Total_Amount','Average_Amount','Transaction_Count',
                    'Std_Amount','PricingStrategy','Transaction_Hour',
                    'Transaction_Day','Transaction_Month','Transaction_Year']
CATEGORICAL_FEATURES = ['CurrencyCode','CountryCode']

# ── Step 1: Aggregate ──────────────────────────────────────────────
class DataAggregator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        agg = X.groupby('CustomerId').agg(
            Total_Amount      =('Amount','sum'),
            Average_Amount    =('Amount','mean'),
            Transaction_Count =('TransactionId','count'),
            Std_Amount        =('Amount','std'),
            Last_Transaction  =('TransactionStartTime','max'),
            CurrencyCode      =('CurrencyCode','first'),
            CountryCode       =('CountryCode','first'),
            PricingStrategy   =('PricingStrategy','first')
        ).reset_index()
        agg['Std_Amount'] = agg['Std_Amount'].fillna(0)
        return agg

# ── Step 2: Time features ──────────────────────────────────────────
class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X['Last_Transaction'] = pd.to_datetime(X['Last_Transaction'])
        X['Transaction_Hour']  = X['Last_Transaction'].dt.hour
        X['Transaction_Day']   = X['Last_Transaction'].dt.day
        X['Transaction_Month'] = X['Last_Transaction'].dt.month
        X['Transaction_Year']  = X['Last_Transaction'].dt.year
        return X.drop(columns=['Last_Transaction'])

# ── Step 3-5: Impute + Scale + Encode ─────────────────────────────
def build_preprocessor():
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer([
        ('num', numeric_pipeline,     NUMERIC_FEATURES),
        ('cat', categorical_pipeline, CATEGORICAL_FEATURES)
    ], remainder='drop')

# ── Full pipeline to DataFrame ─────────────────────────────────────
def pipeline_to_dataframe(raw_df):
    agg_df   = DataAggregator().fit_transform(raw_df)
    timed_df = TimeFeatureExtractor().fit_transform(agg_df)
    
    preprocessor = build_preprocessor()
    X_array = preprocessor.fit_transform(timed_df)
    
    ohe      = preprocessor.named_transformers_['cat']['onehot']
    cat_cols = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_cols = NUMERIC_FEATURES + cat_cols
    
    result_df = pd.DataFrame(X_array, columns=all_cols)
    result_df.insert(0, 'CustomerId', agg_df['CustomerId'].values)
    return result_df

# ── Task 4: RFM ────────────────────────────────────────────────────
def build_rfm(raw_df):
    raw_df = raw_df.copy()
    raw_df['TransactionStartTime'] = pd.to_datetime(raw_df['TransactionStartTime'])
    snapshot_date = raw_df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    rfm = raw_df.groupby('CustomerId').agg(
        Recency   =('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
        Frequency =('TransactionId','count'),
        Monetary  =('Amount','sum')
    ).reset_index()
    return rfm

def assign_high_risk(rfm_df, n_clusters=3, random_state=42):
    scaler     = MinMaxScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['Recency','Frequency','Monetary']])
    kmeans     = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    rfm_df     = rfm_df.copy()
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
    cluster_summary   = rfm_df.groupby('Cluster')[['Frequency','Monetary']].mean()
    high_risk_cluster = (cluster_summary['Frequency'] + cluster_summary['Monetary']).idxmin()
    print("Cluster summary:\n", cluster_summary)
    print(f"\nHigh-risk cluster: {high_risk_cluster}")
    rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster).astype(int)
    return rfm_df[['CustomerId','is_high_risk']]

def get_processed_dataset_with_target(raw_df):
    processed_df = pipeline_to_dataframe(raw_df)
    rfm_df       = build_rfm(raw_df)
    target_df    = assign_high_risk(rfm_df)
    final_df     = processed_df.merge(target_df, on='CustomerId', how='left')
    print(f"\nShape: {final_df.shape}")
    print(f"High-risk: {final_df['is_high_risk'].sum()} / {len(final_df)}")
    return final_df
def apply_woe_transformation(raw_df):
    """
    Applies WoE transformation on aggregated features.
    
    """
    try:
        from xverse.transformer import WOE
    except ImportError:
        print("Run: pip install xverse")
        return None, None

    # Get aggregated data (before scaling)
    agg_df   = DataAggregator().fit_transform(raw_df)
    timed_df = TimeFeatureExtractor().fit_transform(agg_df)

    # Get target
    rfm_df    = build_rfm(raw_df)
    target_df = assign_high_risk(rfm_df)
    merged    = timed_df.merge(target_df, on="CustomerId", how="left")

    # Features for WoE (numeric only, no CustomerId)
    features = NUMERIC_FEATURES
    X = merged[features]
    y = merged["is_high_risk"]

    # Fit WoE
    woe = WOE()
    woe.fit(X, y)
    X_woe = woe.transform(X)

    # Show IV scores
    iv_df = woe.iv_df.sort_values("IV", ascending=False).reset_index(drop=True)
    print("\n── Information Value ──────────────────")
    print(iv_df[["Variable", "IV"]].to_string(index=False))
    print("\nIV Guide: <0.02 useless | 0.1-0.3 medium | 0.3-0.5 strong")

    return woe, iv_df