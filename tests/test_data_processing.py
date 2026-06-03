import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from data_processing import (  # noqa: E402
    DataAggregator,
    TimeFeatureExtractor,
    build_rfm,
    pipeline_to_dataframe,
)


@pytest.fixture
def sample_df():
    np.random.seed(0)
    return pd.DataFrame({
        "TransactionId": [f"T{i}" for i in range(20)],
        "CustomerId": np.random.choice(["C1", "C2", "C3"], 20),
        "Amount": np.random.uniform(10, 500, 20),
        "TransactionStartTime": pd.date_range("2023-01-01", periods=20, freq="6h"),
        "CurrencyCode": np.random.choice(["UGX", "USD"], 20),
        "CountryCode": np.random.choice([256, 254], 20),
        "PricingStrategy": np.random.choice([0, 1, 2], 20),
    })


def test_aggregator_one_row_per_customer(sample_df):
    agg = DataAggregator().fit_transform(sample_df)
    assert agg.shape[0] == sample_df["CustomerId"].nunique()


def test_time_feature_columns(sample_df):
    agg = DataAggregator().fit_transform(sample_df)
    timed = TimeFeatureExtractor().fit_transform(agg)
    for col in ["Transaction_Hour", "Transaction_Day", "Transaction_Month", "Transaction_Year"]:
        assert col in timed.columns


def test_no_nulls_after_pipeline(sample_df):
    result = pipeline_to_dataframe(sample_df)
    assert result.isnull().sum().sum() == 0


def test_rfm_columns(sample_df):
    rfm = build_rfm(sample_df)
    for col in ["CustomerId", "Recency", "Frequency", "Monetary"]:
        assert col in rfm.columns
