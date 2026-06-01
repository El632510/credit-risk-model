import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import pandas as pd
import joblib

# Import your pipeline functions
from src.data_processing import get_full_pipeline

def train_and_track():
    # 1. Load Data
    data = pd.read_csv('../data/raw/data.csv')
    
    # 2. Prepare Features and Target
    # Note: Ensure you have performed Task 4 to add 'is_high_risk'
    X = data.drop('is_high_risk', axis=1)
    y = data['is_high_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 3. Initialize Pipeline
    # Ensure these lists match your column names
    numeric_features = ['Amount', 'Value', 'PricingStrategy'] 
    categorical_features = ['CurrencyCode', 'CountryCode', 'ProviderId', 
                            'ProductId', 'ProductCategory', 'ChannelId']
    
    pipeline = get_full_pipeline(numeric_features, categorical_features)
    
    # 4. MLflow Experiment Tracking
    mlflow.set_experiment("Credit_Risk_Model")
    
    with mlflow.start_run():
        # Train a baseline model
        model = RandomForestClassifier(random_state=42)
        
        # Hyperparameter tuning setup
        param_grid = {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [None, 10, 20]
        }
        
        # Grid Search
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc')
        grid_search.fit(X_train, y_train)
        
        # Log params and metrics
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, grid_search.predict(X_test)))
        
        # Save model to registry
        mlflow.sklearn.log_model(grid_search.best_estimator_, "best_model")
        
        print(f"Best Model AUC: {grid_search.best_score_}")

if __name__ == "__main__":
    train_and_track()