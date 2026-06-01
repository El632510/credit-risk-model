from pydantic import BaseModel

class CustomerFeatures(BaseModel):
    Total_Amount      : float
    Average_Amount    : float
    Transaction_Count : int
    Std_Amount        : float
    PricingStrategy   : float
    Transaction_Hour  : int
    Transaction_Day   : int
    Transaction_Month : int
    Transaction_Year  : int
    CurrencyCode      : str
    CountryCode       : int

class PredictionResponse(BaseModel):
    customer_id      : str
    risk_probability : float
    risk_label       : str   # "High Risk" or "Low Risk"