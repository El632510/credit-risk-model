Credit Risk Probability Model for Bati Bank
Credit Scoring Business Understanding
1. Basel II Accord and Model Interpretability
Because Bati Bank is a regulated financial institution, our model must comply with the Basel II Capital Accord
. This regulation requires that our risk measurements are well-documented and interpretable
. We cannot use "black-box" models without explanation; we must be able to justify credit decisions to regulators and the bank's leadership to ensure institutional stability
.
2. Necessity and Risks of a Proxy Variable
Why a Proxy is Necessary: The provided dataset from the eCommerce platform does not contain a direct label for "default"
. Therefore, we must engineer a proxy target variable using customer Recency, Frequency, and Monetary (RFM) patterns
.
Business Risks: Using a proxy introduces the risk that our definition of "high-risk" (bad) or "low-risk" (good) is an assumption, not ground truth
. If the proxy does not accurately reflect actual repayment behavior, the bank could lose money by approving bad loans or lose revenue by rejecting good customers
.
3. Model Trade-offs in a Regulated Context
We face a choice between two main modeling approaches
:
Simple/Interpretable (e.g., Logistic Regression): These models are highly favored by regulators because they are easy to explain and use standard techniques like Weight of Evidence (WoE)
.
High-Performance (e.g., Gradient Boosting): These provide state-of-the-art accuracy but are more complex
.
The Trade-off: While complex models might be more accurate, they require Explainable AI (XAI) to remain compliant with banking regulations
. For Bati Bank, transparency is as important as accuracy
.
