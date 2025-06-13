# Credit Card Default Prediction using Classification Techniques

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, accuracy_score, recall_score, precision_score, make_scorer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap
import warnings
warnings.filterwarnings("ignore")

# Step 2: Load the Dataset
df = pd.read_csv('C:/Users/jayde/Downloads/Datasets_final/train_dataset_final1.csv')

# Step 3: Initial Data Checks
print(df.head())
print(df.info())
print(df['next_month_default'].value_counts())

# Step 4: Check for Missing Values
print("Missing values:")
print(df.isnull().sum())

# Step 5: Exploratory Data Analysis (EDA)
sns.histplot(df['LIMIT_BAL'], bins=50, kde=True)
plt.title('Distribution of Credit Limit')
plt.show()

sns.countplot(data=df, x='next_month_default')
plt.title('Default Distribution')
plt.show()

corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Step 6: Feature Engineering
# Avoid divide-by-zero errors
df['CREDIT_UTILIZATION'] = df.apply(lambda row: row['AVG_Bill_amt'] / row['LIMIT_BAL'] if row['LIMIT_BAL'] > 0 else 0, axis=1)

# Delinquency streak (how often they are late)
pay_cols = ['pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']
df['DELINQUENCY_COUNT'] = (df[pay_cols] >= 1).sum(axis=1)

# Repayment consistency (standard deviation of payments)
pay_amt_cols = ['pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6']
df['PAYMENT_STD'] = df[pay_amt_cols].std(axis=1)

# PAY_TO_BILL_ratio feature
total_pay = df[pay_amt_cols].sum(axis=1)
total_bill = df[['Bill_amt1','Bill_amt2','Bill_amt3','Bill_amt4','Bill_amt5','Bill_amt6']].sum(axis=1)
df['PAY_TO_BILL_ratio'] = total_pay / total_bill.replace(0, np.nan)

# Step 6.1: Handle Missing Values
df = df.dropna()

# Visualize Delinquency Count after feature engineering
sns.boxplot(data=df, x='next_month_default', y='DELINQUENCY_COUNT')
plt.title('Delinquency Count vs Default')
plt.show()

# Step 7: Prepare Data for Modeling
X = df.drop(['Customer_ID', 'next_month_default'], axis=1)
y = df['next_month_default']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle Class Imbalance
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_scaled, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 8: Train and Evaluate Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    f2_score = 5 * (precision_score(y_test, y_pred) * recall_score(y_test, y_pred)) / (4 * precision_score(y_test, y_pred) + recall_score(y_test, y_pred))
    print("F2 Score:", f2_score)
    print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

# Step 9: Plot ROC Curves
plt.figure(figsize=(10, 6))
for name, model in models.items():
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid()
plt.show()

# Step 10: Choose best model and tune threshold
best_model = LGBMClassifier()
best_model.fit(X_train, y_train)
y_val_prob = best_model.predict_proba(X_test)[:, 1]

# Find best threshold using F2 Score
thresholds = np.arange(0.1, 0.9, 0.01)
best_thresh = 0.5
best_f2 = 0
for thresh in thresholds:
    preds = (y_val_prob >= thresh).astype(int)
    f2 = 5 * (precision_score(y_test, preds) * recall_score(y_test, preds)) / (4 * precision_score(y_test, preds) + recall_score(y_test, preds))
    if f2 > best_f2:
        best_f2 = f2
        best_thresh = thresh
print(f"Best Threshold (F2 optimized): {best_thresh:.2f}, F2 Score: {best_f2:.4f}")

# Step 10.1: Model Explainability using SHAP
explainer = shap.Explainer(best_model, X_train, feature_names=X.columns)
shap_values = explainer(X_test, check_additivity=False)

# SHAP summary plot
shap.summary_plot(shap_values, X_test, feature_names=X.columns.tolist())

# SHAP bar plot for global importance
shap.plots.bar(shap_values)

# SHAP dependence plot for key feature
shap.dependence_plot("DELINQUENCY_COUNT", shap_values.values, X_test, feature_names=X.columns.tolist())

# SHAP force plot for first prediction
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values.values[0], features=X_test[0], feature_names=X.columns.tolist())

# Step 11: Predict on Unlabeled Data (Validation Set)
validation_df = pd.read_csv('c:/Users/jayde/Downloads/Datasets_final/validate_dataset_final.csv')

# Step 12: Apply same feature engineering
validation_df['CREDIT_UTILIZATION'] = validation_df.apply(lambda row: row['AVG_Bill_amt'] / row['LIMIT_BAL'] if row['LIMIT_BAL'] > 0 else 0, axis=1)
validation_df['DELINQUENCY_COUNT'] = (validation_df[pay_cols] >= 1).sum(axis=1)
validation_df['PAYMENT_STD'] = validation_df[pay_amt_cols].std(axis=1)

total_val_pay = validation_df[pay_amt_cols].sum(axis=1)
total_val_bill = validation_df[['Bill_amt1','Bill_amt2','Bill_amt3','Bill_amt4','Bill_amt5','Bill_amt6']].sum(axis=1)
validation_df['PAY_TO_BILL_ratio'] = total_val_pay / total_val_bill.replace(0, np.nan)

# Step 13: Handle missing values
validation_df = validation_df.dropna()

# Step 14: Standardize features
X_val = validation_df.drop(['Customer_ID'], axis=1)
X_val_scaled = scaler.transform(X_val)

# Step 15: Predict default probabilities and classes using tuned threshold
val_preds_proba = best_model.predict_proba(X_val_scaled)[:, 1]
val_preds = (val_preds_proba >= best_thresh).astype(int)

# Step 16: Attach predictions to customer IDs and save
validation_df['Predicted_Default'] = val_preds
validation_df['Default_Probability'] = val_preds_proba

output = validation_df[['Customer_ID', 'Predicted_Default', 'Default_Probability']]

# Save predictions with customer IDs to a CSV file for review or further action
output.to_csv('C:/Users/jayde/Downloads/Datasets_final/predictions_on_unlabeled.csv', index=False)
print("Predictions saved to predictions_on_unlabeled.csv")
