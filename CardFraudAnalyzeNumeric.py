import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

# Load the dataset
try:
    CardFraudDataset = pd.read_csv('luxury_cosmetics_fraud_analysis_2025.csv')
    CardFraudDatasetF = pd.DataFrame(data=CardFraudDataset)
except FileNotFoundError:
    print("Error: The file 'luxury_cosmetics_fraud_analysis_2025.csv' was not found. Please upload the file and try again.")
    exit()

# Filling missing null data
CardFraudDatasetF['Customer_Age'] = CardFraudDatasetF['Customer_Age'].fillna(CardFraudDatasetF['Customer_Age'].mean())
CardFraudDatasetF['Customer_Loyalty_Tier'] = CardFraudDatasetF['Customer_Loyalty_Tier'].fillna('Absent')
CardFraudDatasetF['Payment_Method'] = CardFraudDatasetF['Payment_Method'].fillna('Other')

# Format data in datetime
CardFraudDatasetF['Transaction_Date'] = pd.to_datetime(CardFraudDatasetF['Transaction_Date'], errors='coerce')
CardFraudDatasetF['Day_of_Week'] = CardFraudDatasetF['Transaction_Date'].dt.dayofweek
CardFraudDatasetF['Transaction_Time'] = pd.to_datetime(
    CardFraudDatasetF['Transaction_Time'], format='%H:%M:%S', errors='coerce')
CardFraudDatasetF['Hour_of_Day'] = CardFraudDatasetF['Transaction_Time'].dt.hour

# Define feature and target
categorical_cols = ['Customer_Loyalty_Tier', 'Location', 'Store_ID',
                    'Product_SKU', 'Product_Category', 'Payment_Method', 'Device_Type']
CardFraudDatasetF = pd.get_dummies(CardFraudDatasetF, columns=categorical_cols, drop_first=True)
columns_to_drop = ['Transaction_ID', 'Customer_ID', 'IP_Address', 'Transaction_Date', 'Transaction_Time']
X = CardFraudDatasetF.drop(columns_to_drop + ['Fraud_Flag'], axis=1)
y = CardFraudDatasetF['Fraud_Flag']

# Spliting Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Applying SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Training and Evaluation
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    class_weight='balanced'
)
rf.fit(X_train_res, y_train_res)

# Predictions
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]

# Result
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)[:15]
plt.figure(figsize=(10, 6))
importances.plot(kind='barh')
plt.title("Top 15 Feature Importances (Random Forest)")
plt.show()

# --- START OF UPDATED ANALYSIS CODE ---

# Filter the preprocessed DataFrame to include only fraudulent transactions
fraudulent_transactions = CardFraudDatasetF[CardFraudDatasetF['Fraud_Flag'] == 1].copy()

# A list of numeric features for analysis
numeric_features = ['Footfall_Count', 'Purchase_Amount', 'Customer_Age']

# Analyze continuous numeric features with standard histograms and statistics
print("\n--- Descriptive Statistics for Fraudulent Transactions on Key Numeric Features ---")
for feature in numeric_features:
    if feature in fraudulent_transactions.columns:
        print(f"\nAnalysis for: {feature}")
        print(fraudulent_transactions[feature].describe())
    else:
        print(f"Warning: '{feature}' not found in the original DataFrame.")

print("\n--- Generating Histograms for Continuous Numeric Values ---")
for feature in numeric_features:
    if feature in fraudulent_transactions.columns:
        plt.figure(figsize=(8, 5))
        plt.hist(fraudulent_transactions[feature], bins=20, edgecolor='black', color='skyblue')
        plt.title(f'Distribution of {feature} in Fraudulent Transactions')
        plt.xlabel(feature)
        plt.ylabel('Number of Transactions')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

# Analyze discrete numeric features (Hour_of_Day and Day_of_Week) with bar plots
print("\n--- Frequency Analysis for Discrete Time-Based Features ---")

# Frequency of fraud by Hour_of_Day
fraud_by_hour = fraudulent_transactions['Hour_of_Day'].value_counts().sort_index()
print("\nFraudulent Transactions by Hour of the Day:")
print(fraud_by_hour)

plt.figure(figsize=(10, 6))
fraud_by_hour.plot(kind='bar', color='salmon', edgecolor='black')
plt.title('Fraudulent Transactions by Hour of the Day')
plt.xlabel('Hour of the Day (24-hour format)')
plt.ylabel('Number of Fraudulent Transactions')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Frequency of fraud by Day_of_Week
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
fraudulent_transactions['Day_of_Week_Name'] = fraudulent_transactions['Day_of_Week'].map(dict(enumerate(day_names)))
fraud_by_day = fraudulent_transactions['Day_of_Week_Name'].value_counts()[day_names]
print("\nFraudulent Transactions by Day of the Week:")
print(fraud_by_day)

plt.figure(figsize=(10, 6))
fraud_by_day.plot(kind='bar', color='lightgreen', edgecolor='black')
plt.title('Fraudulent Transactions by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Fraudulent Transactions')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()