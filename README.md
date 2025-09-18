# Card Fraud Detection Analysis

This project analyzes a dataset of credit card transactions to build a machine learning model that detects fraudulent activities. The goal is to identify patterns in fraudulent transactions and help prevent them in the future.

### Dataset Schema

This dataset contains transactional data for a pop-up luxury cosmetics company. The table below provides a detailed breakdown of each column, including its data type, a brief description, example values, and information on null values.

| Column Name | Data Type | Description | Example Values | Null Values |
| :--- | :--- | :--- | :--- | :--- |
| `Transaction_ID` | String | Unique identifier for each transaction (UUID format) | `702bdd9b-9c93-41e3-9dbb-a849b2422080` | No |
| `Customer_ID` | String | Unique customer identifier (UUID format) | `119dca0b-8554-4b2d-9bec-e964eaf6af97` | No |
| `Transaction_Date` | Date | Date when the transaction occurred | `2025-07-27` | No |
| `Transaction_Time` | Time | Exact time of transaction (24-hour format) | `04:04:15` | No |
| `Customer_Age` | Integer | Age of the customer (18-65 years) | `56, 46` | Yes (~5%) |
| `Customer_Loyalty_Tier` | Categorical | Customer loyalty program tier | `Bronze, Silver, Gold, Platinum, VIP` | Yes (~5%) |
| `Location` | Categorical | City where the pop-up event occurred | `New York, Paris, Tokyo, Dubai` | No |
| `Store_ID` | Categorical | Unique identifier for each pop-up store | `FLAGSHIP-LA, BOUTIQUE-NYC` | No |
| `Product_SKU` | Categorical | Limited-edition product identifier | `AURORA-LIP-01, CELESTE-EYE-05` | No |
| `Product_Category` | Categorical | Type of cosmetic product purchased | `Lipstick, Foundation, Mascara` | No |
| `Purchase_Amount` | Float | Transaction value in USD ($50-$300 range) | `158.24, 86.03` | No |
| `Payment_Method` | Categorical | Method used for payment | `Credit Card, Debit Card, Mobile Payment, Gift Card` | Yes (~5%) |
| `Device_Type` | Categorical | Device used for the transaction | `Mobile, Desktop, Tablet, Laptop` | No |
| `IP_Address` | String | IP address of the transaction | `239.249.58.237, 84.49.227.90` | No |
| `Fraud_Flag` | Binary | **Target variable** (0 = No Fraud, 1 = Fraud) | `0 (97%), 1 (3%)` | No |
| `Footfall_Count` | Integer | Daily visitor count at the pop-up event | `333, 406` | No |

## Approach
My analysis follows a standard machine learning workflow, focusing on a **classification algorithm** to distinguish between fraudulent and legitimate transactions.

### **1. Choosing the Right Algorithm**
I chose the **Random Forest Classifier** because it's a powerful and easy-to-configure algorithm that excels at identifying important features in complex datasets. It allowed me to highlight the variables most indicative of fraudulent activity without extensive parameter tuning.

### **2. Addressing Data Imbalance with SMOTE**
A key challenge in this project was the **extreme class imbalance**; fraudulent transactions are very rare compared to legitimate ones. To ensure the model could effectively learn the patterns of fraud and not simply predict the majority class, I used **SMOTE (Synthetic Minority Over-sampling Technique)**. This technique created synthetic examples of fraudulent transactions, balancing the dataset and leading to more accurate results.

### **3. Evaluation**
After training the model on the balanced data, I evaluated its performance using key metrics like the **Classification Report**, **Confusion Matrix**, and **ROC-AUC Score** to ensure the model's reliability in a real-world scenario.

## Result
![diagram_result](../assets/Figure_1.png)

![diagram_result](../assets/Figure_11.png)

![diagram_result](..assets/Figure_12.png)

![diagram_result](../assets/Figure_111.png)

![diagram_result](../assets/Figure_13.png)

![diagram_result](../assets/Figure_14.png)

As you can see the most frequent frauds happen on Wednesday or Sunday on 4AM in Milans store branches by people that age are higher 60+. The amount of the transaction is 50-60$. This numbers are most likely the cause of the fraudulent activity.

--- 
