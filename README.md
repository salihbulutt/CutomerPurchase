# CutomerPurchase
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load Data ---
# Assuming your data is in a CSV file named 'customer_purchases.csv'.
# If your file is in a different format (e.g., Excel, JSON), adjust pd.read_csv accordingly.
# For demonstration purposes, I'll create a dummy dataset.
try:
    df = pd.read_csv('customer_purchases.csv')
    print("Dataset loaded successfully from 'customer_purchases.csv'.")
except FileNotFoundError:
    print("'customer_purchases.csv' not found. Creating a dummy dataset for demonstration.")
    data = {
        'CustomerID': np.random.randint(1001, 1050, 1000),
        'PurchaseID': np.arange(10001, 11001),
        'PurchaseDate': pd.to_datetime(pd.date_range(start='2023-01-01', periods=1000, freq='H')) + pd.to_timedelta(np.random.randint(0, 3600, 1000), unit='s'),
        'ProductCategory': np.random.choice(['Electronics', 'Apparel', 'Books', 'Home Goods', 'Food'], 1000),
        'ProductName': np.random.choice([f'Product_{i}' for i in range(1, 50)], 1000),
        'Quantity': np.random.randint(1, 6, 1000),
        'UnitPrice': np.round(np.random.uniform(5.0, 500.0, 1000), 2),
        'PaymentMethod': np.random.choice(['Credit Card', 'Debit Card', 'PayPal'], 1000),
        'Region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], 1000)
    }
    df = pd.DataFrame(data)
    # Calculate PurchaseAmount
    df['PurchaseAmount'] = df['Quantity'] * df['UnitPrice']
    # Add some missing values for demonstration
    df.loc[df.sample(frac=0.02).index, 'PurchaseAmount'] = np.nan
    df.loc[df.sample(frac=0.01).index, 'ProductCategory'] = np.nan


# --- 2. Initial Data Inspection ---
print("\n--- 2.1 Head of the Dataset ---")
print(df.head())

print("\n--- 2.2 Dataset Information (Data Types, Non-Null Counts) ---")
df.info()

print("\n--- 2.3 Descriptive Statistics for Numerical Columns ---")
print(df.describe())

print("\n--- 2.4 Missing Values Check ---")
print(df.isnull().sum())

# --- 3. Data Cleaning and Preparation ---
# Convert 'PurchaseDate' to datetime objects if not already
if not pd.api.types.is_datetime64_any_dtype(df['PurchaseDate']):
    df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce')
    print("\n'PurchaseDate' column converted to datetime.")

# Handle missing values in 'PurchaseAmount' (e.g., fill with median or mean, or drop)
# For this exploration, we'll fill with the median to retain rows.
if df['PurchaseAmount'].isnull().any():
    median_purchase_amount = df['PurchaseAmount'].median()
    df['PurchaseAmount'].fillna(median_purchase_amount, inplace=True)
    print(f"\nMissing 'PurchaseAmount' values filled with median: {median_purchase_amount:.2f}")

# Handle missing values in categorical columns (e.g., fill with 'Unknown' or mode)
for col in ['ProductCategory']:
    if df[col].isnull().any():
        df[col].fillna('Unknown', inplace=True)
        print(f"Missing '{col}' values filled with 'Unknown'.")


# --- 4. Exploratory Data Visualization and Analysis ---

# Set a style for the plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6) # Set default figure size

print("\n--- 4.1 Distribution of Purchase Amounts ---")
plt.figure()
sns.histplot(df['PurchaseAmount'], bins=50, kde=True)
plt.title('Distribution of Purchase Amounts')
plt.xlabel('Purchase Amount ($)')
plt.ylabel('Frequency')
plt.show()

print("\n--- 4.2 Purchase Amount by Product Category ---")
plt.figure()
sns.boxplot(x='ProductCategory', y='PurchaseAmount', data=df)
plt.title('Purchase Amount by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Purchase Amount ($)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\n--- 4.3 Daily Sales Trend ---")
# Aggregate daily sales
df['PurchaseDay'] = df['PurchaseDate'].dt.to_period('D')
daily_sales = df.groupby('PurchaseDay')['PurchaseAmount'].sum().to_timestamp()

plt.figure()
daily_sales.plot(kind='line')
plt.title('Daily Sales Trend')
plt.xlabel('Date')
plt.ylabel('Total Sales ($)')
plt.show()

print("\n--- 4.4 Top 10 Product Categories by Total Sales ---")
category_sales = df.groupby('ProductCategory')['PurchaseAmount'].sum().sort_values(ascending=False)
plt.figure()
sns.barplot(x=category_sales.index[:10], y=category_sales.values[:10], palette='viridis')
plt.title('Top 10 Product Categories by Total Sales')
plt.xlabel('Product Category')
plt.ylabel('Total Sales ($)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\n--- 4.5 Top 10 Customers by Total Purchase Amount ---")
customer_sales = df.groupby('CustomerID')['PurchaseAmount'].sum().sort_values(ascending=False)
plt.figure()
sns.barplot(x=customer_sales.index[:10], y=customer_sales.values[:10], palette='crest')
plt.title('Top 10 Customers by Total Purchase Amount')
plt.xlabel('Customer ID')
plt.ylabel('Total Purchase Amount ($)')
plt.tight_layout()
plt.show()

print("\n--- 4.6 Distribution of Quantity per Purchase ---")
plt.figure()
sns.countplot(x='Quantity', data=df, palette='magma')
plt.title('Distribution of Quantity per Purchase')
plt.xlabel('Quantity')
plt.ylabel('Number of Purchases')
plt.show()

print("\n--- 4.7 Purchases by Payment Method ---")
payment_method_counts = df['PaymentMethod'].value_counts()
plt.figure()
sns.barplot(x=payment_method_counts.index, y=payment_method_counts.values, palette='cubehelix')
plt.title('Number of Purchases by Payment Method')
plt.xlabel('Payment Method')
plt.ylabel('Number of Purchases')
plt.tight_layout()
plt.show()

print("\n--- 4.8 Purchases by Region ---")
region_counts = df['Region'].value_counts()
plt.figure()
sns.barplot(x=region_counts.index, y=region_counts.values, palette='plasma')
plt.title('Number of Purchases by Region')
plt.xlabel('Region')
plt.ylabel('Number of Purchases')
plt.tight_layout()
plt.show()
print("\n--- 4.9 Correlation Heatmap of Numerical Features (if multiple numerical features exist) ---")
# For this dummy dataset, we only have Quantity, UnitPrice, PurchaseAmount.
# Let's see the correlation between Quantity, UnitPrice, and PurchaseAmount (which is derived)
numerical_df = df[['Quantity', 'UnitPrice', 'PurchaseAmount']]
correlation_matrix = numerical_df.corr()
plt.figure()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()
print("\nData exploration complete. Review the generated plots and console outputs for insights.")

```

