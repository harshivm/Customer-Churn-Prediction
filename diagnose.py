# import pandas as pd

# df = pd.read_csv('telco_dataset.csv')
# df.drop(columns=['customerID'], inplace=True)
# df['Churn'] = df['Churn'].map({'yes':1,'no':0})
# binary_cols = ['Partner','Dependents','PhoneService','MultipleLines','PaperlessBilling']
# for cols in binary_cols:
#     df[cols] = df[cols].map({'yes':1, 'no':0})

# df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# print(f"Rows before get_dummies: {len(df)}")
# print(f"NaN before get_dummies:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# df = pd.get_dummies(df, drop_first=True)
# print(f"\nRows after get_dummies: {len(df)}")
# print(f"NaN after get_dummies:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# print(f"\nRows before dropna: {len(df)}")
# df = df.dropna()
# print(f"Rows after dropna: {len(df)}")
