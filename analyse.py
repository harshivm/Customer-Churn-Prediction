import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
import pickle

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.1]
}

df = pd.read_csv('telco_dataset.csv')
df.head()
df.shape
df.columns
df.info()
df.describe()
df['Churn'].value_counts()
df['Churn'].value_counts(normalize=True)
df.drop(columns=['customerID'], inplace=True)
df['Churn'] = df['Churn'].map({'Yes':1,'No':0})
binary_cols = ['Partner','Dependents','PhoneService','MultipleLines','PaperlessBilling']
for cols in binary_cols:
    df[cols] = df[cols].map({'Yes':1, 'No':0})

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = pd.get_dummies(df, drop_first=True)
print("Missing values before dropna:")
print(df.isnull().sum())
print(f"Total rows before dropna: {len(df)}")

df = df.dropna()
print(f"Total rows after dropna: {len(df)}")

scaler = StandardScaler()
X = df.drop(columns=['Churn'])
Y = df['Churn']

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42)

# print("data are",X_train.shape, X_test.shape, y_train.shape, y_test.shape)

lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)


y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

models = {
    'Logistic Regression': lr,
    'Random Forest': rf,
    'XGBoost': xgb
}


for name, model in models.items():
    y_pred = model.predict(X_test)
    print(name, "Accuracy:", accuracy_score(y_test, y_pred))


feature_importance = pd.Series(
    rf.feature_importances_,
    index=df.drop('Churn', axis=1).columns
).sort_values(ascending=False)

print(feature_importance.head(10))

print("Logistic Regression")

print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall:", recall_score(y_test, y_pred_lr))
print("F1 Score:", f1_score(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_lr))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

print("Random Forest")

print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))

print("XGBoost")

print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Precision:", precision_score(y_test, y_pred_xgb))
print("Recall:", recall_score(y_test, y_pred_xgb))
print("F1 Score:", f1_score(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_xgb))

cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix (Random Forest)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


results = pd.DataFrame({
    "Model": ["Logistic", "Random Forest", "XGBoost"],
    "Recall": [
        recall_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_xgb)
    ],
    "ROC-AUC": [
        roc_auc_score(y_test, y_prob_lr),
        roc_auc_score(y_test, y_prob_rf),
        roc_auc_score(y_test, y_prob_xgb)
    ]
})

print(results)


fpr, tpr, _ = roc_curve(y_test, y_prob_rf)

plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Random Forest)")
plt.show()


grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring='roc_auc',   
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# save the best model and scaler for later use
with open("churn_model.pkl", "wb") as f:
    pickle.dump(best_rf, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

y_pred_best = best_rf.predict(X_test)
y_prob_best = best_rf.predict_proba(X_test)[:, 1]

print("Recall:", recall_score(y_test,y_pred_best))
print("ROC-AUC", roc_auc_score(y_test, y_prob_best))

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

grid_xgb = GridSearchCV(
    xgb,
    param_grid_xgb,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1
)

grid_xgb.fit(X_train, y_train)

best_xgb = grid_xgb.best_estimator_
print("Best Performance XGBoost Para:",grid_xgb.best_params_)

print("Before tuning ROC:", roc_auc_score(y_test, y_prob_rf))
print("After tuning ROC:", roc_auc_score(y_test, y_prob_best))

df['tenure'].hist()
plt.show()

# sns.countplot(x='Contract', hue='Churn', data=df)
# plt.show()
sns.boxplot(x='Churn', y='tenure', data=df)
plt.show()
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True)
plt.show()