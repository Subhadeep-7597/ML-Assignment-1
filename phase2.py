# ML Assignment 1 - Phase 2 (var2)
# Thermal reservoir problem: predict Thermal Anomaly Score (y) from x1, x2, x3
# Roll no: BT2024219

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')  # lasso gives a lot of convergence warnings

# ---------------- Load the data ----------------
train_data = pd.read_csv("data/BT2024219_train_var2.csv")
test_data = pd.read_csv("data/BT2024219_test_var2.csv")

print("Train shape:", train_data.shape)
print("Test shape:", test_data.shape)
print(train_data.head())
print(train_data.describe())

X = train_data.drop('y', axis=1)
y = train_data['y']
X_test = test_data

# keep 20% of train data aside to check how the model does on unseen data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# same helper functions as phase 1
def make_model(degree, reg):
    return make_pipeline(PolynomialFeatures(degree=degree, include_bias=False),
                         StandardScaler(),
                         reg)


def evaluate(model):
    model.fit(X_train, y_train)
    train_mse = mean_squared_error(y_train, model.predict(X_train))
    val_pred = model.predict(X_val)
    val_mse = mean_squared_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    return train_mse, val_mse, val_r2


# ---------------- Try 1: plain Linear Regression with different degrees ----------------
# question says it can be up to degree 20, but only 3 features here so
# going high is still ok. tried till 15, above that it was just getting worse
print("\n===== Try 1: Linear Regression (no regularization) =====")
degrees = range(1, 16)
train_errors = []
val_errors = []

for d in degrees:
    train_mse, val_mse, val_r2 = evaluate(make_model(d, LinearRegression()))
    train_errors.append(train_mse)
    val_errors.append(val_mse)
    print(f"degree {d}: train MSE = {train_mse:.4f}, val MSE = {val_mse:.4f}, val R2 = {val_r2:.4f}")

best_lin_degree = list(degrees)[np.argmin(val_errors)]
print("Best degree for linear regression:", best_lin_degree)
# observation: val error keeps dropping till degree 8, after that it slowly
# goes up and after 12 it completely blows up (overfitting)

plt.figure()
plt.plot(degrees, train_errors, marker='o', label='Train MSE')
plt.plot(degrees, val_errors, marker='o', label='Validation MSE')
plt.yscale('log')
plt.xlabel("Polynomial degree")
plt.ylabel("MSE")
plt.title("Phase 2: Degree vs Error (Linear Regression)")
plt.legend()
plt.grid(True)
plt.savefig("plots/var2_degree_vs_error.png")
plt.close()


# ---------------- Try 2: Ridge with different alphas ----------------
# maybe with regularization the higher degrees will stop overfitting
print("\n===== Try 2: Ridge Regression =====")
ridge_alphas = [0.001, 0.01, 0.1, 1, 10]
best_ridge = {'val_mse': float('inf')}

for d in range(5, 15):
    for a in ridge_alphas:
        train_mse, val_mse, val_r2 = evaluate(make_model(d, Ridge(alpha=a)))
        print(f"degree {d}, alpha {a}: val MSE = {val_mse:.4f}, val R2 = {val_r2:.4f}")
        if val_mse < best_ridge['val_mse']:
            best_ridge = {'degree': d, 'alpha': a, 'val_mse': val_mse}

print("Best Ridge:", best_ridge)


# ---------------- Try 3: Lasso with different alphas ----------------
print("\n===== Try 3: Lasso Regression =====")
lasso_alphas = [0.0001, 0.001, 0.01, 0.1]
best_lasso = {'val_mse': float('inf')}

# lasso is slow for big degrees, so stopping at 12
for d in range(5, 13):
    for a in lasso_alphas:
        train_mse, val_mse, val_r2 = evaluate(make_model(d, Lasso(alpha=a, max_iter=10000)))
        print(f"degree {d}, alpha {a}: val MSE = {val_mse:.4f}, val R2 = {val_r2:.4f}")
        if val_mse < best_lasso['val_mse']:
            best_lasso = {'degree': d, 'alpha': a, 'val_mse': val_mse}

print("Best Lasso:", best_lasso)


# ---------------- Compare the best of each using 5-fold cross validation ----------------
print("\n===== Comparing best models with 5-fold CV =====")
candidates = {
    'Linear': make_model(best_lin_degree, LinearRegression()),
    'Ridge': make_model(best_ridge['degree'], Ridge(alpha=best_ridge['alpha'])),
    'Lasso': make_model(best_lasso['degree'], Lasso(alpha=best_lasso['alpha'], max_iter=10000)),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}
for name, model in candidates.items():
    mse = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error').mean()
    r2 = cross_val_score(model, X, y, cv=kf, scoring='r2').mean()
    cv_results[name] = mse
    print(f"{name}: CV MSE = {mse:.4f}, CV R2 = {r2:.4f}")

best_name = min(cv_results, key=cv_results.get)
final_model = candidates[best_name]
print("\nFinal choice:", best_name)
if best_name == 'Ridge':
    print("degree =", best_ridge['degree'], ", alpha =", best_ridge['alpha'])
elif best_name == 'Lasso':
    print("degree =", best_lasso['degree'], ", alpha =", best_lasso['alpha'])
else:
    print("degree =", best_lin_degree)


# ---------------- Plot actual vs predicted on validation set ----------------
final_model.fit(X_train, y_train)
val_pred = final_model.predict(X_val)

plt.figure()
plt.scatter(y_val, val_pred, alpha=0.5, color='green')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel("Actual y")
plt.ylabel("Predicted y")
plt.title(f"Phase 2: Actual vs Predicted (validation) - {best_name}")
plt.grid(True)
plt.savefig("plots/var2_plot.png")
plt.close()


# ---------------- Train on full training data and predict test ----------------
final_model.fit(X, y)
predictions = final_model.predict(X_test)

output = pd.DataFrame({'y': predictions})
output.to_csv("BT2024219_pred_var2.csv", index=False)
print("\nSaved predictions to BT2024219_pred_var2.csv")
