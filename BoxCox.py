import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load data
df = pd.read_csv("concrete.csv")

# Features and target
x = df.drop(columns=['strength'])
y = df['strength']

# Train/Test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Predict and evaluate
y_pred = lr.predict(x_test)
print("Test R2 Score:", r2_score(y_test, y_pred))

# Cross-validation score
lr = LinearRegression()
cross_val_mean = np.mean(cross_val_score(lr, x, y, scoring='r2'))
print("Cross-validated R2 Score:", cross_val_mean)

# Visualization for each feature
for col in x_train.columns:
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(x_train[col], kde=True)
    plt.title(col)

    plt.subplot(1, 2, 2)
    stats.probplot(x_train[col], dist="norm", plot=plt)
    plt.title(f"{col} QQ Plot")
    plt.tight_layout()
    plt.show()


# Applying Box-Cox Tranform

pt = PowerTransformer(method="box-cox")

x_train_transformed = pt.fit_transform(x_train+0.000001)
x_test_transformed = pt.transform(x_test+0.000001)

pd.DataFrame({'cols':x_train.columns, 'box_cox_lambdas':pt.lambdas_})
# Applying linear regression on tranformed data
lr = LinearRegression()
lr.fit(x_train_transformed, y_train)

y_pred2 = lr.predict(x_test_transformed)

r2_score(y_test,y_pred2)
