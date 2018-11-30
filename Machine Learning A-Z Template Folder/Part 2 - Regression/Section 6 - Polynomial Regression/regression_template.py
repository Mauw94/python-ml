# Polynomial Regression
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.fit_transform(y_test)"""

# Fitting the Regression Model to the dataset
# Create your regressor here

# Predict a new result result
y_pred = regressor.predict(6.5)

# Visualise the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color ='blue')
plt.title('Truth or BLuff (Regression Model)')
plt.xlabel('Position level:')
plt.ylabel('Salary')
plt.show()

# Visualise the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color ='blue')
plt.title('Truth or BLuff (Regression Model)')
plt.xlabel('Position level:')
plt.ylabel('Salary')
plt.show()