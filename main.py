import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the dataset
data = pd.read_csv('Salary_Data.csv')

# Explore the dataset
print("First 5 rows of the dataset:")
print(data.head())

print("\nDataset information:")
print(data.info())

print("\nSummary statistics of the dataset:")
print(data.describe())

# Visualize the data
plt.figure(figsize=(10, 6))
sns.scatterplot(x='YearsExperience', y='Salary', data=data)
plt.title('Scatter Plot of Years of Experience vs. Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Prepare the data for training
# Extract the columns as arrays
X = data['YearsExperience'].values
y = data['Salary'].values

# Check the shapes of X and y (number of data points)
print("X shape:", X.shape)
print("y shape:", y.shape)

# Split the data into training and testing sets
# Use 70% of the data for training and 30% for testing (random_state=100 for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)

# Reshape X_train and X_test to be 2D arrays
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluation
# Display the coefficients (slope) and intercept (bias) of the trained model
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Calculate and display the R-squared value on the training set
print("R-squared (training set):", model.score(X_train, y_train))

# Make predictions on the test set
y_pred = model.predict(X_test)

# Visualize the results on the training set
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, model.predict(X_train), color='red')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualize the results on the test set
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Plot actual vs. predicted values
data_point_indices = [i for i in range(1, len(y_test) + 1, 1)]  # Generate indices for data points
plt.figure(figsize=(10, 6))
plt.plot(data_point_indices, y_test, color='blue', linewidth=2.5, linestyle='-', label='Actual')
plt.plot(data_point_indices, y_pred, color='red', linewidth=2.5, linestyle='-', label='Predicted')
plt.title('Actual vs Predicted')
plt.xlabel('Index')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Plot error terms
plt.figure(figsize=(10, 6))
plt.plot(data_point_indices, y_test - y_pred, color='blue', linewidth=2.5, linestyle='-')
plt.title('Error Terms')
plt.xlabel('Index')
plt.ylabel('ytest - ypred')
plt.show()

# Scatter plot of actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Salary (Y Test)')
plt.ylabel('Predicted Salary (Predicted Y)')
plt.title('Actual vs Predicted Salary')
plt.show()

# Model evaluation metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2_score = metrics.r2_score(y_test, y_pred)

print('Mean Absolute Error (MAE):', mae)
print('Mean Squared Error (MSE):', mse)
print('Root Mean Squared Error (RMSE):', rmse)
print('R-squared Score:', r2_score)
