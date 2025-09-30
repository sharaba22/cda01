import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('spam-data.csv')

# Plot the data points
plt.figure(figsize=(8, 6))
plt.scatter(data[data['Is Spam'] == 0]['Word Count'], data[data['Is Spam'] == 0]['Link Count'], label='Not Spam', marker='o')
plt.scatter(data[data['Is Spam'] == 1]['Word Count'], data[data['Is Spam'] == 1]['Link Count'], label='Spam', marker='x')
plt.xlabel('Word Count')
plt.ylabel('Link Count')
plt.legend()
plt.title('Spam Data Visualization')


# Split the data into training and test sets
X = data[['Word Count', 'Link Count']]
y = data['Is Spam']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)

# Plot the decision line
# The decision line is given by the equation: model.coef_[0] * x + model.coef_[1] * y + model.intercept_ = 0
# Solving for y: y = -(model.coef_[0] * x + model.intercept_) / model.coef_[1]
x_values = np.array([X['Word Count'].min(), X['Word Count'].max()])
y_values = -(model.coef_[0][0] * x_values + model.intercept_[0]) / model.coef_[0][1]
plt.plot(x_values, y_values, color='green', label='Decision Line')

plt.title('Spam Data with Logistic Regression Decision Line')
plt.show()