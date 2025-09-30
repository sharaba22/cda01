import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Generate a new dataset with three separable classes
np.random.seed(42)
word_count_not_spam = np.random.randint(40, 60, size=50)
link_count_not_spam = np.random.randint(0, 3, size=50)

word_count_spam = np.random.randint(20, 40, size=50)
link_count_spam = np.random.randint(4, 7, size=50)

word_count_promotional = np.random.randint(60, 80, size=50)
link_count_promotional = np.random.randint(20, 40, size=50)

word_count = np.concatenate([word_count_not_spam, word_count_spam, word_count_promotional])
link_count = np.concatenate([link_count_not_spam, link_count_spam, link_count_promotional])
class_label = np.array([0] * 50 + [1] * 50 + [2] * 50)  # Not Spam, Spam, Promotional

data_multi = pd.DataFrame({
    'Word Count': word_count,
    'Link Count': link_count,
    'Class': class_label
})

# Save the data to a CSV file
data_multi.to_csv('multi_class_data.csv', index=False)

# Load the data from the CSV file
data_multi = pd.read_csv('multi_class_data.csv')


# Plot the data points
plt.figure(figsize=(8, 6))
for class_value, marker, label in zip([0, 1, 2], ['o', 'x', 's'], ['Not Spam', 'Spam', 'Promotional']):
    plt.scatter(data_multi[data_multi['Class'] == class_value]['Word Count'],
                data_multi[data_multi['Class'] == class_value]['Link Count'],
                marker=marker, label=label)

plt.xlabel('Word Count')
plt.ylabel('Link Count')
plt.title('Logistic Regression Decision Lines for Multi-Class Classification')
plt.legend()


# Split the data into training and test sets
X_multi = data_multi[['Word Count', 'Link Count']]
y_multi = data_multi['Class']
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size=0.3, random_state=42)

# Train the multinomial logistic regression model
model_multi = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model_multi.fit(X_train_multi, y_train_multi)


# Plot the decision lines
x_values = np.array([X_multi['Word Count'].min(), X_multi['Word Count'].max()])
for i, color in zip(range(3), ['blue', 'red', 'green']):
    y_values = -(model_multi.coef_[i][0] * x_values + model_multi.intercept_[i]) / model_multi.coef_[i][1]
#    plt.plot(x_values, y_values, color=color, linestyle='--', label=f'Decision Line {i}')

plt.show()

# Evaluate the model
y_pred_multi = model_multi.predict(X_test_multi)
accuracy_multi = accuracy_score(y_test_multi, y_pred_multi)
conf_matrix_multi = confusion_matrix(y_test_multi, y_pred_multi)
print(f'Accuracy: {accuracy_multi:.2f}')
print('Confusion Matrix:')
print(conf_matrix_multi)
