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
class_label = np.array([0] * 50 + [1] * 50 + [2] * 50)

data_multi = pd.DataFrame({
    'Word Count': word_count,
    'Link Count': link_count,
    'Class': class_label
})

# Split the data and train model
X_multi = data_multi[['Word Count', 'Link Count']]
y_multi = data_multi['Class']
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi, test_size=0.3, random_state=42)

model_multi = LogisticRegression(solver='lbfgs', max_iter=1000)
model_multi.fit(X_train_multi, y_train_multi)

# Plot with proper decision boundaries
plt.figure(figsize=(10, 8))

# Create a mesh for decision boundary visualization
h = 0.5
x_min, x_max = X_multi['Word Count'].min() - 5, X_multi['Word Count'].max() + 5
y_min, y_max = X_multi['Link Count'].min() - 5, X_multi['Link Count'].max() + 5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict on the mesh
Z = model_multi.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundaries
plt.contour(xx, yy, Z, colors='black', linestyles='--', alpha=0.5)

# Plot data points
colors = ['blue', 'red', 'green']
markers = ['o', 'x', 's']
labels = ['Not Spam', 'Spam', 'Promotional']

for class_value, color, marker, label in zip([0, 1, 2], colors, markers, labels):
    plt.scatter(data_multi[data_multi['Class'] == class_value]['Word Count'],
                data_multi[data_multi['Class'] == class_value]['Link Count'],
                c=color, marker=marker, label=label, s=50)

plt.xlabel('Word Count')
plt.ylabel('Link Count')
plt.title('Logistic Regression Decision Boundaries for Multi-Class Classification')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Evaluate the model
y_pred_multi = model_multi.predict(X_test_multi)
accuracy_multi = accuracy_score(y_test_multi, y_pred_multi)
conf_matrix_multi = confusion_matrix(y_test_multi, y_pred_multi)

print(f'Accuracy: {accuracy_multi:.2f}')
print('Confusion Matrix:')
print(conf_matrix_multi)

# Additional metrics
from sklearn.metrics import classification_report
print('\nClassification Report:')
print(classification_report(y_test_multi, y_pred_multi, 
                          target_names=['Not Spam', 'Spam', 'Promotional']))