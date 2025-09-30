import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate synthetic data with 3 features and 2 classes
X, y = make_classification(n_features=3, n_redundant=0, n_informative=3,
                           n_clusters_per_class=1, n_classes=2, random_state=42, n_samples=100)

# Fit logistic regression model
model = LogisticRegression(solver='liblinear')
model.fit(X, y)

# Create a mesh grid for plotting decision boundary
x_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 50)
y_range = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 50)
X_mesh, Y_mesh = np.meshgrid(x_range, y_range)
Z_mesh = -(model.coef_[0][0] * X_mesh + model.coef_[0][1] * Y_mesh + model.intercept_[0]) / model.coef_[0][2]

# Correcting the visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for each class
for class_value in np.unique(y):
    # Get indices of rows with the class label
    row_ix = np.where(y == class_value)
    # Create scatter of these samples
    ax.scatter(X[row_ix, 0], X[row_ix, 1], X[row_ix, 2], label=f'Class {class_value}')

# Plotting the decision boundary
ax.plot_surface(X_mesh, Y_mesh, Z_mesh, color='green', alpha=0.3, edgecolor='none')

# Axis labels and title
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.legend()
plt.title('Logistic Regression Decision Boundary')
plt.show()
