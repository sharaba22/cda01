import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# ხელოვნურად მონაცემების გენერაცია
np.random.seed(42)
n_samples = 500

# ნორმალური ტრაფიკის მახასიათებლები
normal_traffic = np.random.multivariate_normal([100, 1000, 0.1], [[2000, 0, 0], [0, 50000, 0], [0, 0, 0.01]], size=n_samples//2)
normal_labels = np.zeros(n_samples//2)

# თავდასხმის ტრაფიკის მახასიათებლები
attack_traffic = np.random.multivariate_normal([200, 1500, 0.5], [[8000, 0, 0], [0, 200000, 0], [0, 0, 0.04]], size=n_samples//2)
attack_labels = np.ones(n_samples//2)

# მონაცემების გაერთიანება
X = np.concatenate((normal_traffic, attack_traffic))
y = np.concatenate((normal_labels, attack_labels))

# ლოგისტიკური რეგრესიის მოდელის შექმნა და გაწვრთნა
model = LogisticRegression()
model.fit(X, y)

# რეგრესიის სიბრტყე
x1, x2 = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 10), np.linspace(X[:, 1].min(), X[:, 1].max(), 10))
x3 = -(model.intercept_[0] + model.coef_[0][0]*x1 + model.coef_[0][1]*x2) / model.coef_[0][2]

# 3D ვიზუალიზაცია
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# ნორმალური ტრაფიკი
ax.scatter(normal_traffic[:, 0], normal_traffic[:, 1], normal_traffic[:, 2], c='blue', marker='o', label='Normal', alpha=0.8)

# თავდასხმის ტრაფიკი
ax.scatter(attack_traffic[:, 0], attack_traffic[:, 1], attack_traffic[:, 2], c='red', marker='^', label='Attack', alpha=0.8)

# რეგრესიის სიბრტყე
ax.plot_surface(x1, x2, x3, alpha=0.7, color=(1.0, 0.5, 0))

# Labels
ax.set_xlabel('Connection Rate (per minute)')
ax.set_ylabel('Average Session Duration (in seconds)')
ax.set_zlabel('Error Rate')
plt.title('Network Traffic Classification: Normal vs DDoS Attack')
plt.legend()

# ვიზუალიზაციის გაუმჯობესება
ax.dist = 10
plt.tight_layout()
plt.show()