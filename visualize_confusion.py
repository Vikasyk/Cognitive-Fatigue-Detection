import matplotlib.pyplot as plt
import numpy as np

# CONFUSION MATRIX VALUES (from your graph)
cm = np.array([
    [105, 23, 5],   # Actual: Neutral (0)
    [38, 21, 0],    # Actual: Stress  (1)
    [18, 1, 14]     # Actual: Amusement (2)
])

# ------------------------------
# 1️⃣ BAR CHART: Correct vs Incorrect Per Class
# ------------------------------
correct = np.diag(cm)
incorrect = np.sum(cm, axis=1) - correct
classes = ['Neutral', 'Stress', 'Amusement']

plt.figure(figsize=(8, 5))
x = np.arange(len(classes))
plt.bar(x - 0.2, correct, width=0.4, label='Correct')
plt.bar(x + 0.2, incorrect, width=0.4, label='Incorrect', color='orange')

plt.xticks(x, classes)
plt.ylabel("Number of Samples")
plt.title("Correct vs Incorrect Predictions Per Class")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------
# 2️⃣ CLASS ACCURACY BAR CHART
# ------------------------------
accuracy_per_class = correct / np.sum(cm, axis=1)

plt.figure(figsize=(8, 5))
plt.bar(classes, accuracy_per_class, color='skyblue')
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.title("Accuracy Per Class")
for i, v in enumerate(accuracy_per_class):
    plt.text(i, v + 0.02, f"{v*100:.1f}%", ha='center')
plt.tight_layout()
plt.show()

# ------------------------------
# 3️⃣ PIE CHART: Correct vs Wrong Overall
# ------------------------------
total_correct = np.sum(correct)
total_wrong = np.sum(cm) - total_correct

plt.figure(figsize=(6, 6))
plt.pie(
    [total_correct, total_wrong],
    labels=['Correct', 'Incorrect'],
    autopct='%1.1f%%',
    colors=['green', 'red']
)
plt.title("Overall Prediction Accuracy")
plt.show()
