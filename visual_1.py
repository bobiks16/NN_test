import matplotlib.pyplot as plt

train_classes_distribution = np.bincount(y_train_classes)
test_classes_distribution = np.bincount(y_test_classes)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.bar(range(len(train_classes_distribution)), train_classes_distribution, color='blue')
plt.title('title')
plt.xlabel('sub1')
plt.ylabel('sub2')

plt.subplot(1, 2, 2)
plt.bar(range(len(test_classes_distribution)), test_classes_distribution, color='orange')
plt.title('title')
plt.xlabel('sub1')
plt.ylabel('sub2')

plt.tight_layout()
plt.show()
