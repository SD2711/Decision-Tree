import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import tree

# Импортируем данные
df = pd.read_csv("dataR2.csv")
X, y = df[["Age", "Glucose"]].values, df["Classification"]

print(df[["Age", "Glucose", "Classification"]].head())
print()
print(X.shape, y.shape)

# Формируем обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

print()
print(X_train.shape, y_train.shape)
print()
print(X_test.shape, y_test.shape)

# Применяем метод деревьев решений
classifier = DecisionTreeClassifier(random_state=1)
classifier = classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(
    "Number of mislabeled points out of a total %d points : %d"
    % (X_test.shape[0], (y_test != y_pred).sum())
)

# Построение дерева решений
plt.figure(figsize=(14, 8))
tree.plot_tree(
    classifier,
    feature_names=["Age", "Glucose"],
    class_names=["1", "2"],
    filled=True,
    fontsize=8,
)
plt.show()

# Формируем кросс-валидационную таблицу
dat = {"y_Actual": y_test, "y_Predicted": y_pred}
dff = pd.DataFrame(dat, columns=["y_Actual", "y_Predicted"])
cross_table = pd.crosstab(
    dff["y_Actual"],
    dff["y_Predicted"],
    rownames=["Actual"],
    colnames=["Predicted"],
    margins=True,
)
print(cross_table)

# Задаем новые точки для прогнозирования
n = np.array([[45, 80], [62, 98], [55, 120], [70, 150]])

# Делаем прогноз для новых точек
new_points = classifier.predict(n)
print(new_points)

# Визуализируем результат классификации тестовой выборки
plt.figure(1)
c = ["blue" if e == 1 else "red" for e in y_pred]
plt.scatter(X_test[:, 0], X_test[:, 1], color=c, linewidths=0.1)
plt.xlabel("Age")
plt.ylabel("Glucose")

# Визуализируем новые точки для прогнозирования
plt.figure(2)
plt.scatter(X_test[:, 0], X_test[:, 1], color=c, linewidths=0.01)
plt.scatter(n[0, 0], n[0, 1], color="green", label="Первая точка", linewidths=2)
plt.scatter(n[1, 0], n[1, 1], color="orange", label="Вторая точка", linewidths=2)
plt.scatter(n[2, 0], n[2, 1], color="purple", label="Третья точка", linewidths=2)
plt.scatter(n[3, 0], n[3, 1], color="black", label="Четвертая точка", linewidths=2)
plt.xlabel("Age")
plt.ylabel("Glucose")
plt.legend()
plt.show()

print("Accuracy:", accuracy_score(y_test, y_pred))
