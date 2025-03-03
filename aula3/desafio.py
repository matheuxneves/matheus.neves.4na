from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, 
                             recall_score, f1_score, roc_curve, auc)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Carregando o dataset Wine
wine = load_wine()
X = wine.data
y = wine.target

# Dividindo os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizando as features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Encontrando o melhor valor de k com validação cruzada
k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    accuracies.append(scores.mean())

best_k = k_values[np.argmax(accuracies)]
print(f"Melhor valor de k encontrado: {best_k}")

# Treinando o modelo com o melhor k
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Calculando as métricas
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Acurácia:", accuracy)
print("Matriz de Confusão:\n", conf_matrix)
print("Precisão:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# Visualização da matriz de confusão
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=wine.target_names, yticklabels=wine.target_names)
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

# Gráfico de escolha do melhor k
plt.figure(figsize=(8, 4))
plt.plot(k_values, accuracies, marker='o', linestyle='dashed', color='b')
plt.xlabel('Número de Vizinhos (k)')
plt.ylabel('Acurácia Média')
plt.title('Escolha do Melhor k')
plt.show()

# Curva ROC e AUC para multiclasse
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=best_k))
classifier.fit(X_train, label_binarize(y_train, classes=[0, 1, 2]))
y_score = classifier.predict_proba(X_test)

plt.figure(figsize=(8, 6))
colors = ['blue', 'red', 'green']
for i in range(3):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i], label=f'Classe {i} (AUC = {auc_score:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('Falso Positivo Rate')
plt.ylabel('Verdadeiro Positivo Rate')
plt.title('Curvas ROC')
plt.legend()
plt.show()
