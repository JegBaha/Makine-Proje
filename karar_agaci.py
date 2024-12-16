import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Veri setini yükle
file_path = "heart_normalized.csv"
data = pd.read_csv(file_path)

# Özellikler ve hedef değişkeni ayır
X = data.drop(columns=["HeartDisease"])  # Girdi özellikleri
y = data["HeartDisease"]  # Hedef değişken

# Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Grid Search ile hiperparametre optimizasyonu
param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'max_depth': range(1, 10),
    'min_weight_fraction_leaf': [0.0, 0.1, 0.2, 0.3],
    'max_features': [None, 'sqrt', 'log2', 0.5, 0.7],
    'ccp_alpha': [0.0, 0.01, 0.1, 1.0]
}

# Karar ağacı modeli
clf = DecisionTreeClassifier(random_state=0)

# Grid Search ile hiperparametre arama
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# En iyi parametreleri ve modeli al
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
print("En İyi Parametreler (Grid Search Sonucu):", best_params)

# Test seti üzerinde tahmin yap
y_pred = best_model.predict(X_test)

# Sonuçları değerlendir
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Eğitim setindeki doğruluk
y_train_pred = best_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Eğitim Setindeki Doğruluk: {train_accuracy:.2f}")

# Confusion matrix oluştur ve tablo(dataframe) olarak göster
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_df = pd.DataFrame(conf_matrix, 
                               columns=["No Heart Disease (Pred)", "Heart Disease (Pred)"],
                               index=["No Heart Disease (Actual)", "Heart Disease (Actual)"])
print("\nConfusion Matrix:")
print(conf_matrix_df)

# Karar ağacını görselleştir
plt.figure(figsize=(20, 10))
plot_tree(best_model, feature_names=X.columns, class_names=["No Heart Disease", "Heart Disease"], filled=True)
plt.title("Karar Ağacı (Optimizasyon Sonrası)")
plt.show()

# Performans sonuçlarını pandas DataFrame formatında organize et
classification_rep = classification_report(y_test, y_pred, output_dict=True)
results_df = pd.DataFrame({
    "Metric": ["Accuracy", "Macro Avg Precision", "Macro Avg Recall", "Macro Avg F1-Score", 
               "Weighted Avg Precision", "Weighted Avg Recall", "Weighted Avg F1-Score"],
    "Score": [
        accuracy,
        classification_rep["macro avg"]["precision"],
        classification_rep["macro avg"]["recall"],
        classification_rep["macro avg"]["f1-score"],
        classification_rep["weighted avg"]["precision"],
        classification_rep["weighted avg"]["recall"],
        classification_rep["weighted avg"]["f1-score"]
    ]
})

# Performans tablosunu çiz
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=results_df.values, colLabels=results_df.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(results_df.columns))))
plt.title("Performans Tablosu", pad=20)
plt.show()

# Confusion Matrix'i görselleştir
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap='Blues', alpha=0.7)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', fontsize=12)
        
ax.set_xticks(range(2))
ax.set_yticks(range(2))
ax.set_xticklabels(["No Heart Disease (Pred)", "Heart Disease (Pred)"], rotation=45, ha="left")
ax.set_yticklabels(["No Heart Disease (Actual)", "Heart Disease (Actual)"])
plt.title("Confusion Matrix", pad=20)
plt.show()

# Sonuçları yazdır
print(f"Doğruluk: {accuracy * 100:.2f}%")
print(f"F1 Skoru: {classification_rep['weighted avg']['f1-score'] * 100:.2f}%")
print(f"Precision: {classification_rep['weighted avg']['precision'] * 100:.2f}%")
print(f"Recall: {classification_rep['weighted avg']['recall'] * 100:.2f}%")




#  GRID SEARCH İLE Hiperparametre arama sonucu 0.85 geldi """  """

#random search ve grid search arasındaki fark:
# random search verilen aralığı rastgele değerler ile dener. genellikle zaman kısıtı varsa uygundur
#grid search her kombinansyonu dener bu yüzden yavaştır ve zaman sınırı olmadığında daha uygundur

""" #Random search sonrası 0.847 accuracy 
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from scipy.stats import randint

# Veri setini yükle
file_path = "heart_normalized.csv"
data = pd.read_csv(file_path)

# Özellikler ve hedef değişkeni ayır
X = data.drop(columns=["HeartDisease"])  # Girdi özellikleri
y = data["HeartDisease"]  # Hedef değişken

# Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomizedSearchCV ile hiperparametre optimizasyonu
param_dist = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': randint(2, 11),  # Randomized aralık: 2 ile 10 arasında
    'min_samples_leaf': randint(1, 11)    # Randomized aralık: 1 ile 10 arasında
}

# Karar ağacı modeli
clf = DecisionTreeClassifier(random_state=42)

# Randomized Search
random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_dist, n_iter=100, cv=5, scoring='accuracy', random_state=42)
random_search.fit(X_train, y_train)

# En iyi parametreleri ve modeli al
best_params = random_search.best_params_  # En iyi parametreler
best_model = random_search.best_estimator_  # En iyi model
print("En İyi Parametreler (Random Search Sonucu):", best_params)  # Random Search'ten alınan en iyi parametreler

# Test seti üzerinde tahmin yap
y_pred = best_model.predict(X_test)

# Sonuçları değerlendir
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix oluştur ve tablo olarak göster
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_df = pd.DataFrame(conf_matrix, 
                               columns=["No Heart Disease (Pred)", "Heart Disease (Pred)"],
                               index=["No Heart Disease (Actual)", "Heart Disease (Actual)"])
print("\nConfusion Matrix:")
print(conf_matrix_df)

# Karar ağacını görselleştir
plt.figure(figsize=(20, 10))
plot_tree(best_model, feature_names=X.columns, class_names=["No Heart Disease", "Heart Disease"], filled=True)
plt.title("Karar Ağacı (Random Search Sonrası)")
plt.show()

# Performans metrikleri
classification_rep = classification_report(y_test, y_pred, output_dict=True)

# Performans sonuçlarını pandas DataFrame formatında organize et
results_df = pd.DataFrame({
    "Metric": ["Accuracy", "Macro Avg Precision", "Macro Avg Recall", "Macro Avg F1-Score", 
               "Weighted Avg Precision", "Weighted Avg Recall", "Weighted Avg F1-Score"],
    "Score": [
        accuracy,
        classification_rep["macro avg"]["precision"],
        classification_rep["macro avg"]["recall"],
        classification_rep["macro avg"]["f1-score"],
        classification_rep["weighted avg"]["precision"],
        classification_rep["weighted avg"]["recall"],
        classification_rep["weighted avg"]["f1-score"]
    ],
    "Explanation": [
        "Genel doğruluk oranı (tüm sınıflar için)", 
        "Her sınıfın doğruluk ortalamasının hesaplanması", 
        "Her sınıfın recall (duyarlılık) ortalaması", 
        "Her sınıfın F1-Skor ortalaması",
        "Her sınıfın ağırlıklı doğruluk oranı (sınıf örnek sayısına göre ağırlıklı)", 
        "Her sınıfın ağırlıklı recall değeri",
        "Her sınıfın ağırlıklı F1-Skoru"
    ]
})

# Karar Ağacı görselleştir (Figure 1)
plt.figure(figsize=(20, 10))
plot_tree(
    best_model, 
    feature_names=X.columns, 
    class_names=["No Heart Disease", "Heart Disease"], 
    filled=True
)
plt.title("Figure 1: Karar Ağacı")
plt.show()

# Performans tablosunu çiz (Figure 2)
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.axis('tight')
ax2.axis('off')
table2 = ax2.table(cellText=results_df.values, colLabels=results_df.columns, cellLoc='center', loc='center')
table2.auto_set_font_size(False)
table2.set_fontsize(10)
table2.auto_set_column_width(col=list(range(len(results_df.columns))))
plt.title("Figure 2: Performans Tablosu", pad=20)
plt.show()

# Confusion Matrix'i pandas DataFrame ile organize et
conf_matrix_df = pd.DataFrame(
    conf_matrix, 
    columns=["No Heart Disease (Pred)", "Heart Disease (Pred)"],
    index=["No Heart Disease (Actual)", "Heart Disease (Actual)"]
)

# Confusion Matrix'i çiz (Figure 3)
fig3, ax3 = plt.subplots(figsize=(5, 5))
ax3.matshow(conf_matrix, cmap='Blues', alpha=0.7)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax3.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', fontsize=12)
        
ax3.set_xticks(range(2))
ax3.set_yticks(range(2))
ax3.set_xticklabels(["No Heart Disease (Pred)", "Heart Disease (Pred)"], rotation=45, ha="left")
ax3.set_yticklabels(["No Heart Disease (Actual)", "Heart Disease (Actual)"])
plt.title("Figure 3: Confusion Matrix", pad=20)
plt.show() """


