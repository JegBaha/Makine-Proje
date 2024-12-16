import pandas as pd
from sklearn.model_selection import train_test_split
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
#test size 0,2= verinin %20'si test için kalanı eğitim için kullanılır
#random_state veri setinin rastgelelik kontrol eder
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Karar ağacı modelini oluştur ve eğit (en iyi hiperparametrelerle)

best_params = {'criterion': 'gini','ccp_alpha': 0.0, 'max_depth': 4, 'min_samples_leaf': 5,'max_features': None, 'min_samples_split': 4,'splitter':'random',}
#bölünme kriteri gini,derinliği 5 ,bir yaprak düğümde en az 4 örnek olmalı,bir düğümde daha fazla dallanma yapmak için
#en az 5 örnek gereklidir
clf = DecisionTreeClassifier(**best_params, random_state=0)
#karar ağacı sınıflandırıcı. karar ağacını oluşturup eğitim verisi ile eğitir (x ve y)
clf.fit(X_train, y_train)

# Tahmin yap 
y_pred = clf.predict(X_test)
#eğitilen model x test verileri üzerinden tahmin yapar

# Performans metrikleri
accuracy = accuracy_score(y_test, y_pred)
#doğruluk oranı hesaplar y_test true positive diğeri true negatif toplanır
classification_rep = classification_report(y_test, y_pred, output_dict=True)
#modelin doğruluğu,precision,recall,f1 hesaplar ve sözlük olarak döner
conf_matrix = confusion_matrix(y_test, y_pred)
#confusion matrix oluşturur

#Performans sonuçlarını pandas DataFrame formatında oluştur
#results_df performans metriklerini dataframe oluşturur
#Metric Score ve Explanation satırlarını oluşturur
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

# Etiketlerin döndürülmesi
plt.figure(figsize=(25, 7.5))
plot_tree(
    clf, 
    feature_names=X.columns, 
    class_names=["NHD", "HD"], 
    filled=True
)

# Etiketleri döndürme
plt.xticks(rotation=45)  # Etiketlerin yatayda döndürülmesi
plt.yticks(rotation=45)  # Etiketlerin dikeyde döndürülmesi

plt.title("Figure 1: Karar Ağacı")
plt.show()



# Performans tablosunu çiz (Figure 2)
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.axis('tight')
#axis eksen sınırlarını ayarlar
ax2.axis('off')
# x y eksenlerini gizler. 

# Tabloyu oluştur
#performans sonuçlarını tablolaştırır
table2 = ax2.table(cellText=results_df.values, colLabels=results_df.columns, cellLoc='center', loc='center')

# 1. satır ve sütunu koyu gri yapmak
for (i, j), cell in table2.get_celld().items():
    if i == 0 or j == 0:  # İlk satır ve ilk sütun
        cell.set_text_props(weight='bold', color='white')  # Yazıyı beyaz yap
        cell.set_facecolor('#4C4C4C')  # Koyu gri renk
    else:
        cell.set_text_props(weight='normal', color='black')  # Diğer hücreler normal
        cell.set_facecolor('white')  # Diğer hücreler beyaz

# Tabloyu stilize et
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
plt.show()


from sklearn.model_selection import cross_val_score

# ROC AUC hesapla
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_test, y_pred)

# Çapraz doğrulama metriklerini hesapla
cv_accuracy = cross_val_score(clf, X, y, cv=5, scoring='accuracy').mean() * 100
cv_f1 = cross_val_score(clf, X, y, cv=5, scoring='f1_weighted').mean() * 100
cv_precision = cross_val_score(clf, X, y, cv=5, scoring='precision_weighted').mean() * 100
cv_recall = cross_val_score(clf, X, y, cv=5, scoring='recall_weighted').mean() * 100
cv_roc_auc = cross_val_score(clf, X, y, cv=5, scoring='roc_auc').mean() * 100

# Sonuçları yazdır
print("En İyi Parametreler (Grid Search Sonucu):", best_params)
print(f"Doğruluk: {accuracy * 100:.2f}%")
print(f"F1 Skoru: {classification_rep['weighted avg']['f1-score'] * 100:.2f}%")
print(f"Precision: {classification_rep['weighted avg']['precision'] * 100:.2f}%")
print(f"Recall: {classification_rep['weighted avg']['recall'] * 100:.2f}%")
print(f"Çapraz Doğrulama (ROC AUC): {cv_roc_auc:.2f}%")
print(f"Çapraz Doğrulama (Accuracy): {cv_accuracy:.2f}%")
print(f"Çapraz Doğrulama (F1): {cv_f1:.2f}%")
print(f"Çapraz Doğrulama (Precision): {cv_precision:.2f}%")
print(f"Çapraz Doğrulama (Recall): {cv_recall:.2f}%")
print(f"ROC AUC: {roc_auc * 100:.2f}%\n")

# Classification Report'u yazdır
print("Classification Report:\n")
print(classification_report(y_test, y_pred))



from sklearn.metrics import roc_curve, precision_recall_curve

# ROC Curve Hesaplama
fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])

# Precision-Recall Curve Hesaplama
precision, recall, _ = precision_recall_curve(y_test, clf.predict_proba(X_test)[:, 1])

# ROC Eğrisini Çiz (Figure 4)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Figure 4: ROC Curve')
plt.legend(loc='lower right')
plt.grid(alpha=0.5)
plt.show()

# Precision-Recall Eğrisini Çiz (Figure 5)
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='green', label='Precision-Recall Curve')
plt.xlabel('Recall (Sensitivity)')
plt.ylabel('Precision')
plt.title('Figure 5: Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(alpha=0.5)
plt.show()
