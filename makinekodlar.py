import pandas as panda
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns


##Ön iþlenmiþ verimizi dataSet verisine atýyoruz(Normalizasyon, dump veriler, duplicate verileri düzenlenmiþ bir þekilde) 
dataSet = panda.read_csv('C:/Users/bahab/OneDrive/Masaüstü/heart_normalized.csv')
## veri setinin satýr ve sütun sayýsýný bir tuple ;(demet) olarak döner.
dataSet.shape



##deep=True veri setinin baðýmsýz bir kopyasýný oluþturur. Böylece, dataCopy(test veri seti gibi düþünülebilir ama model üzerindeki test deðil) üzerinde yapýlan deðiþiklikler data'yý etkilemez.
dataCopy = dataSet.copy(deep = True)

correlation = dataCopy[['RestingBP', 'RestingECG']].corr()

print("Korelasyon Matrisi:")
print(correlation)



#yaavaþtan model eðitim süreçleri
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_score
#Çapraz doðrulama (Cross-Validation) kullanarak her kombinasyonu test eder.
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
## roc eðri f1 skore deðþimler için grafik oluþturmaya saðlar 
from sklearn.model_selection import RepeatedStratifiedKFold
##pre ve recall eðricsi çizmek için kullanýlýr.
from sklearn.metrics import precision_recall_curve


## Ana veriyi deðiþtirmemek adýna copyalanmýþ verideki öznitelikler features deðerine atýldý.
## Target özniteliklerini listeden çýkartýyoruyz çünkü data leakage yaratmasýn diye. 
features = dataCopy[dataCopy.columns.drop(['HeartDisease'])].values
## Modelin tahmin etmeye çalýþacaðý deðer.
target = dataCopy['HeartDisease'].values
## features ve target verilerini eðitim ve test setlerine böler.
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.20, random_state = 2)
## test_size = 0.20: Verinin %20'ini test seti olarak ayýrýr, geri kalan %80'i ise eðitim seti olarak kullanýlýr.
## Veriyi bölerken rastgelelik uygulanýr, ancak bu parametre sabit bir deðer (2) ile belirlenmiþ, yani her çalýþtýrýldýðýnda ayný veri bölümü elde edilir. Bu, tekrar edilebilirlik saðlar.
## x_train: Eðitim verisinin öznitelikleri (features) (eðitim seti).
## x_test: Test verisinin öznitelikleri (features) (test seti).
## y_train: Eðitim verisinin hedef deðiþkeni (HeartDisease) (eðitim seti).
## y_test: Test verisinin hedef deðiþkeni (HeartDisease) (test seti).

def modelsThing(classifier):
    ## .fit sayesinde model, verilerin ve hedeflerin iliþkisini öðrenmeye baþlar.
    classifier.fit(x_train,y_train)
    ## predict sayesinde eðitilen veri tahminde bulunmaya baþlar bizde burada eðitim verisinin özniteliklerini deðer olarak veriyoruz.
    prediction = classifier.predict(x_test)
    

    ## Çapraz doðrulama yönetimidir. Amacý veri setini verdiðimiz katman(folds) (buradaki n_splits) katsayýsýna böler ve birden fazla kez tekrar etmesini saðlar(n_repeats)
    # Neden yapar dersek aslýnda bir nevi veri seti dengeli ise bize genelleþtirmesi daha iyi olan overfittingden uzak daha iyi bir sonuç vermesi için çalýþýr.  
    #random_state=1 verilmesinin sebebi katlama ve bölümlerin ayný þekilde oluþmasýný saðlar. Verilmez ise her çalýþtýrmada farklý bir sonuç elde edilebilr.
    cv = RepeatedStratifiedKFold(n_splits = 10,n_repeats = 5,random_state = 1)
    
    ## Bunlar classification_report kütüphanesinde dahil deðildi ek olrk farklý kütüphane üzerinden yaptým.
    ### Anlamlý 2 basamak için böyle bir kod yazýldý; 3.141593 yerine 3.14 yazýlacak '{0:.2%}
    print("Doðruluk : ",'{0:.2%}'.format(accuracy_score(y_test,prediction)))
    
    #Çapraz doðrulamna yapmaya karar verdik sebebi çapraz doðrulama da verilen metrikler ile normal classification reports lar ile 
    #eþ deðer çýkacak mý veya yakýn bir deðer çýkacak mý diye ek olarak kontrol yapmamýza olanak saðladýðý için
    #.mean() sebebi bize bir array döndürdükleri için ve bize sayýsal bir veri lazým oldugundan dolayý ortalamalýrýný bizim için alýp bir deðer döndürür.
    
    print("Çapraz Doðrulama(RocAuc) : ",'{0:.2%}'.format(cross_val_score(classifier,x_train,y_train,cv = cv,scoring = 'roc_auc').mean()))
    print("Çapraz Doðrulama(Accuracy(Doðruluk)) : ",'{0:.2%}'.format(cross_val_score(classifier,x_train,y_train,cv = cv,scoring = 'accuracy').mean()))
    print("Çapraz Doðrulama(F1)) : ",'{0:.2%}'.format(cross_val_score(classifier,x_train,y_train,cv = cv,scoring = 'f1').mean()))
    print("Çapraz Doðrulama(Precision(Kesinlik) : ",'{0:.2%}'.format(cross_val_score(classifier,x_train,y_train,cv = cv,scoring = 'precision').mean()))
    print("ROC_AUC  : ",'{0:.2%}'.format(roc_auc_score(y_test,prediction)))
    
    # modelin sýnýflandýrma (classification) performansýný deðerlendirmek için precision, recall f1-score burada hesaplanýr, Support veri setindeki örnek sayýsýdýr.
    # support genelde 0 a farklý bir sayý 1 e farkýlý bir sayý verir ve toplamlarý genel örnek sayýmýzýn test setindeki verdiðimis yüzdelik ile orantýlýdr.
    print(classification_report(y_test,classifier.predict(x_test)))


#Macro Precision = (0.88 + 0.86) / 2 = 0.87
#Macro Recall = (0.85 + 0.87) / 2 = 0.86
#Macro F1-Score = (0.86 + 0.86) / 2 = 0.86

#Weighted Precision = (Precision(Sýnýf 1) * Örnek sayýsý(Sýnýf 1) + Precision(Sýnýf 2) * Örnek sayýsý(Sýnýf 2)) / (Toplam örnek sayýsý)
#Weighted Recall = (Recall(Sýnýf 1) * Örnek sayýsý(Sýnýf 1) + Recall(Sýnýf 2) * Örnek sayýsý(Sýnýf 2)) / (Toplam örnek sayýsý)
#Weighted F1-Score = (F1(Sýnýf 1) * Örnek sayýsý(Sýnýf 1) + F1(Sýnýf 2) * Örnek sayýsý(Sýnýf 2)) / (Toplam örnek sayýsý)

    
from sklearn.neighbors import KNeighborsClassifier
## Manhattan uzaklýk ölçütü için
#K katsayýsý genel olarak baya denendi. 15 ten sonra belirli bir sayýya kadar düþüþ yaþandý 200 ler gibi sayýlara geldiði zaman %0.10 luk gibi bir doðruluk
#arttý, fakat bu hýzý bozacaðýndan dolayý 15 ideal sayý olarak düþünülmüþtür.
#p=1 oldugu zaman manhattan 2 oldugu zaman öklid
#manhattan oldugu için metriði de manhattan a ayarlýyoruz.
classKNN= KNeighborsClassifier( n_neighbors =15,p = 1,metric='manhattan')
modelsThing(classKNN)

## Manhattan uzaklýk ölçütü için
# weights=Komþularýn sýnýflandýrma üzerindeki aðýrlýklarýný belirler;
#'uniform' tüm komþular eþit aðýrlýk,'distance' Daha yakýn komþular daha yüksek aðýrlýk
classKNN= KNeighborsClassifier(weights='distance', n_neighbors =15,p = 1,metric='manhattan')
modelsThing(classKNN)

## Öklid uzaklýk ölçütü için
classKNN= KNeighborsClassifier( n_neighbors = 15,p = 2,metric="euclidean")
modelsThing(classKNN)

## Öklid uzaklýk ölçütü için
classKNN= KNeighborsClassifier(weights='distance', n_neighbors = 15,p=2,metric="euclidean")
modelsThing(classKNN)


classKNN= KNeighborsClassifier(n_neighbors = 15,metric="chebyshev")
modelsThing(classKNN)

classKNN= KNeighborsClassifier(weights='distance',n_neighbors = 15,metric="chebyshev")
modelsThing(classKNN)

classKNN= KNeighborsClassifier(n_neighbors = 15,metric="hamming")
modelsThing(classKNN)

classKNN= KNeighborsClassifier(weights='distance',n_neighbors = 15,metric="hamming")
modelsThing(classKNN)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Veri setini yükle

data = pd.read_csv('C:/Users/bahab/OneDrive/Masaüstü/heart_normalized.csv')

# Özellikler ve hedef deðiþkeni ayýr
X = data.drop(columns=["HeartDisease"])  # Girdi özellikleri
y = data["HeartDisease"]  # Hedef deðiþken

# Eðitim ve test setlerine ayýr
#test size 0,2= verinin %20'si test için kalaný eðitim için kullanýlýr
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar aðacý modelini oluþtur ve eðit (en iyi hiperparametrelerle)

best_params = {'criterion': 'gini', 'max_depth': 5, 'min_samples_leaf': 4, 'min_samples_split': 5}
#bölünme kriteri gini,derinliði 5 ,bir yaprak düðümde en az 4 örnek olmalý,bir düðümde daha fazla dallanma yapmak için
#en az 5 örnek gereklidir
clf = DecisionTreeClassifier(**best_params, random_state=42)
#karar aðacý sýnýflandýrýcý. karar aðacýný oluþturup eðitim verisi ile eðitir (x ve y)
clf.fit(X_train, y_train)

# Tahmin yap 
y_pred = clf.predict(X_test)
#eðitilen model x test verileri üzerinden tahmin yapar

# Performans metrikleri
accuracy = accuracy_score(y_test, y_pred)
#doðruluk oraný hesaplar y_test true positive diðeri true negatif toplanýr
classification_rep = classification_report(y_test, y_pred, output_dict=True)
#modelin doðruluðu,precision,recall,f1 hesaplar ve sözlük olarak döner
conf_matrix = confusion_matrix(y_test, y_pred)
#confusion matrix oluþturur

#Performans sonuçlarýný pandas DataFrame formatýnda oluþtur
#results_df performans metriklerini dataframe oluþturur
#Metric Score ve Explanation satýrlarýný oluþturur
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
        "Genel doðruluk oraný (tüm sýnýflar için)", 
        "Her sýnýfýn doðruluk ortalamasýnýn hesaplanmasý", 
        "Her sýnýfýn recall (duyarlýlýk) ortalamasý", 
        "Her sýnýfýn F1-Skor ortalamasý",
        "Her sýnýfýn aðýrlýklý doðruluk oraný (sýnýf örnek sayýsýna göre aðýrlýklý)", 
        "Her sýnýfýn aðýrlýklý recall deðeri",
        "Her sýnýfýn aðýrlýklý F1-Skoru"
    ]
})

# Etiketlerin döndürülmesi
plt.figure(figsize=(25, 40))
plot_tree(
    clf, 
    feature_names=X.columns, 
    class_names=["No Heart Disease", "Heart Disease"], 
    filled=True
)

# Etiketleri döndürme
plt.xticks(rotation=45)  # Etiketlerin yatayda döndürülmesi
plt.yticks(rotation=45)  # Etiketlerin dikeyde döndürülmesi

plt.title("Figure 1: Karar Aðacý")
plt.show()



# Performans tablosunu çiz (Figure 2)
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.axis('tight')
#axis eksen sýnýrlarýný ayarlar
ax2.axis('off')
# x y eksenlerini gizler. 

# Tabloyu oluþtur
#performans sonuçlarýný tablolaþtýrýr
table2 = ax2.table(cellText=results_df.values, colLabels=results_df.columns, cellLoc='center', loc='center')

# 1. satýr ve sütunu koyu gri yapmak
for (i, j), cell in table2.get_celld().items():
    if i == 0 or j == 0:  # Ýlk satýr ve ilk sütun
        cell.set_text_props(weight='bold', color='white')  # Yazýyý beyaz yap
        cell.set_facecolor('#4C4C4C')  # Koyu gri renk
    else:
        cell.set_text_props(weight='normal', color='black')  # Diðer hücreler normal
        cell.set_facecolor('white')  # Diðer hücreler beyaz

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
