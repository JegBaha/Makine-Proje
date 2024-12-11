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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score, precision_score, recall_score

# Veri setini okuma ve ön iþleme
dataSet = pd.read_csv('C:/Users/bahab/OneDrive/Masaüstü/heart_normalized.csv')
dataSet.shape

dataCopy = dataSet.copy(deep=True)

# Öznitelikler ve hedef deðiþkenlerin ayrýlmasý
features = dataCopy[dataCopy.columns.drop(['HeartDisease'])].values
target = dataCopy['HeartDisease'].values

# Eðitim ve test verilerinin bölünmesi
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=2)

# Performans ölçüm fonksiyonu
def evaluate_model(classifier, model_name):
    print(f"Model: {model_name}\n")
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)

    # Çapraz doðrulama
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)

    accuracy = accuracy_score(y_test, prediction)
    roc_auc = roc_auc_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)

    print("Doðruluk:", '{0:.2%}'.format(accuracy))
    print("F1 Skoru:", '{0:.2%}'.format(f1))
    print("Precision:", '{0:.2%}'.format(precision))
    print("Recall:", '{0:.2%}'.format(recall))
    print("Çapraz Doðrulama (ROC AUC):", '{0:.2%}'.format(cross_val_score(classifier, x_train, y_train, cv=cv, scoring='roc_auc').mean()))
    print("Çapraz Doðrulama (Accuracy):", '{0:.2%}'.format(cross_val_score(classifier, x_train, y_train, cv=cv, scoring='accuracy').mean()))
    print("Çapraz Doðrulama (F1):", '{0:.2%}'.format(cross_val_score(classifier, x_train, y_train, cv=cv, scoring='f1').mean()))
    print("Çapraz Doðrulama (Precision):", '{0:.2%}'.format(cross_val_score(classifier, x_train, y_train, cv=cv, scoring='precision').mean()))
    print("Çapraz Doðrulama (Recall):", '{0:.2%}'.format(cross_val_score(classifier, x_train, y_train, cv=cv, scoring='recall').mean()))
    print("ROC AUC:", '{0:.2%}'.format(roc_auc))

    # Sýnýflandýrma raporu
    print("\nClassification Report:\n")
    print(classification_report(y_test, prediction))
    print("-" * 60)

    return accuracy, roc_auc, f1, precision, recall

# Grid Search ile hiperparametre optimizasyonu
def optimize_with_grid_search(classifier, param_grid, model_name):
    print(f"Grid Search ile {model_name} Hiperparametre Optimizasyonu\n")
    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, error_score='raise')
    grid_search.fit(x_train, y_train)

    print(f"En Ýyi Parametreler: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    return evaluate_model(best_model, f"{model_name} (Optimizasyonlu)")

# Modellerin tanýmlanmasý ve parametre aralýklarý
classKNN = KNeighborsClassifier()
knn_param_grid = {
    'n_neighbors': range(1, 200),
    'p': [1, 2],
    'metric': ['euclidean', 'manhattan'],
    'weights': ['distance', 'uniform']
}

classifier_lr = LogisticRegression(random_state=0,max_iter=1000)
lr_param_grid = [
    {'penalty': ['l1'], 'solver': ['liblinear', 'saga'], 'C': [0.1, 1, 10]},
    {'penalty': ['l2'], 'solver': ['lbfgs', 'liblinear', 'sag', 'saga'], 'C': [0.1, 1, 10]},
    {'penalty': ['elasticnet'], 'solver': ['saga'], 'C': [0.1, 1, 10], 'l1_ratio': [0.1, 0.5, 0.9]}
]
classifier_dt = DecisionTreeClassifier(random_state=0)
dt_param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': range(1, 20),
    'min_samples_split': range(2, 20),
    'min_samples_leaf': range(1, 10)
}

# Modellerin deðerlendirilmesi ve optimizasyonu
accuracy_knn, roc_auc_knn, f1_knn, precision_knn, recall_knn = evaluate_model(classKNN, "K-Nearest Neighbors")
accuracy_lr, roc_auc_lr, f1_lr, precision_lr, recall_lr = evaluate_model(classifier_lr, "Logistic Regression")
accuracy_dt, roc_auc_dt, f1_dt, precision_dt, recall_dt = evaluate_model(classifier_dt, "Decision Tree")

opt_accuracy_knn, opt_roc_auc_knn, opt_f1_knn, opt_precision_knn, opt_recall_knn = optimize_with_grid_search(classKNN, knn_param_grid, "K-Nearest Neighbors")
opt_accuracy_lr, opt_roc_auc_lr, opt_f1_lr, opt_precision_lr, opt_recall_lr = optimize_with_grid_search(classifier_lr, lr_param_grid, "Logistic Regression")
opt_accuracy_dt, opt_roc_auc_dt, opt_f1_dt, opt_precision_dt, opt_recall_dt = optimize_with_grid_search(classifier_dt, dt_param_grid, "Decision Tree")

# En iyi modelin seçimi
best_model_name = ""
best_metric = 0
best_metrics = {}

for model_name, metrics in {
    "K-Nearest Neighbors": (opt_accuracy_knn, opt_roc_auc_knn, opt_f1_knn, opt_precision_knn, opt_recall_knn),
    "Logistic Regression": (opt_accuracy_lr, opt_roc_auc_lr, opt_f1_lr, opt_precision_lr, opt_recall_lr),
    "Decision Tree": (opt_accuracy_dt, opt_roc_auc_dt, opt_f1_dt, opt_precision_dt, opt_recall_dt)
}.items():
    if metrics[0] > best_metric:
        best_metric = metrics[0]
        best_model_name = model_name
        best_metrics = metrics

print(f"Bu veri seti için en iyi model: {best_model_name}")
print(f"Doðruluk: {best_metrics[0]:.2%}, ROC AUC: {best_metrics[1]:.2%}, F1: {best_metrics[2]:.2%}, Precision: {best_metrics[3]:.2%}, Recall: {best_metrics[4]:.2%}")
