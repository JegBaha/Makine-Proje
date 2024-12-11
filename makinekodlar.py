import pandas as panda
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns


##�n i�lenmi� verimizi dataSet verisine at�yoruz(Normalizasyon, dump veriler, duplicate verileri d�zenlenmi� bir �ekilde) 
dataSet = panda.read_csv('C:/Users/bahab/OneDrive/Masa�st�/heart_normalized.csv')
## veri setinin sat�r ve s�tun say�s�n� bir tuple ;(demet) olarak d�ner.
dataSet.shape



##deep=True veri setinin ba��ms�z bir kopyas�n� olu�turur. B�ylece, dataCopy(test veri seti gibi d���n�lebilir ama model �zerindeki test de�il) �zerinde yap�lan de�i�iklikler data'y� etkilemez.
dataCopy = dataSet.copy(deep = True)

correlation = dataCopy[['RestingBP', 'RestingECG']].corr()

print("Korelasyon Matrisi:")
print(correlation)



#yaava�tan model e�itim s�re�leri
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_score
#�apraz do�rulama (Cross-Validation) kullanarak her kombinasyonu test eder.
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
## roc e�ri f1 skore de��imler i�in grafik olu�turmaya sa�lar 
from sklearn.model_selection import RepeatedStratifiedKFold
##pre ve recall e�ricsi �izmek i�in kullan�l�r.
from sklearn.metrics import precision_recall_curve


## Ana veriyi de�i�tirmemek ad�na copyalanm�� verideki �znitelikler features de�erine at�ld�.
## Target �zniteliklerini listeden ��kart�yoruyz ��nk� data leakage yaratmas�n diye. 
features = dataCopy[dataCopy.columns.drop(['HeartDisease'])].values
## Modelin tahmin etmeye �al��aca�� de�er.
target = dataCopy['HeartDisease'].values
## features ve target verilerini e�itim ve test setlerine b�ler.
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.20, random_state = 2)
## test_size = 0.20: Verinin %20'ini test seti olarak ay�r�r, geri kalan %80'i ise e�itim seti olarak kullan�l�r.
## Veriyi b�lerken rastgelelik uygulan�r, ancak bu parametre sabit bir de�er (2) ile belirlenmi�, yani her �al��t�r�ld���nda ayn� veri b�l�m� elde edilir. Bu, tekrar edilebilirlik sa�lar.
## x_train: E�itim verisinin �znitelikleri (features) (e�itim seti).
## x_test: Test verisinin �znitelikleri (features) (test seti).
## y_train: E�itim verisinin hedef de�i�keni (HeartDisease) (e�itim seti).
## y_test: Test verisinin hedef de�i�keni (HeartDisease) (test seti).

def modelsThing(classifier):
    ## .fit sayesinde model, verilerin ve hedeflerin ili�kisini ��renmeye ba�lar.
    classifier.fit(x_train,y_train)
    ## predict sayesinde e�itilen veri tahminde bulunmaya ba�lar bizde burada e�itim verisinin �zniteliklerini de�er olarak veriyoruz.
    prediction = classifier.predict(x_test)
    

    ## �apraz do�rulama y�netimidir. Amac� veri setini verdi�imiz katman(folds) (buradaki n_splits) katsay�s�na b�ler ve birden fazla kez tekrar etmesini sa�lar(n_repeats)
    # Neden yapar dersek asl�nda bir nevi veri seti dengeli ise bize genelle�tirmesi daha iyi olan overfittingden uzak daha iyi bir sonu� vermesi i�in �al���r.  
    #random_state=1 verilmesinin sebebi katlama ve b�l�mlerin ayn� �ekilde olu�mas�n� sa�lar. Verilmez ise her �al��t�rmada farkl� bir sonu� elde edilebilr.
    cv = RepeatedStratifiedKFold(n_splits = 10,n_repeats = 5,random_state = 1)
    
    ## Bunlar classification_report k�t�phanesinde dahil de�ildi ek olrk farkl� k�t�phane �zerinden yapt�m.
    ### Anlaml� 2 basamak i�in b�yle bir kod yaz�ld�; 3.141593 yerine 3.14 yaz�lacak '{0:.2%}
    print("Do�ruluk : ",'{0:.2%}'.format(accuracy_score(y_test,prediction)))
    
    #�apraz do�rulamna yapmaya karar verdik sebebi �apraz do�rulama da verilen metrikler ile normal classification reports lar ile 
    #e� de�er ��kacak m� veya yak�n bir de�er ��kacak m� diye ek olarak kontrol yapmam�za olanak sa�lad��� i�in
    #.mean() sebebi bize bir array d�nd�rd�kleri i�in ve bize say�sal bir veri laz�m oldugundan dolay� ortalamal�r�n� bizim i�in al�p bir de�er d�nd�r�r.
    
    print("�apraz Do�rulama(RocAuc) : ",'{0:.2%}'.format(cross_val_score(classifier,x_train,y_train,cv = cv,scoring = 'roc_auc').mean()))
    print("�apraz Do�rulama(Accuracy(Do�ruluk)) : ",'{0:.2%}'.format(cross_val_score(classifier,x_train,y_train,cv = cv,scoring = 'accuracy').mean()))
    print("�apraz Do�rulama(F1)) : ",'{0:.2%}'.format(cross_val_score(classifier,x_train,y_train,cv = cv,scoring = 'f1').mean()))
    print("�apraz Do�rulama(Precision(Kesinlik) : ",'{0:.2%}'.format(cross_val_score(classifier,x_train,y_train,cv = cv,scoring = 'precision').mean()))
    print("ROC_AUC  : ",'{0:.2%}'.format(roc_auc_score(y_test,prediction)))
    
    # modelin s�n�fland�rma (classification) performans�n� de�erlendirmek i�in precision, recall f1-score burada hesaplan�r, Support veri setindeki �rnek say�s�d�r.
    # support genelde 0 a farkl� bir say� 1 e fark�l� bir say� verir ve toplamlar� genel �rnek say�m�z�n test setindeki verdi�imis y�zdelik ile orant�l�dr.
    print(classification_report(y_test,classifier.predict(x_test)))


#Macro Precision = (0.88 + 0.86) / 2 = 0.87
#Macro Recall = (0.85 + 0.87) / 2 = 0.86
#Macro F1-Score = (0.86 + 0.86) / 2 = 0.86

#Weighted Precision = (Precision(S�n�f 1) * �rnek say�s�(S�n�f 1) + Precision(S�n�f 2) * �rnek say�s�(S�n�f 2)) / (Toplam �rnek say�s�)
#Weighted Recall = (Recall(S�n�f 1) * �rnek say�s�(S�n�f 1) + Recall(S�n�f 2) * �rnek say�s�(S�n�f 2)) / (Toplam �rnek say�s�)
#Weighted F1-Score = (F1(S�n�f 1) * �rnek say�s�(S�n�f 1) + F1(S�n�f 2) * �rnek say�s�(S�n�f 2)) / (Toplam �rnek say�s�)

    
from sklearn.neighbors import KNeighborsClassifier
## Manhattan uzakl�k �l��t� i�in
#K katsay�s� genel olarak baya denendi. 15 ten sonra belirli bir say�ya kadar d���� ya�and� 200 ler gibi say�lara geldi�i zaman %0.10 luk gibi bir do�ruluk
#artt�, fakat bu h�z� bozaca��ndan dolay� 15 ideal say� olarak d���n�lm��t�r.
#p=1 oldugu zaman manhattan 2 oldugu zaman �klid
#manhattan oldugu i�in metri�i de manhattan a ayarl�yoruz.
classKNN= KNeighborsClassifier( n_neighbors =15,p = 1,metric='manhattan')
modelsThing(classKNN)

## Manhattan uzakl�k �l��t� i�in
# weights=Kom�ular�n s�n�fland�rma �zerindeki a��rl�klar�n� belirler;
#'uniform' t�m kom�ular e�it a��rl�k,'distance' Daha yak�n kom�ular daha y�ksek a��rl�k
classKNN= KNeighborsClassifier(weights='distance', n_neighbors =15,p = 1,metric='manhattan')
modelsThing(classKNN)

## �klid uzakl�k �l��t� i�in
classKNN= KNeighborsClassifier( n_neighbors = 15,p = 2,metric="euclidean")
modelsThing(classKNN)

## �klid uzakl�k �l��t� i�in
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

# Veri setini y�kle

data = pd.read_csv('C:/Users/bahab/OneDrive/Masa�st�/heart_normalized.csv')

# �zellikler ve hedef de�i�keni ay�r
X = data.drop(columns=["HeartDisease"])  # Girdi �zellikleri
y = data["HeartDisease"]  # Hedef de�i�ken

# E�itim ve test setlerine ay�r
#test size 0,2= verinin %20'si test i�in kalan� e�itim i�in kullan�l�r
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar a�ac� modelini olu�tur ve e�it (en iyi hiperparametrelerle)

best_params = {'criterion': 'gini', 'max_depth': 5, 'min_samples_leaf': 4, 'min_samples_split': 5}
#b�l�nme kriteri gini,derinli�i 5 ,bir yaprak d���mde en az 4 �rnek olmal�,bir d���mde daha fazla dallanma yapmak i�in
#en az 5 �rnek gereklidir
clf = DecisionTreeClassifier(**best_params, random_state=42)
#karar a�ac� s�n�fland�r�c�. karar a�ac�n� olu�turup e�itim verisi ile e�itir (x ve y)
clf.fit(X_train, y_train)

# Tahmin yap 
y_pred = clf.predict(X_test)
#e�itilen model x test verileri �zerinden tahmin yapar

# Performans metrikleri
accuracy = accuracy_score(y_test, y_pred)
#do�ruluk oran� hesaplar y_test true positive di�eri true negatif toplan�r
classification_rep = classification_report(y_test, y_pred, output_dict=True)
#modelin do�rulu�u,precision,recall,f1 hesaplar ve s�zl�k olarak d�ner
conf_matrix = confusion_matrix(y_test, y_pred)
#confusion matrix olu�turur

#Performans sonu�lar�n� pandas DataFrame format�nda olu�tur
#results_df performans metriklerini dataframe olu�turur
#Metric Score ve Explanation sat�rlar�n� olu�turur
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
        "Genel do�ruluk oran� (t�m s�n�flar i�in)", 
        "Her s�n�f�n do�ruluk ortalamas�n�n hesaplanmas�", 
        "Her s�n�f�n recall (duyarl�l�k) ortalamas�", 
        "Her s�n�f�n F1-Skor ortalamas�",
        "Her s�n�f�n a��rl�kl� do�ruluk oran� (s�n�f �rnek say�s�na g�re a��rl�kl�)", 
        "Her s�n�f�n a��rl�kl� recall de�eri",
        "Her s�n�f�n a��rl�kl� F1-Skoru"
    ]
})

# Etiketlerin d�nd�r�lmesi
plt.figure(figsize=(25, 40))
plot_tree(
    clf, 
    feature_names=X.columns, 
    class_names=["No Heart Disease", "Heart Disease"], 
    filled=True
)

# Etiketleri d�nd�rme
plt.xticks(rotation=45)  # Etiketlerin yatayda d�nd�r�lmesi
plt.yticks(rotation=45)  # Etiketlerin dikeyde d�nd�r�lmesi

plt.title("Figure 1: Karar A�ac�")
plt.show()



# Performans tablosunu �iz (Figure 2)
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.axis('tight')
#axis eksen s�n�rlar�n� ayarlar
ax2.axis('off')
# x y eksenlerini gizler. 

# Tabloyu olu�tur
#performans sonu�lar�n� tablola�t�r�r
table2 = ax2.table(cellText=results_df.values, colLabels=results_df.columns, cellLoc='center', loc='center')

# 1. sat�r ve s�tunu koyu gri yapmak
for (i, j), cell in table2.get_celld().items():
    if i == 0 or j == 0:  # �lk sat�r ve ilk s�tun
        cell.set_text_props(weight='bold', color='white')  # Yaz�y� beyaz yap
        cell.set_facecolor('#4C4C4C')  # Koyu gri renk
    else:
        cell.set_text_props(weight='normal', color='black')  # Di�er h�creler normal
        cell.set_facecolor('white')  # Di�er h�creler beyaz

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

# Confusion Matrix'i �iz (Figure 3)
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
