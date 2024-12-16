import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('C:/Users/Selin/PycharmProjects/pythonProject/heart.csv') #dosyayı açarız
print(data.isnull().sum()) #null veri kontrolü

duplicates = data[data.duplicated(keep=False)]
print(duplicates) #duplicate veri kontrolü
print(data.dtypes) #hangi verilerin kategorik olduğunu görmek için

le = LabelEncoder() # kategorik verileri nümeriğe çevirmek için

data['Sex'] = le.fit_transform(data['Sex'])
data['ChestPainType'] = le.fit_transform(data['ChestPainType'])
data['RestingECG'] = le.fit_transform(data['RestingECG'])
data['ExerciseAngina'] = le.fit_transform(data['ExerciseAngina'])
data['ST_Slope'] = le.fit_transform(data['ST_Slope'])

scaler = MinMaxScaler() #normalizasyon işlemleri içim
normallesecekler = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

data[normallesecekler] = scaler.fit_transform(data[normallesecekler]).round(2)
data.to_csv('C:/Users/Selin/PycharmProjects/pythonProject/heart_normalized.csv', index=False)
print(data) #veri ön işleme tamamlandı ve csv dosyasına kaydedildi