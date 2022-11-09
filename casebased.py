'''import file dan cek file'''
import pandas as pd
import matplotlib.pyplot as plt
file = pd.read_csv('Documents/audit_risk.csv')   #karena pada saat saya membuat program ini di linux, maka untuk direktorinya bisa disesuaikan seperti (./audit_risk.csv)
#print(file.head())
#print(file)

#menampilkan apa aja isi dari suatu kolom 
print("---ISI DARI KOLOM - RISK ---")
print(file['Risk'].unique())
print()

#menampilkan info dari dataset
print("---INFO DARI DATASET---")
print(file.info())
print()

#menampilkan data statistik dari file audit_risk.csv
print("---DATA STATISTIK DARI FILE YANG DIIMPORT---")
print(file.describe())
print()

#menampilkan berapa banyak (sum) dari suatu kolom
print("---MENAMPILKAN BANYAK DATA DARI SUATU KOLOM---")
print(file['Risk'].value_counts())
print()

#menampilkan histogram
plt.hist(file['Risk'], bins = 4, color = 'blue')
plt.title("count audit")
plt.show()




'''masuk preprocessing data'''
#untuk memisahkan label dan feature
feature = file.drop(['Risk', 'LOCATION_ID', 'PARA_A', 'Score_A', 'Risk_A', 'numbers', 'PROB', 'Risk_C', 'Risk_D', 'District_Loss', 'RiSk_E', 'History', 'Risk_F'], axis=1)
label = file[['Risk']]
feature = feature.iloc[:, :5]  #agar data yang diambil hanya 5 kolom pertama

print("---MENAMPILKAN 5 KOLOM PERTAMA DARI FEATURE---")
print(feature.head())  #agar data yang ditampilkan hanya 5 kolom pertama
print()

#normalisasi data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
print("---DATA FEATURE TELAH DINORMALISASI---")
feature = sc.fit_transform(feature)
print(feature[0:])
print()

#mengubah kelas dari integer menjadi biner
from sklearn.preprocessing import OneHotEncoder
oh = OneHotEncoder()
label = oh.fit_transform(label).toarray()
print("---KELAS DARI KOLOM RISK DIUBAH KE BINER---")
print(label[:3])
print()

#pemecahan data menjadi data test dan training
from sklearn.model_selection import train_test_split
feature_train, feature_test, label_train, label_test = train_test_split(feature,label,test_size = 0.1, random_state =0)
print("---DATA TELAH DIPECAH MENJADI TEST DAN TRAINING---")
print(feature_test.shape, label_test.shape)
print(feature_train.shape, label_train.shape)
print()



'''masuk ke ANN (Artificial Neural Network)'''
from keras.models import Sequential
from keras.layers import Dense, InputLayer


#deklarasi awal
model = Sequential()
model.add(InputLayer(input_shape=(5,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.summary()

#compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
training = model.fit(feature_train, label_train, validation_split = 0.1, shuffle = True, epochs = 50)  #melakukan training data
print()


'''masuk ke prediksi'''
print()

label_prediksi = model.predict(feature_test)
print(label_prediksi[0])
print(label_test[0])
print()

import numpy as np
print(np.argmax(label_prediksi[0]))
print(np.argmax(label_test[0]))
print()

#untuk membuat list nilai argumen maksimalnya
prediksi = []
for i in range (len(label_prediksi)):
    prediksi.append(np.argmax(label_prediksi[i]))

tes = []
for i in range (len(label_test)):
    tes.append(np.argmax(label_test[i]))

print("---NILAI ARGUMEN MAKSIMAL---")
print(prediksi[:2])
print(tes[:2])
print()


'''evaluasi data'''
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#untuk melihat nilai akurasinya
print("---NILAI AKURASI---")
print(accuracy_score(tes, prediksi))
print()

#untuk melihat hasil klasifikasi
print("---HASIL KLASIFIKASI---")
print(classification_report(tes, prediksi))
