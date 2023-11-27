import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
import numpy as np

st.title("PROYEK SAINS DATA")
st.write("Nama  : Jennatul Macwe ")
st.write("Nim   : 210411100151 ")
st.write("Kelas : Proyek Sains Data A ")

data_set_description, dataset, prepro, model, endgame = st.tabs(["Deskripsi Data Set", "Dataset", "Pre-processing", "Model", "tic-tac-toe endgame"])

with data_set_description:
    st.write("### Deskripsi Data Set")
    st.write("Dataset ini merupakan kumpulan lengkap konfigurasi papan di akhir permainan tic-tac-toe, di mana 'x' diasumsikan bermain terlebih dahulu. Konsep targetnya adalah 'kemenangan untuk x' (membuat 'tiga berjejer').")
    st.write("Dalam permainan Tic-Tac-Toe, ada dua pemain, biasanya disebut 'x' dan 'o,' yang bergiliran menempatkan tanda mereka di papan permainan 3x3. Tujuan permainan adalah mencapai 'win for x,' yang berarti pemain 'x' harus berhasil menciptakan salah satu dari 8 kemungkinan cara untuk mencapai 'three-in-a-row,' yaitu menempatkan tiga tanda 'x' secara berurutan dalam satu baris, kolom, atau diagonal.")
    st.write("Deskripsi tersebut mencatat bahwa Dataset ini mencakup semua konfigurasi papan pada akhir permainan Tic-Tac-Toe, dengan asumsi bahwa 'x' selalu bermain pertama.")
    st.write("Tujuan Dataset : Dataset ini digunakan untuk memprediksi apakah permainan akan dimenangkan oleh pemain 'x' atau tidak.")
    st.write("Ini adalah tugas klasifikasi di mana model pembelajaran mesin mencoba memprediksi hasil permainan, yaitu apakah 'x' akan memenangkan permainan atau tidak.")
    st.write("***Tipe Data: Categorical***")
    st.write("***Subject Area : Game***")
    st.write("***Jumlah data : 958 baris***")
    st.write("***Dataset ini TIDAK MEMILIKI Missing Values*** yang berarti bahwa setiap bagian data pada dataset lengkap dengan informasi.")
    st.write("#### Penjelasan Fitur")
    st.write("Jumlah Fitur : 9 (setiap fitur mempresentasikan setiap kotak pada permainan)")
    st.image('tictactoe.png', use_column_width=False, width=250)
    st.write("* top-left-square : kotak pada bagian **kiri** baris teratas")
    st.write("* top-middle-square : kotak pada bagian **tengah** baris teratas")
    st.write("* top-right-square : kotak pada bagian **kanan** baris teratas")
    st.write("* middle-left-square : kotak pada bagian **kiri** baris tengah")
    st.write("* middle-middle-square : kotak pada bagian **tengah** baris tengah")
    st.write("* middle-right-square : kotak pada bagian **kanan** baris tengah")
    st.write("* bottom-left-square : kotak pada bagian **kiri** baris bawah")
    st.write("* bottom-middle-square : kotak pada bagian **tengah** baris bawah")
    st.write("* bottom-righ-square : kotak pada bagian **kiri** baris bawah")

    st.write("### Sumber Dataset")
    st.write("https://archive.ics.uci.edu/dataset/101/tic+tac+toe+endgame")

with dataset:
    st.write("### Dataset Tic-ac-Toe Endgame")
    # Membaca data dari URL
    url = "https://raw.githubusercontent.com/jennamacwe/ProyekSainData/main/tic-tac-toe_Endgame2.csv"
    data = pd.read_csv(url, header=None)

    # Menambahkan kolom
    data.columns = ["top-left-square", "top-middle-square", "top-right-square", "middle-left-square", "middle-middle-square", "middle-right-square", "bottom-left-square", "bottom-middle-square", "bottom-right-square", "Class"]

    # Menampilkan dataset dengan kolom tambahan
    st.dataframe(data)

    st.write("#### Penjelasan Fitur")
    st.write("Jumlah Fitur : 9 (setiap fitur mempresentasikan setiap kotak pada permainan)")
    st.image('tictactoe.png', use_column_width=False, width=250)
    st.write("* top-left-square : kotak pada bagian **kiri** baris teratas")
    st.write("* top-middle-square : kotak pada bagian **tengah** baris teratas")
    st.write("* top-right-square : kotak pada bagian **kanan** baris teratas")
    st.write("* middle-left-square : kotak pada bagian **kiri** baris tengah")
    st.write("* middle-middle-square : kotak pada bagian **tengah** baris tengah")
    st.write("* middle-right-square : kotak pada bagian **kanan** baris tengah")
    st.write("* bottom-left-square : kotak pada bagian **kiri** baris bawah")
    st.write("* bottom-middle-square : kotak pada bagian **tengah** baris bawah")
    st.write("* bottom-righ-square : kotak pada bagian **kiri** baris bawah")

with prepro:
    st.write("### Dataset awal Endgame Permainan TicTacToe")

    # Membaca data dari URL
    url = "https://raw.githubusercontent.com/jennamacwe/ProyekSainData/main/tic-tac-toe_Endgame2.csv"
    data = pd.read_csv(url, header=None)

    # Menambahkan kolom
    data.columns = ["top-left-square", "top-middle-square", "top-right-square", "middle-left-square", "middle-middle-square", "middle-right-square", "bottom-left-square", "bottom-middle-square", "bottom-right-square", "Class"]

    # Menampilkan dataset dengan kolom tambahan
    st.dataframe(data)
    st.write("### Split Data")
    # Memisahkan fitur (X) dan target (y)
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Membagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=670, test_size=288, random_state=42)

    # Menampilkan data latih dan data uji
    st.write("#### Data Latih")
    st.dataframe(X_train)

    st.write("#### Data Uji")
    st.dataframe(X_test)

    st.write("### Mengubah categorical menjadi numerik menggunakan Label Encoding")
    st.write("Label encoding adalah suatu metode dalam pra-pemrosesan data yang melibatkan penggantian nilai-nilai kategori pada suatu fitur dengan nilai-nilai numerik yang unik.")
    st.write("Mengubah categorical menjadi numerik menggunakan Label Encoding dilakukan karena sebagian besar algoritma machine learning memerlukan input yang bersifat numerik. Beberapa algoritma, terutama yang berbasis pada perhitungan jarak (seperti k-Nearest Neighbors), dapat memberikan hasil yang lebih baik jika nilai kategorikal diubah menjadi bentuk numerik. Hal ini karena perhitungan jarak lebih mudah dilakukan pada data numerik.")
    st.write("x => 1")
    st.write("o => 0")
    st.write("b => -1")
    st.write("**catatan** : Karena fitur-fitur ini berada dalam rentang yang sangat terbatas (hanya 1, 0, dan -1), normalisasi tidak diperlukan. Normalisasi biasanya digunakan  pada data numerik yang memiliki rentang nilai yang berbeda agar fitur-fitur tersebut memiliki skala yang serupa.")

    # Mengimpor data CSV tanpa nama kolom (header)
    data = pd.read_csv(url, header=None)

    # Memberikan kolom pada dataset
    column_names = ["top-left-square", "top-middle-square", "top-right-square", "middle-left-square", "middle-middle-square", "middle-right-square", "bottom-left-square", "bottom-middle-square", "bottom-right-square", "Class"]
    data.columns = column_names

    # Mapping karakter ke angka (label encoding)
    char_to_num = {'x': 1, 'o': 0, 'b': -1}

    # Melakukan encoding untuk data train
    encoded_data_train = []

    # Mengonversi setiap baris dalam DataFrame X_train menjadi karakter dan menyimpannya dalam encoded_data_train
    for index, row in X_train.iterrows():
        encoded_row = [char_to_num[c] for c in row]
        encoded_data_train.append(encoded_row)

    # Membuat DataFrame dari data train yang telah diencode
    encoded_df_train = pd.DataFrame(encoded_data_train, columns=X_train.columns)

    # Menampilkan DataFrame data train yang sudah diencode
    st.write("#### Data Train yang sudah diencode:")
    st.dataframe(encoded_df_train)

    # Melakukan encoding untuk data test
    encoded_data_test = []

    # Mengonversi setiap baris dalam DataFrame X_test menjadi karakter dan menyimpannya dalam encoded_data_test
    for index, row in X_test.iterrows():
        encoded_row = [char_to_num[c] for c in row]
        encoded_data_test.append(encoded_row)

    # Membuat DataFrame dari data test yang telah diencode
    encoded_df_test = pd.DataFrame(encoded_data_test, columns=X_test.columns)

    # Menampilkan DataFrame data test yang sudah diencode
    st.write("#### Data Test yang sudah diencode:")
    st.dataframe(encoded_df_test)

    # st.write("Data Test dan Data Train yang sudah diencode ")

    # Menyimpan dataset yang telah di encoding ke csv dan pickle
    # File CSV
    encoded_df_train.to_csv("encoded_data_train.csv", index=False)
    encoded_df_test.to_csv("encoded_data_test.csv", index=False)

    # File pickle
    with open('encoded_data_train.pkl', 'wb') as file:
        pickle.dump(encoded_df_train, file)
        
    with open('encoded_data_test.pkl', 'wb') as file:
        pickle.dump(encoded_df_test, file)

with model:
    st.write("### Decission Tree")
    # Membangun model Decision Tree
    dt = DecisionTreeClassifier()
    dt.fit(encoded_df_train, y_train)

    # Melakukan prediksi
    y_pred = dt.predict(encoded_df_test)

    # Evaluasi model
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f'Akurasi Model: {accuracy:.2f}')

    # Menampilkan hasil perbandingan actual dan prediksi
    results_compare = pd.DataFrame({'Real Values': y_test, 'Prediksi': y_pred})
    st.write("Perbandingan Nilai yang Sesungguhnya dan Prediksi:")
    st.dataframe(results_compare)

    # Naive Bayes
    st.write("### Naive Bayes")
    # Inisialisasi model Naive Bayes Multinomial
    nb = GaussianNB()
    nb.fit(encoded_df_train, y_train)

    # Lakukan prediksi pada data uji
    y_pred = nb.predict(encoded_df_test)

    # Evaluasi model
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f'Akurasi Model Naive Bayes: {accuracy:.2f}')

    # Menampilkan hasil perbandingan actual dan prediksi
    results_nb = pd.DataFrame({'Real Values': y_test, 'Prediksi': y_pred})
    st.write("Perbandingan Nilai yang Sesungguhnya dan Prediksi:")
    st.dataframe(results_nb)

    # Membangun model K-NN
    st.write("### K-NN")
    k = 30
    acc = np.zeros((k - 1))

    for n in range(1, k, 2):
        knn = KNeighborsClassifier(n_neighbors=n, metric="euclidean").fit(encoded_df_train, y_train)
        y_pred = knn.predict(encoded_df_test)
        acc[n - 1] = accuracy_score(y_test, y_pred)

    best_accuracy = acc.max()
    best_k = acc.argmax() + 1

    st.write(f'Akurasi KNN terbaik adalah {best_accuracy:.2f} dengan nilai k = {best_k}')

    # Menampilkan hasil perbandingan actual dan prediksi K-NN
    results_knn = pd.DataFrame({'Real Values': y_test, 'Prediksi': y_pred})
    st.write("Perbandingan Nilai yang Sesungguhnya dan Prediksi:")
    st.dataframe(results_knn)

    # Membangun model Logistic Regression
    st.write("### Logistic Regression")
    # Inisialisasi model Logistic Regression
    logreg_model = LogisticRegression()

    # Melatih model menggunakan data train yang sudah diencode
    logreg_model.fit(encoded_df_train, y_train)

    # Membuat prediksi menggunakan data test yang sudah diencode
    y_pred = logreg_model.predict(encoded_df_test)

    st.write(f'Akurasi Model Logistic Regression: {accuracy:.2f}')

    # Menampilkan hasil perbandingan actual dan prediksi Logistic Regression
    results_lr = pd.DataFrame({'Real Values': y_test, 'Prediksi': y_pred})
    st.write("Perbandingan Nilai yang Sesungguhnya dan Prediksi:")
    st.dataframe(results_lr)    

with endgame:
    
    # Membangun model Decision Tree
    dt = DecisionTreeClassifier()
    dt.fit(encoded_df_train, y_train)

    # Input data untuk prediksi dari pengguna
    user_inputs = {}
    # for feature in encoded_df_train.columns:
    #     user_input = st.text_input(f"Masukkan nilai untuk '{feature}': ")
    #     user_inputs[feature] = char_to_num[user_input]
    
    for feature in X.columns:
        user_input = st.selectbox(f"'{feature}' ('x', 'o', 'b'):", ('x', 'o', 'b'))

        # Mengonversi nilai input pengguna menjadi numerik
        if user_input == 'x':
            user_inputs[feature] = char_to_num['x']
        elif user_input == 'o':
            user_inputs[feature] = char_to_num['o']
        elif user_input == 'b':
            user_inputs[feature] = char_to_num['b']


    # Membuat DataFrame dari input pengguna
    new_data = pd.DataFrame([user_inputs])

    # Menampilkan tombol untuk melakukan prediksi
    if st.button("Prediksi"):
        # Melakukan prediksi CLASS untuk data baru menggunakan model yang dimuat
        prediction = dt.predict(new_data)

        # Menampilkan hasil prediksi menggunakan Streamlit
        st.write("Hasil Prediksi:")
        if prediction[0] == 'positive':
            st.write("Positif (Menang)")
        else:
            st.write("Negatif (Kalah)")
            
    # # Melakukan prediksi CLASS untuk data baru menggunakan model yang dimuat
    # prediction = dt.predict(new_data)

    # # Menampilkan hasil prediksi
    # predicted_class = prediction[0]  # Menggunakan indeks 0 karena kita hanya memprediksi satu data

    # # Menampilkan hasil prediksi menggunakan Streamlit
    # st.write("Hasil Prediksi:")
    # if predicted_class == 'positive':
    #     st.write("Positif (Menang)")
    # else:
    #     st.write("Negatif (Kalah)")

    # Membaca DataFrame dari format pkl
    # with open('encoded_data.pkl', 'rb') as file:
    #     loaded_encoded_df = pickle.load(file)

    # # Memisahkan fitur (X) dan target (y)
    # X = loaded_encoded_df.drop('Class', axis=1)
    # y = loaded_encoded_df['Class']

    # # Membangun model Decision Tree
    # dt = DecisionTreeClassifier()
    # dt.fit(X, y)

    # # Tampilan judul
    # st.write("# Prediksi Akurasi Tic-Tac-Toe")

    # # Input data untuk 9 fitur
    # user_inputs = {}
    # for feature in X.columns:
    #     user_input = st.selectbox(f"'{feature}' ('x', 'o', 'b'):", ('x', 'o', 'b'))
    #     user_inputs[feature] = user_input

    # # Tombol prediksi
    # if st.button("Prediksi"):
    #     # Encoding input pengguna
    #     label_encoder = LabelEncoder()
    #     label_encoder.fit(['x', 'o', 'b'])
    #     encoded_inputs = [label_encoder.transform([user_inputs[feature]])[0] for feature in X.columns]

    #     # Melakukan prediksi
    #     y_pred = dt.predict([encoded_inputs])

    #     # Menampilkan hasil prediksi
    #     if y_pred[0] == 'positive':
    #         st.write("Hasil Prediksi: Positif (Menang)")
    #     else:
    #         st.write("Hasil Prediksi: Negatif (Kalah)")

    # Menampilkan akurasi model
    # st.write("Akurasi Model Decision Tree: ", f'{accuracy:.2%}')


# **catatan** : Karena fitur-fitur ini berada dalam rentang yang sangat terbatas 
# (hanya 1, 0, dan -1), normalisasi tidak diperlukan. Normalisasi biasanya digunakan 
# pada data numerik yang memiliki rentang nilai yang berbeda agar fitur-fitur tersebut 
# memiliki skala yang serupa.