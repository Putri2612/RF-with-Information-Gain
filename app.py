#import modul
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.preprocessing import LabelEncoder
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import streamlit as st
from streamlit_option_menu import option_menu
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import pickle  # Untuk memuat model Anda
from sklearn.feature_extraction.text import TfidfVectorizer  # Untuk mengubah teks menjadi representasi numerik
import re
import string


# Download stopwords and punkt if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Fungsi untuk memuat data normalisasi
@st.cache_data()
def load_normalized_word():
    normalized_word = pd.read_excel("normalisasi.xlsx")
    normalized_word_dict = {}

    for index, row in normalized_word.iterrows():
        if row[0] not in normalized_word_dict:
            normalized_word_dict[row[0]] = row[1]
    return normalized_word_dict

normalized_word_dict = load_normalized_word()

def remove_punct (text):
  #Remove spasi diawal dan akhir, url, hastag, tagar, kata akhiran berlebihan, baris baru, tab
  text = re.sub("^\s+|\s+$", "", text)
  text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text) #url
  text = re.sub(r"@\w+"," ", text)  # Menghapus tag (@) beserta kata-kata
  text = re.sub(r"#\w+","", text)  # Menghapus hashtag (#) beserta kata-kata
  text = text.replace('\\t', " ").replace('\n', " ").replace('\\u'," ").replace('\\'," ")
  #Remove Karakter ASCII, angka, punctuation
  text = text.encode('ascii', 'replace').decode('ascii')
  text = re.sub('x(\d+[a-zA-Z]+|[a-zA-Z]+\d+|\d+])',"",text)
  text = re.sub('[0-9]+', '', text)
  translator = str.maketrans(string.punctuation, ' '* len(string.punctuation))
  text = text.translate(translator)
  return text

# Fungsi untuk normalisasi term
def normalized_term(document, normalized_word_dict):
    if isinstance(document, str):
        return ' '.join([normalized_word_dict[term] if term in normalized_word_dict else term for term in document.split()])

# Fungsi untuk tokenisasi
def tokenization(text):
    text = nltk.tokenize.word_tokenize(text)
    return text

# Fungsi untuk penghapusan stopwords
def stopwords_removal(words):
    list_stopwords = set(nltk.corpus.stopwords.words('indonesian'))
    with open('stopword.txt', 'r') as file:
        for line in file:
            line = line.strip()
            list_stopwords.add(line)

    hapus = {"tidak", "naik", "kenaikan"}
    for i in hapus:
        if i in list_stopwords:
            list_stopwords.remove(i)

    return [word for word in words if word not in list_stopwords]

# Inisialisasi objek Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi untuk stemming
def stemming(text):
    text = [stemmer.stem(word) for word in text]
    return text

@st.cache_data()
def vectorize_data(data):
    tfidf = TfidfVectorizer()
    x_tfidf = tfidf.fit_transform(data['processed_text'])
    feature_names = tfidf.get_feature_names_out()
    tfidf_df = pd.DataFrame(x_tfidf.toarray(), columns=feature_names)
    return tfidf_df

# Melakukan oversampling menggunakan SMOTE
@st.cache_data()
def apply_smote(tfidf_df, y):
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_smote, y_smote = smote.fit_resample(tfidf_df, y)
    return X_smote, y_smote

# Fungsi untuk memuat data
@st.cache_data()
def load_data():
    file_path = 'hasil_stemming.xlsx'
    data = pd.read_excel(file_path)
    return data, tfidf

# Fungsi untuk menghitung Information Gain
def compute_impurity(feature, impurity_criterion):
    probs = feature.value_counts(normalize=True)
    if impurity_criterion == 'entropy':
        impurity = -(np.sum(np.log2(probs) * probs))
    else:
        raise ValueError('Unknown impurity criterion')
    return impurity

# Fungsi untuk menghitung Information Gain
def compute_information_gain(df, target, descriptive_feature, split_criterion):
    target_entropy = compute_impurity(df[target], split_criterion)
    entropy_list = []
    weight_list = []

    for level in df[descriptive_feature].unique():
        df_feature_level = df[df[descriptive_feature] == level]
        entropy_level = compute_impurity(df_feature_level[target], split_criterion)
        entropy_list.append(entropy_level)
        weight_level = len(df_feature_level) / len(df)
        weight_list.append(weight_level)

    feature_remaining_impurity = np.sum(np.array(entropy_list) * np.array(weight_list))
    information_gain = target_entropy - feature_remaining_impurity
    return information_gain


# Fungsi untuk menghitung Information Gain untuk semua fitur
@st.cache_data()
def calculate_information_gains(X_smote_dftiga, y_smote_encoded):
    split_criterion = 'entropy'
    information_gains = {}

    for feature in X_smote_dftiga.columns:
        information_gain = compute_information_gain(
            pd.concat([X_smote_dftiga, pd.Series(y_smote_encoded, name='label')], axis=1),
            'label', feature, split_criterion)
        information_gains[feature] = information_gain

    return information_gains


# Inisialisasi model Random Forest
random_seed = 42
rf_model = RandomForestClassifier(random_state=random_seed)

# Definisikan daftar hyperparameter yang ingin Anda uji
param_grid = {
    'n_estimators': [111, 165, 255],
    'max_depth': [65, 77, 88],
    'min_samples_leaf': [1, 5, 10],
}

# Inisialisasi GridSearchCV dengan model dan parameter grid
grid_search_tiga = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)

# Navigasi sidebar
with st.sidebar:
    selected = option_menu('Analisis Sentimen', 
        ['Dashboard', 'Analysis Data', 'Classification', 'Report', 'Testing'],
        default_index=0)

# Halaman dashboard
if selected == 'Dashboard':
    st.title('Analisis Sentimen')
    st.write("by Putri Lailatul Maghfiroh")
    st.header("Analisis Sentimen Pada Ulasan Kenaikan BBM Dengan Penerapan Metode Random Forest dan Seleksi Fitur Information Gain")
    st.write("Sebuah aplikasi yang mampu mengklasifikasikan sentimen suatu ulasan dengan menggunakan metode Random Forest dan Seleksi fitur Information Gain. Data latih yang digunakan dalam sistem ini diambil dari Twitter dengan kata kunci 'BBM Naik'.")
    st.write("Akurasi dari sistem ini adalah 93.91%")

@st.cache_data()
def load_data_word_cloud():
    file_path = 'hasil_stemming.xlsx'
    data_word_cloud = pd.read_excel(file_path)
    return data_word_cloud

# Memuat data
data_word_cloud = load_data_word_cloud()


# Halaman analysis data
if selected == 'Analysis Data':
    st.title('Analysis Data')
    st.success("Halaman ini digunakan untuk melihat visualisasi data.")
    # Menambahkan select box untuk memilih tampilan
    tampilan = st.selectbox('Pilih Tampilan', ['Word Cloud', 'Analisis Data Sebelum dan Sesudah SMOTE'])

    if tampilan == 'Word Cloud':

        # Load data dari file Excel hanya saat aplikasi dimulai

        def generate_word_cloud(data_word_cloud, title, colormap):
            wordcloud = WordCloud(width=500, height=500, background_color='white', colormap=colormap).generate(' '.join(data_word_cloud['processed_text'].explode().astype(str)))
            st.text(title)
            st.image(wordcloud.to_image(), use_column_width=True)
        
        st.title('Word Cloud')

        # Grupkan data berdasarkan label
        data_negatif = data_word_cloud[data_word_cloud['label'] == 'negatif']
        data_netral = data_word_cloud[data_word_cloud['label'] == 'netral']
        data_positif = data_word_cloud[data_word_cloud['label'] == 'positif']

        # Buat tiga kolom sejajar
        col1, col2, col3 = st.columns(3)
        # Memanggil fungsi untuk menampilkan Word Cloud untuk setiap kelompok dalam tiga kolom
        with col1:
            generate_word_cloud(data_negatif, 'Negatif Data', 'Reds')
        with col2:
            generate_word_cloud(data_netral, 'Netral Data', 'Blues')
        with col3:
            generate_word_cloud(data_positif, 'Positif Data', 'YlGnBu')
        
    if tampilan == 'Analisis Data Sebelum dan Sesudah SMOTE':
        # Fit LabelEncoder di luar kondisional
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(data_word_cloud['label'])
        # Menghitung jumlah data pada setiap kelompok sebelum SMOTE
        jumlah_negatif_sebelum = len(data_word_cloud[data_word_cloud['label'] == 'negatif'])
        jumlah_netral_sebelum = len(data_word_cloud[data_word_cloud['label'] == 'netral'])
        jumlah_positif_sebelum = len(data_word_cloud[data_word_cloud['label'] == 'positif'])

        tfidf_df = vectorize_data(data_word_cloud)

        # Lakukan Label Encoding pada kolom target dan fit LabelEncoder
        @st.cache_data()
        def label_encoding(data):
            y = label_encoder.fit_transform(data['label'])
            return y

        y = label_encoding(data_word_cloud)

        X_smote, y_smote = apply_smote(tfidf_df, y)

        # Menghitung jumlah data sesudah SMOTE
        kelas_label = ['negatif', 'netral', 'positif']
        jumlah_data_sesudah = [np.sum(y_smote == label_encoder.transform([kelas])[0]) for kelas in kelas_label]

        # Gunakan hasilnya seperti yang Anda butuhkan
        jumlah_negatif_sesudah = jumlah_data_sesudah[0]
        jumlah_netral_sesudah = jumlah_data_sesudah[1]
        jumlah_positif_sesudah = jumlah_data_sesudah[2]

        # Data untuk pie chart
        labels_sebelum = ['Negatif', 'Netral', 'Positif']
        sizes_sebelum = [jumlah_negatif_sebelum, jumlah_netral_sebelum, jumlah_positif_sebelum]

        labels_sesudah = ['Negatif', 'Netral', 'Positif']
        sizes_sesudah = [jumlah_negatif_sesudah, jumlah_netral_sesudah, jumlah_positif_sesudah]

        # Membuat subplot pie chart
        st.title('Data Sebelum dan Sesudah SMOTE')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Pie chart sebelum SMOTE
        ax1.pie(sizes_sebelum, labels=labels_sebelum, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Sebelum SMOTE')

        # Pie chart sesudah SMOTE
        ax2.pie(sizes_sesudah, labels=labels_sesudah, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Sesudah SMOTE')

        # Menampilkan pie chart pada aplikasi Streamlit
        col1, col2 = st.columns(2)
        col1.pyplot(fig)

# Halaman Classification
if selected == 'Classification':
    st.title('Classification')
    st.success("Halaman ini untuk melihat proses mulai dari upload data, text preprocessing, tf-idf, hingga classification.")
    
    # Menambahkan upload file
    st.text("Masukkan sebuah file")
    
    # Menambahkan upload file
    uploaded_file = st.file_uploader("Upload file CSV atau Excel", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        st.success("File berhasil diunggah!")

        # Load data dari file yang diunggah
        data = pd.read_excel(uploaded_file)
        data.dropna(subset=['label'], inplace=True)
        data.drop_duplicates(subset='content', inplace=True)
        data['content'] = data['content'].str.lower()
        data['content'] = data['content'].apply(lambda x: remove_punct(x))

        # Normalisasi
        data['content'] = data['content'].apply(lambda x: normalized_term(x, normalized_word_dict))

        # Tokenization
        data['content'] = data['content'].apply(lambda x: tokenization(x))

        # Stopwords Removal
        data['content'] = data['content'].apply(lambda x: stopwords_removal(x))

        # Stemming
        data['stemmed_tokens'] = data['content'].apply(lambda x: stemming(x))
        data['processed_text'] = data['stemmed_tokens'].apply(lambda x: ' '.join(x))

        # TF-IDF Vectorization
        tfidf_df = vectorize_data(data)

        # Melatih model TF-IDF (contoh)
        tfidf = TfidfVectorizer()
        x_tfidf = tfidf.fit_transform(data['processed_text'])

        # Simpan model TF-IDF ke dalam file tfidf_model.pkl
        joblib.dump(tfidf, 'tfidf_model.pkl')

        # Label Encoding
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(data['label'])

        # Oversampling menggunakan SMOTE
        X_smote, y_smote = apply_smote(tfidf_df, y_encoded)

        # Jumlah data sebelum dan sesudah SMOTE
        jumlah_data_sebelum_smote = tfidf_df.shape[0]
        jumlah_data_setelah_smote = X_smote.shape[0]

        # Hitung Information Gain
        information_gains = calculate_information_gains(X_smote, y_smote)

        # Ambil fitur dengan Information Gain di atas ambang tertentu
        threshold = 0.004
        selected_features = [feature for feature, gain in information_gains.items() if gain > threshold]

        # Perbarui vektor fitur TF-IDF hanya untuk fitur yang terpilih
        X_selected = X_smote[selected_features]
        # Simpan X_selected dalam session
        if 'X_selected' not in st.session_state:
            st.session_state.X_selected = X_selected

        # Memisahkan fitur dan label
        X = X_selected
        y = y_smote

        # Membagi data menjadi train dan test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Inisialisasi model Random Forest
        random_seed = 42
        rf_model = RandomForestClassifier(random_state=random_seed)

        # Definisikan daftar hyperparameter yang ingin Anda uji
        param_grid = {
            'n_estimators': [111, 165, 255],
            'max_depth': [65, 77, 88],
            'min_samples_leaf': [1, 5, 10],
        }

        # Inisialisasi GridSearchCV dengan model dan parameter grid
        grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)

        # Melatih model dengan GridSearchCV
        grid_search.fit(X_train, y_train)

        # Menampilkan kombinasi hyperparameter terbaik yang ditemukan
        best_params = grid_search.best_params_

        # Melatih model dengan kombinasi hyperparameter terbaik
        best_rf_model = RandomForestClassifier(random_state=random_seed, **best_params)
        best_rf_model.fit(X_train, y_train)

        # Simpan model ke dalam file
        model_filename = 'best_rf_model_smote_tiga.pkl'
        joblib.dump(grid_search.best_estimator_, model_filename)

        # Prediksi menggunakan model terbaik
        y_pred = best_rf_model.predict(X_test)

        # Evaluasi model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        # Menampilkan hasil tahapan-tahapan
        st.subheader("Data Setelah Preprocessing:")
        st.dataframe(data.head())

        st.subheader("Hasil TF-IDF Vectorization:")
        st.dataframe(tfidf_df.head())

        st.subheader("Hasil Seleksi Fitur dengan Information Gain:")
        st.dataframe(X_selected.head())

        st.subheader("Hasil Pelatihan Model:")
        st.text(f"Kombinasi Hyperparameter Terbaik: {best_params}")
        st.text(f"Akurasi: {accuracy:.2f}")
        st.text(f"Presisi: {precision:.2f}")
        st.text(f"Recall: {recall:.2f}")
        st.text(f"F1-Score: {f1:.2f}")

# Halaman Report
if selected == 'Report':
    st.title('Report')
    st.success("Halaman ini untuk melihat Report dan Grafik Batang Tiap Model")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Accuracy/Time", "RF", "RF+IG 0,02", "RF+IG 0,002", "RF+IG 0,004", "RF+IG 0,006"])
    report_data = {
        "RF": {
            "accuracy": 0.93,
            "precision": [0.88, 0.93, 0.99],
            "recall": [0.93, 0.88, 0.98],
            "f1_score": [0.90, 0.91, 0.98],
            "processing_time": 1.33,
        },
        "RF+IG 0,02": {
            "accuracy": 0.92,
            "precision": [0.87, 0.90, 0.99],
            "recall": [0.90, 0.88, 0.97],
            "f1_score": [0.88, 0.89, 0.98],
            "processing_time": 0.79,
        },
        "RF+IG 0,002": {
            "accuracy": 0.93,
            "precision": [0.87, 0.94, 0.99],
            "recall": [0.94, 0.88, 0.97],
            "f1_score": [0.90, 0.91, 0.98],
            "processing_time": 1.25,
        },
        "RF+IG 0,004": {
            "accuracy": 0.94,
            "precision": [0.89, 0.95, 0.99],
            "recall": [0.94, 0.89, 0.99],
            "f1_score": [0.92, 0.92, 0.99],
            "processing_time": 0.63,
        },
        "RF+IG 0,006": {
            "accuracy": 0.93,
            "precision": [0.90, 0.93, 0.99],
            "recall": [0.92, 0.90, 0.99],
            "f1_score": [0.91, 0.91, 0.99],
            "processing_time": 1.04,
        },
    }

    # Tab untuk grafik line perbandingan akurasi dan waktu
    with tab1:
        st.subheader("Perbandingan Akurasi")
        plt.figure(figsize=(10, 6))
        accuracies = [0.93, 0.92, 0.93, 0.94, 0.93]
        labels = ["RF", "RF+IG 0,02", "RF+IG 0,002", "RF+IG 0,004", "RF+IG 0,006"]
        plt.plot(labels, accuracies, marker='o', label='Akurasi')
        plt.xlabel("Model")
        plt.ylabel("Akurasi (%)")  # Ubah label sumbu y
        plt.title("Perbandingan Akurasi antara Model")
        plt.grid(True)
        plt.legend()

        # Tambahkan label angka di atas garis dalam format persen
        for x, y in zip(labels, accuracies):
            plt.text(x, y, f'{y*100:.0f}%', ha='center', va='bottom')  # Ubah format angka ke persen


        st.pyplot(plt)

        st.subheader("Perbandingan Waktu Proses")
        plt.figure(figsize=(10, 6))
        plt.plot(["RF", "RF+IG 0,02", "RF+IG 0,002", "RF+IG 0,004", "RF+IG 0,006"], [1.00, 0.79, 1.25, 0.63, 1.04], marker='o', label='Waktu Proses (detik)', color='orange')
        plt.xlabel("Model")
        plt.ylabel("Waktu Proses (second)")
        plt.title("Perbandingan Waktu Proses antara Model")
        plt.grid(True)
        plt.legend()

        # Tambahkan label angka di atas garis
        for x, y in zip(["RF", "RF+IG 0,02", "RF+IG 0,002", "RF+IG 0,004", "RF+IG 0,006"], [1.00, 0.79, 1.25, 0.63, 1.04]):
            plt.text(x, y, f'{y:.2f}s', ha='center', va='bottom')

        st.pyplot(plt)

    # Tab untuk grafik batang report RF tanpa IG
    with tab2:
        st.subheader("Grafik Batang - RF Tanpa IG")
        accuracy_rf = report_data["RF"]["accuracy"] * 100  # Ubah ke persen
        precision_rf = [value * 100 for value in report_data["RF"]["precision"]]  # Ubah ke persen
        recall_rf = [value * 100 for value in report_data["RF"]["recall"]]  # Ubah ke persen
        f1_rf = [value * 100 for value in report_data["RF"]["f1_score"]]  # Ubah ke persen

        fig, ax = plt.subplots(figsize=(10, 6))
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        values = [accuracy_rf, precision_rf[0], recall_rf[0], f1_rf[0]]
        bars = ax.bar(metrics, values, color=['blue', 'green', 'red', 'purple'])
        ax.set_ylabel('Score')
        ax.set_title('Grafik Batang - RF Tanpa IG')
        
         # Tambahkan label angka di atas batang
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{value:.2f}%',  # Format nilai ke persen
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        st.pyplot(fig)

    # Tab untuk grafik batang report RF + IG 0,02
    with tab3:
        st.subheader("Grafik Batang - RF + IG 0,02")
        accuracy_rf_002 = report_data["RF+IG 0,02"]["accuracy"]*100
        precision_rf_002 = [value * 100 for value in report_data["RF+IG 0,02"]["precision"]]
        recall_rf_002 = [value * 100 for value in report_data["RF+IG 0,02"]["recall"]]
        f1_rf_002 = [value * 100 for value in report_data["RF+IG 0,02"]["f1_score"]]

        fig, ax = plt.subplots(figsize=(10, 6))
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        values = [accuracy_rf_002, precision_rf_002[0], recall_rf_002[0], f1_rf_002[0]]
        bars = ax.bar(metrics, values, color=['blue', 'green', 'red', 'purple'])
        ax.set_ylabel('Score')
        ax.set_title('Grafik Batang - RF + IG 0,02')
        
        # Tambahkan label angka di atas batang
        for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{value:.2f}%',  # Format nilai ke persen
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        st.pyplot(fig)

    # Tab untuk grafik batang report RF + IG 0,002
    with tab4:
        st.subheader("Grafik Batang - RF + IG 0,002")
        accuracy_rf_002 = report_data["RF+IG 0,002"]["accuracy"]*100
        precision_rf_002 = [value * 100 for value in report_data["RF+IG 0,002"]["precision"]]
        recall_rf_002 = [value * 100 for value in report_data["RF+IG 0,002"]["recall"]]
        f1_rf_002 = [value * 100 for value in report_data["RF+IG 0,002"]["f1_score"]]

        fig, ax = plt.subplots(figsize=(10, 6))
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        values = [accuracy_rf_002, precision_rf_002[0], recall_rf_002[0], f1_rf_002[0]]
        bars = ax.bar(metrics, values, color=['blue', 'green', 'red', 'purple'])
        ax.set_ylabel('Score')
        ax.set_title('Grafik Batang - RF + IG 0,002')
        
        # Tambahkan label angka di atas batang
        for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{value:.2f}%',  # Format nilai ke persen
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        st.pyplot(fig)

    # Tab untuk grafik batang report RF + IG 0,004
    with tab5:
        st.subheader("Grafik Batang - RF + IG 0,004")
        accuracy_rf_004 = report_data["RF+IG 0,004"]["accuracy"]*100
        precision_rf_004 = [value * 100 for value in report_data["RF+IG 0,004"]["precision"]]
        recall_rf_004 = [value * 100 for value in report_data["RF+IG 0,004"]["recall"]]
        f1_rf_004 = [value * 100 for value in report_data["RF+IG 0,004"]["f1_score"]]

        fig, ax = plt.subplots(figsize=(10, 6))
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        values = [accuracy_rf_004, precision_rf_004[0], recall_rf_004[0], f1_rf_004[0]]
        bars = ax.bar(metrics, values, color=['blue', 'green', 'red', 'purple'])
        ax.set_ylabel('Score')
        ax.set_title('Grafik Batang - RF + IG 0,004')
        
        # Tambahkan label angka di atas batang
        for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{value:.2f}%',  # Format nilai ke persen
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        st.pyplot(fig)

    # Tab untuk grafik batang report RF + IG 0,006
    with tab6:
        st.subheader("Grafik Batang - RF + IG 0,006")
        accuracy_rf_006 = report_data["RF+IG 0,006"]["accuracy"]*100
        precision_rf_006 = [value * 100 for value in report_data["RF+IG 0,006"]["precision"]]
        recall_rf_006 = [value * 100 for value in report_data["RF+IG 0,006"]["recall"]]
        f1_rf_006 = [value * 100 for value in report_data["RF+IG 0,006"]["f1_score"]]

        fig, ax = plt.subplots(figsize=(10, 6))
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        values = [accuracy_rf_006, precision_rf_006[0], recall_rf_006[0], f1_rf_006[0]]
        bars = ax.bar(metrics, values, color=['blue', 'green', 'red', 'purple'])
        ax.set_ylabel('Score')
        ax.set_title('Grafik Batang - RF + IG 0,006')
        
        # Tambahkan label angka di atas batang
        for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{value:.2f}%',  # Format nilai ke persen
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        st.pyplot(fig)

# ...

# Halaman "Testing"
if selected == 'Testing':
    st.title('Testing')
    st.success("Masukkan kalimat yang akan diuji sentimennya.")

    # Form untuk menginput kalimat baru
    input_text = st.text_area("Masukkan Kalimat:", "")

    if st.button("Uji Sentimen"):
        
        tfidf = TfidfVectorizer()

       # Memuat data dan model TF-IDF dari fungsi load_data
        data, tfidf_model = load_data()

        # Fungsi untuk melakukan preprocessing teks
        def preprocess_text(text):
            text = text.lower()
            text = remove_punct(text)
            text = normalized_term(text, normalized_word_dict)
            text = tokenization(text)
            text = stopwords_removal(text)
            text = stemming(text)
            return ' '.join(text)


       # Load the model and TfidfVectorizer
        loaded_rf_model = joblib.load('trained_rf_model.joblib')
        selected_features = joblib.load('selected_features.joblib')

        # Contoh data uji yang diunggah oleh pengguna
        new_data = [input_text]

        # Lakukan preprocessing pada data uji
        new_data[0] = preprocess_text(new_data[0])

        # Gunakan TfidfVectorizer yang telah dimuat ulang dan hanya untuk fitur yang dipilih
        tfidf = TfidfVectorizer(max_features=841)
        tfidf.fit(selected_features)
        # Vektorisasi teks baru menggunakan objek TF-IDF yang telah Anda muat kembali
        new_data_vector = tfidf.transform(new_data)

            # Lakukan prediksi dengan model yang dimuat
        predictions = loaded_rf_model.predict(new_data_vector)

            # Tampilkan hasil prediksi
        st.subheader("Hasil Prediksi:")
        if predictions[0] == 0:
            st.write("Sentimen: Negatif")
        elif predictions[0] == 1:
            st.write("Sentimen: Netral")
        else:
            st.write("Sentimen: Positif")
