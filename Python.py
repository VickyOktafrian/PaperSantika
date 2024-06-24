import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Fungsi untuk mencetak metrik evaluasi
def print_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    return accuracy, precision, recall, f1

# Contoh data (dummy dataset)
# Gantilah dengan dataset asli Anda
data = pd.DataFrame({
    'feature1': np.random.rand(1000),
    'feature2': np.random.rand(1000),
    'text': np.random.choice(['example threat', 'normal text', 'suspicious activity'], 1000),
    'label': np.random.choice([0, 1], 1000)
})

# Pemrosesan data teks menggunakan TF-IDF
vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(data['text'])

# Memisahkan fitur dan label
X = data[['feature1', 'feature2']]
X = np.hstack((X, X_text.toarray()))
y = data['label']

# Membagi data menjadi train dan test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Jaringan Saraf Tiruan
print("Training Neural Network...")
nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
nn.fit(X_train, y_train)
y_pred_nn = nn.predict(X_test)
print("Neural Network Metrics:")
nn_metrics = print_metrics(y_test, y_pred_nn)

# Pembelajaran Mendalam
print("\nTraining Deep Learning Model...")
dl_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dl_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
y_pred_dl = (dl_model.predict(X_test) > 0.5).astype("int32")
print("Deep Learning Metrics:")
dl_metrics = print_metrics(y_test, y_pred_dl)

# Pemrosesan Bahasa Alami (NLP) dengan Multinomial Naive Bayes
print("\nTraining NLP Model...")
X_text_train, X_text_test, y_text_train, y_text_test = train_test_split(X_text, y, test_size=0.2, random_state=42)
nb = MultinomialNB()
nb.fit(X_text_train, y_text_train)
y_pred_nb = nb.predict(X_text_test)
print("NLP Model Metrics:")
nb_metrics = print_metrics(y_text_test, y_pred_nb)

# Menampilkan hasil komparasi
results = pd.DataFrame({
    'Algorithm': ['Neural Network', 'Deep Learning', 'NLP'],
    'Accuracy': [nn_metrics[0], dl_metrics[0], nb_metrics[0]],
    'Precision': [nn_metrics[1], dl_metrics[1], nb_metrics[1]],
    'Recall': [nn_metrics[2], dl_metrics[2], nb_metrics[2]],
    'F1-Score': [nn_metrics[3], dl_metrics[3], nb_metrics[3]]
})

print("\nComparison of Algorithms:")
print(results)
