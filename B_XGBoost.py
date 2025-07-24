# 传入XGBoost
import re
import os
import string
import pandas as pd
import numpy as np
import xgboost as xgb
from unidecode import unidecode
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv('500_Reddit_users_posts_labels.csv')

results_dir = os.path.join(os.getcwd(), 'results')

label_map = {'Supportive': 0, 'Indicator': 1, 'Ideation': 2, 'Behavior': 3, 'Attempt': 4}
df['Label_encoded'] = df['Label'].map(label_map)

# Data Preprocessing
stop_words = set(stopwords.words('english'))
words_to_remove = ['ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'don', "don't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'no', 'nor', 'not', 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]                 
stop_words.difference_update(words_to_remove)

def preprocess_text(text):
    # Remove Email and URL
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text) 
    
    # transform to lowercase
    text = text.lower()
    
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # remove accents
    text = unidecode(text)

    # remove stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    # remove extra whitespace
    text = text.strip()

    return text

df['Post_processed'] = df['Post'].apply(preprocess_text)

# Split dataset
X = df['Post_processed']
y = df['Label_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

print("正在将文本转换为TF-IDF向量...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words=None)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("TF-IDF转换完成。")

print("正在训练XGBoost模型...")
model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(label_map),
    use_label_encoder=False,
    eval_metric='mlogloss'
)
model.fit(X_train_tfidf, y_train)
print("模型训练完成。")

y_pred = model.predict(X_test_tfidf)

# Evaluation Metrics
def gr_metrics(pred, real):
    pred = np.array(pred)
    real = np.array(real)
    
    TP = (pred == real).sum()
    FN = (real > pred).sum()
    FP = (real < pred).sum()

    Precision = TP / (TP + FP) 
    Recall = TP / (TP + FN) 
    FS = 2 * Precision * Recall / (Precision + Recall) 
    
    OE = (np.abs(real - pred) > 1).sum()
    OE = OE / pred.shape[0] 

    return Precision, Recall, FS, OE

metrics_dict = {}
precision, recall, fscore, oe = gr_metrics(y_pred, y_test.to_numpy())

# get confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)


metrics_dict['Precision'] = [precision]
metrics_dict['Recall'] = [recall]
metrics_dict['FScore'] = [fscore]
metrics_dict['OE'] = [oe]

print("\n--- XGBoost基线模型评估指标 ---")
print(metrics_dict)
print(conf_matrix)

md_save_path = os.path.join(results_dir, 'XGBoost_metrics.csv')
cm_save_path = os.path.join(results_dir, 'XGBoost_confusion_matrix.csv')
pd.DataFrame(metrics_dict).to_csv(md_save_path, index=False)
pd.DataFrame(conf_matrix).to_csv(cm_save_path, index=False)
print("\nmetrics saved.")

print("---------------------------------")
