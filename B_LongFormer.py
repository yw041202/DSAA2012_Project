import re
import os
import torch
import string
import pandas as pd
import numpy as np
from unidecode import unidecode
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from transformers import LongformerTokenizer, LongformerForSequenceClassification, Trainer, TrainingArguments

# Load dataset
df = pd.read_csv('test.csv')

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
print("数据预处理完成。")

df[['Post_processed', 'Label_encoded']].to_csv('./data_for_longformer/processed_data.csv', index=False)

# Split dataset
X = df['Post_processed'].tolist()
y = df['Label_encoded'].tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Longformer Tokenizer and Dataset
MODEL_NAME = 'allenai/longformer-base-4096'
print(f"正在加载Longformer Tokenizer: {MODEL_NAME}...")
tokenizer = LongformerTokenizer.from_pretrained(MODEL_NAME)

# Tokenization and Dataset Preparation
class SuicideDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=1024):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_dataset = SuicideDataset(X_train, y_train, tokenizer)
test_dataset = SuicideDataset(X_test, y_test, tokenizer)

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

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, fscore, oe = gr_metrics(preds, labels)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'FScore': fscore,
        'Precision': precision,
        'Recall': recall,
        'OE': oe
    }

# Train Longformer Model
print(f"正在加载Longformer 模型: {MODEL_NAME}...")
model = LongformerForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label_map))

training_args = TrainingArguments(
    output_dir='./longformer_baseline',
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./longformer_baseline/logs',
    logging_steps=10,
    eval_strategy="epoch", # evaluate at the end of each epoch
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Start Training
print("开始模型训练...")
trainer.train()
print("模型训练完成。")


# Evaluation
print("正在进行最终评估...")
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = y_test


precision, recall, fscore, oe = gr_metrics(y_pred, y_true)
conf_matrix = confusion_matrix(y_true, y_pred)

metrics_dict = {
    'Precision': [precision],
    'Recall': [recall],
    'FScore': [fscore],
    'OE': [oe]
}

print("\n--- Longformer基线模型评估指标 ---")
print(metrics_dict)
print(conf_matrix)

md_save_path = os.path.join(results_dir, 'Longformer_metrics.csv')
cm_save_path = os.path.join(results_dir, 'Longformer_confusion_matrix.csv')
pd.DataFrame(metrics_dict).to_csv(md_save_path, index=False)
pd.DataFrame(conf_matrix).to_csv(cm_save_path, index=False)

print("-----------------------------------")