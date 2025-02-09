import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
import re

url_regexp = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
number_regexp = r"^[-+]?[0-9]+$"

def remove_junk_from_summary(summary):
  summary = summary.lower()

  summary = re.sub(url_regexp, '', summary)
  summary = re.sub(number_regexp, '', summary)

  return summary

def move_title_to_news_summary(row):
   return row["title"] + row["summary"]

def serialiseScore(score):
    return 1 if score > 0 else 0

df = pd.read_csv("ai-model/data.tsv", sep="\t")

# print(df.head())

df.drop(["link", "published", "tickers"], axis=1, inplace=True)

# print(df.info())

#pd.set_option('display.max_colwidth', None)

# df[df['summary'].isnull()]['title']
#as I can see, 106, 150, 253, 342 news makes no sense without summary. So I can transfer titles of other news to summary column and delete title column.

ids = [20, 139, 213, 272, 303, 319, 334, 490]

df.loc[ids, 'summary'] = df.loc[ids, 'title']

df.dropna(inplace=True)

# df.info()
df['summary'] = df['summary'].apply(remove_junk_from_summary)

df['score'] = df['score'].apply(serialiseScore)

df["summary"] = df.apply(move_title_to_news_summary, axis=1)

df.drop('title', axis=1, inplace=True)

df.to_csv("ai-model/prepared-data.csv")

x, y = df["summary"], df["score"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(X_train)

with open("ai-model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=150)
X_test_pad = pad_sequences(X_test_seq, maxlen=150)

cnn_model = Sequential([
    Embedding(input_dim=5000, output_dim=128),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cnn_model.fit(X_train_pad, y_train, epochs=3, batch_size=32, validation_split=0.1)

y_pred = cnn_model.predict(X_test_pad)

y_pred_classes = (y_pred > 0.5).astype(int)

# Метрики
print("Accuracy:", accuracy_score(y_test, y_pred_classes))
print("Classification Report:\n", classification_report(y_test, y_pred_classes))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_classes))

cnn_model.save("ai-model/stocks_news_scoring.keras")