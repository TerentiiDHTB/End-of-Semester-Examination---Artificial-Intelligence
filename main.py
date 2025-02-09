from fastapi import FastAPI
import pickle
from tensorflow import keras
import pandas as pd
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = None

with open("ai-model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

cnn_model = keras.models.load_model("ai-model/stocks_news_scoring.keras")

app = FastAPI()

@app.get("/check-model/accuracy")
def check_model():
    df = pd.read_csv("ai-model/prepared-data.csv", index_col=0)

    X_test_pad = pad_sequences(tokenizer.texts_to_sequences(df['summary']), maxlen=150)

    y_pred = cnn_model.predict(X_test_pad)

    y_pred_classes = (y_pred > 0.5).astype(int)

    return {"model_accuracy": accuracy_score(df['score'], y_pred_classes)}

@app.post("/analysis/stock-news")
def analysis_stock_news(stock: str):
    return {"analysis_result": str(cnn_model.predict(pad_sequences(tokenizer.texts_to_sequences([stock]), maxlen=150))[0][0])}
