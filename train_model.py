import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = { 
    "comment":[
        "I love this"
        "This is amazing",
        "Very Good work",
        "Very bad experience",
        "I hate this",
    ],
    "label":[
        "positive"
        "positive",
        "positive",
        "negative",
        "negative",
        ]
}
df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["comment"])

model = LogisticRegression()
model.fit(X, df["label"])

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained and saved")
