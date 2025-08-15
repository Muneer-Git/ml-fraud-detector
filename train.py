import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from joblib import dump

rng = np.random.default_rng(42)
n=6000
X = pd.DataFrame({
    "amount": rng.gamma(2., 50., n),
    "hour": rng.integers(0,24,n),
    "merchant_risk": rng.random(n),
    "distance_km": rng.gamma(2., 5., n),
})
y = ((X["amount"]>140) & (X["merchant_risk"]>0.7) | (X["distance_km"]>40)).astype(int)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
pipe = Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", LogisticRegression(max_iter=1000))])
pipe.fit(X_train, y_train)
print(classification_report(y_test, pipe.predict(X_test)))
dump(pipe, "model.joblib")
print("Saved model.joblib")
