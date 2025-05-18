# normalization compare
# @author: weidongfeng
# @date: 2025-05-18
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
iris = load_iris()
x,y=iris.data,iris.target
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)
scalers = {
    "standard": StandardScaler(),
    "minmax": MinMaxScaler(),
    "robust": RobustScaler()
}

for scaler_name,scaler in scalers.items():
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    print("Accuracy for", scaler_name, ":", accuracy_score(y_test, predictions))