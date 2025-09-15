import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Train a model and prepare metadata for logging
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

params = {
    "solver": "lbfgs",
    "max_iter": 1200,
    "multi_class": "auto",
    "random_state": 4242,
}
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Log the model and its metadata to MLflow
mlflow.set_tracking_uri(uri="http://localhost:8000")
mlflow.set_experiment("MLflow Quickstart")
with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)
    signature = infer_signature(X_train, lr.predict(X_train))
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        name="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )
    mlflow.set_logged_model_tags(
        model_info.model_id, {"Training Info": "Basic LR model for iris data"}
    )
    
# Load the model back for predictions as a generic Python Function model
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
# Load the model as a Python Function (pyfunc) and use it for inference
predictions = loaded_model.predict(X_test)
iris_feature_names = datasets.load_iris().feature_names
result = pd.DataFrame(X_test, columns=iris_feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions
result[:4]
