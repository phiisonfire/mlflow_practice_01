0. setup the environment variable
`export MLFLOW_TRACKING_URI=http://localhost:5000`

1. start the mlflow server backend storage at port 5000 of localhost
`mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 5000`
Usage - Load model from the server into the program and do inference inside the program
```python
import mlflow.pyfunc
model_name = "iris-classifier"
stage = 'Production'
model = mlflow.sklearn.load_model(
    model_uri=f"models:/{model_name}/{stage}"
)
y_pred = model.predict([[6.7,3.3,5.7,2.1]])
print(y_pred) # ['Iris-virginica']
y_pred_prob = model.predict_proba([[6.7,3.3,5.7,2.1]]) # [[2.92676901e-05 4.04386355e-02 9.59532097e-01]]
print(y_pred_prob)
```
2. serving the model at port 1234 of localhost
`mlflow models serve --model-uri models:/iris-classifier/Production -p 1234 --no-conda`
Usage - Make a request to the Model Serving Server and receive the response
```python
# request a single sample
import requests
inference_request = {"dataframe_records": [[6.7,3.3,5.7,2.1]]}
endpoint = "http://localhost:1234/invocations"
response = requests.post(endpoint, json=inference_request)
print(response.text) # {"predictions": ["Iris-virginica"]}

# request a batch of samples
import requests
lst = X_test.values.tolist()
inference_request = {"dataframe_records": lst}
endpoint = "http://localhost:1234/invocations"
response = requests.post(endpoint, json=inference_request)
print(response) # <Response [200]>
print(response.text) # {"predictions": ["Iris-setosa", "Iris-setosa", "Iris-setosa", "Iris-setosa", ...]}
```