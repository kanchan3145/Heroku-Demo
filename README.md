# Iris Classifier API

A tiny, ready-to-deploy ML model: a RandomForest classifier trained on the
classic Iris dataset, served through a Flask REST API. Test accuracy: ~90%.

## Files
- `train_model.py` – trains the model and saves it to `model.joblib`
- `app.py` – Flask API that loads `model.joblib` and serves predictions
- `model.joblib` – the trained model (already generated for you)
- `requirements.txt` – Python dependencies

## Run locally
```bash
pip install -r requirements.txt
python train_model.py   # optional, model.joblib is already included
python app.py
```
The API runs on `http://localhost:5000`.

### Test it
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```
Response:
```json
{"prediction": "setosa", "prediction_index": 0,
 "probabilities": {"setosa": 1.0, "versicolor": 0.0, "virginica": 0.0}}
```

Features, in order: sepal length (cm), sepal width (cm), petal length (cm), petal width (cm).

## Deploy options

### Render / Railway / Fly.io (easiest)
1. Push this folder to a GitHub repo.
2. Create a new "Web Service" and point it at the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app --bind 0.0.0.0:$PORT`

### Docker (works anywhere)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
```
```bash
docker build -t iris-api .
docker run -p 5000:5000 iris-api
```

### Heroku
Add a `Procfile` with:
```
web: gunicorn app:app
```
Then `git push heroku main`.

## Swap in your own data
Replace the dataset-loading section in `train_model.py` with your own
`X` (features) and `y` (labels), re-run it to regenerate `model.joblib`,
and update `feature_names`/`class_names` if needed — `app.py` doesn't
need any changes.
