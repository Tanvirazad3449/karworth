# Karworth — Used Car Price Prediction (UK)

Karworth is an open-source web app and ML pipeline that predicts UK used-car prices using data from 1996–2020 (~100k records). It benchmarks classic regression models against a neural network and serves the best model via a lightweight Flask API. 

## Key Features
- **Models compared:** Linear Regression, Random Forest Regressor, Gradient Boosting Regressor, ANN (Keras).  
- **Best performer:** Random Forest (R² ≈ **0.96** on held-out test set).  
- **Predictors:** `brand`, `model`, `year`, `transmission`, `mileage`, `fuelType`, `mpg`, `engineSize`.  
- **Deployed app:** Simple form UI posting features to the Flask backend and returning an estimated resale price.  

---

## Project Structure
```
.
├─ data/                      # (optional) local CSVs; dataset is from Kaggle
├─ notebooks/                 # EDA, visualization, experiments
├─ src/
│  ├─ train.py                # trains models, evaluates R², saves best model
│  ├─ preprocess.py           # cleaning, OHE/label encoding, split
│  ├─ model_rf.pkl            # exported Random Forest model (created after training)
│  └─ utils.py                # helpers (metrics, plots, IO)
├─ app/
│  ├─ app.py                  # Flask server (GET form, POST /predict)
│  ├─ templates/
│  │  └─ index.html           # simple UI for inputs & result
│  └─ requirements.txt        # runtime deps for serving
├─ README.md
└─ LICENSE
```

> The repo uses a **train → export → serve** pattern: train/evaluate offline, persist the best model (`pickle`), then load it at runtime in the API.  

---

## Data
- **Source:** *100,000 UK Used Car Dataset* on Kaggle (1996–2020). Merge of manufacturer-wise CSVs.  
- **Target:** `price` (GBP)  
- **Cleaning highlights:** drop impossible years/engine sizes, unrealistic mpg, and missing rows (final ~98,377 records).  
- **Encoding:**  
  - OHE for tree/linear models on `brand`, `model`, `transmission`, `fuelType`  
  - Label encoding (for ANN) on categorical features  
- **Split:** 80% train, 10% val (for ANN), 10% test.  

---

## Results (R² on test set)
| Model                        | R² (≈) | Notes                          |
|-----------------------------|--------:|--------------------------------|
| Random Forest Regressor     | **0.96** | best generalization, few outliers |
| Gradient Boosting Regressor | 0.93    | good, some outliers            |
| Linear Regression           | 0.87    | negative preds, outliers       |
| ANN (ReLU, Adam)            | 0.84    | moderate, single negative pred |

*Random Forest chosen for deployment.*  

---

## Quickstart

### 1) Environment
```bash
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r app/requirements.txt
# If training locally:
pip install -r notebooks/requirements.txt  # or a top-level requirements-dev.txt
```

### 2) Train (optional)
If you want to reproduce or improve the model:
```bash
# Ensure data CSV(s) are available locally or fetched from Kaggle
python -m src.train --data data/used_cars_uk.csv --out src/model_rf.pkl
```
This script:
- loads & cleans data  
- encodes features  
- compares LR, RF, GBR, ANN  
- saves the best model to `src/model_rf.pkl` (default).  

### 3) Serve
```bash
export MODEL_PATH=src/model_rf.pkl  # path to the saved model
export FLASK_ENV=production
python app/app.py
# visit http://127.0.0.1:5000
```

---

## API

### `POST /predict`
**Body (JSON):**
```json
{
  "brand": "BMW",
  "model": "3 Series",
  "year": 2017,
  "transmission": "Automatic",
  "mileage": 45000,
  "fuelType": "Petrol",
  "mpg": 44.8,
  "engineSize": 2.0
}
```

**Response:**
```json
{ "predicted_price_gbp": 14520.34 }
```

---

## Development Notes
- **ANN**: 8→32→64→1 (ReLU), Adam optimizer, ~200 epochs, batch size 10; used 10% validation split. Designed for comparison, not deployment.  
- **Why Random Forest?** Robust to mixed feature types, handles non-linearities, minimal scaling, fast to train, and best R² in tests.  
- **Visualization**: scatter plots & correlation heatmap guided feature importance and cleaning.  

---

## Roadmap
- Add **XGBoost** / **CatBoost** benchmarks  
- Hyper-parameter search (Optuna)  
- Feature enrichment (accident history, options, safety index)  
- Dockerfile + CI (tests & lint)  
- Simple JavaScript client widget

---

## Citation
This implementation and documentation are based on:  
*“Comparison of Regression Models with Neural Network Models for Predicting Used Car Prices”* by Tanvir Azad.  

---

## License
MIT — see `LICENSE`.
