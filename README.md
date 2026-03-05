
````md
# Weather ML Notebook (Classification + Regression + LSTM)

This project explores a weather dataset and trains multiple models for:

- **Classification:** Predicting **Precip Type** (rain vs snow)
- **Regression:** Predicting **Temperature (C)** using classic ML
- **Time-series Regression (Deep Learning)** : Predicting **Temperature (C)** using an **LSTM** model

All experiments are contained in the notebook: **`IC_ML_01.ipynb`**.

---

## Project Files

- `IC_ML_01.ipynb` — Main notebook (EDA + preprocessing + models + evaluation + plots)
- `weatherHistory.csv` — Weather dataset used by the notebook

---

## Dataset Overview

The dataset contains hourly weather observations with columns such as:

- `Formatted Date`
- `Summary`
- `Precip Type`
- `Temperature (C)`
- `Apparent Temperature (C)`
- `Humidity`
- `Wind Speed (km/h)`
- `Wind Bearing (degrees)`
- `Visibility (km)`
- `Pressure (millibars)`
- (and others like `Loud Cover`, `Daily Summary`)

---

## Notebook Workflow (What happens inside)

### 1) Load Data
The notebook loads data using:
```python
weather = pd.read_csv(r'weatherHistory.csv')
````

### 2) Cleaning + Feature Engineering

* Converts `Formatted Date` to datetime
* Extracts:

  * `year`, `month`, `day`, `hour`
* Drops columns that are not needed:

  * `Loud Cover`
  * `Daily Summary`
* Handles missing values in `Precip Type` (fills missing as `"rain"`)
* Encodes `Precip Type`:

  * `rain -> 1`
  * `snow -> 0`

### 3) Outlier Check (Wind Speed)

The notebook checks outliers visually with a boxplot and decides to keep them.

### 4) Classification Task (Precip Type)

Target:

* `y_c = weather['Precip Type']`

Features:

* `x_c = weather.drop(columns=['Precip Type'])`
* Some columns are later dropped (e.g. date parts / summary) to focus on numeric predictors.

Models used:

* **Logistic Regression**
* **Random Forest Classifier**

Evaluation includes:

* accuracy
* precision / recall / f1
* classification report

### 5) Regression Task (Temperature)

Target:

* `y_r = weather['Temperature (C)']`

Features:

* `x_r = weather.drop(columns=['Temperature (C)', 'Summary'])`

Models used:

* **Linear Regression**
* **Random Forest Regressor**

Evaluation includes:

* Mean Squared Error (MSE)

### 6) Time-Series Regression (LSTM)

Uses TensorFlow/Keras and:

* `timeseries_dataset_from_array(...)`
* Builds an **LSTM -> Dense(1)** model to predict future temperature values.

---

## Requirements

Recommended Python: **3.9+**

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

> Note: TensorFlow installation depends on your machine (CPU vs GPU). If TensorFlow gives issues, install the CPU version for stability.

---

## How to Run

1. Clone repo / download files
2. Make sure `weatherHistory.csv` is in the same folder as `IC_ML_01.ipynb`
3. Launch Jupyter:

```bash
jupyter notebook
```

4. Open `IC_ML_01.ipynb` and run all cells.

---

## Notes / Common Issues

### Path issues

If your notebook can’t find the CSV, confirm:

* the CSV file is in the same directory
* or update the path in:

```python
pd.read_csv("weatherHistory.csv")
```

### Large CSV in GitHub

If GitHub complains about file size:

* use **Git LFS**
* or upload dataset to Google Drive/Kaggle and keep only a download link in the README.

---

## Next Improvements (Optional Ideas)

* Use proper preprocessing pipeline (OneHotEncoder for `Summary`, scaling for numeric)
* Add train/validation split for LSTM (and plot loss curves)
* Compare more models (XGBoost / LightGBM)
* Save trained models (`joblib` for sklearn, `.keras` for TensorFlow)

---

## Author

Built and experimented in `IC_ML_01.ipynb`.

```

---

## ✅ “Should I make a CSV for the ipynb?”

Yes — **keep the CSV separate** (the way you already have it). That’s the normal, correct workflow.

Here’s the rule of thumb:

- **Notebook (`.ipynb`)** = your experiments + code + plots  
- **CSV (`.csv`)** = your dataset input  
- Keeping them separate makes your project reproducible and clean.

### When you *might not* commit the CSV
If the CSV is **too large for GitHub**, do one of these:

- Use **Git LFS** (best if you must keep it in repo)
- Or **don’t upload the CSV**, and instead:
  - add a link to Kaggle/Drive in the README
  - and add `weatherHistory.csv` to `.gitignore`

---

If you want, I can also generate a clean `.gitignore` for this project (Jupyter + Python + dataset files) and a short “dataset download” section for your README.
```
