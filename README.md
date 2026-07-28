# 🚦 Metro Interstate Traffic Volume — Prediction & Decision Support System

---

## 🚀 Live Demo

Check out the live interactive web application hosted on Streamlit:

<p align="center">
  <a href="https://traffic-volume-prediction-ztpwawmvy2sgt4mfsium7m.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App">
  </a>
</p>

Or access it via direct link: [Traffic Volume Prediction & Congestion App](https://traffic-volume-prediction-ztpwawmvy2sgt4mfsium7m.streamlit.app/)

---

## 📌 Project Overview

This project analyzes the **Metro Interstate Traffic Volume** dataset to build a complete Machine Learning pipeline that:

- Predicts **hourly traffic volume** (Regression)
- Classifies traffic into a **congestion level** — Low / Medium / High (Classification)
- Feeds both predictions into a lightweight **Decision Support System** that returns a plain-language traffic recommendation

The project follows a full, notebook-driven ML workflow: data understanding → cleaning → EDA → feature engineering → preprocessing → modeling → hyperparameter tuning → final evaluation.

---

## 🎯 Problem Statement

Traffic volume on interstate roads is influenced by weather conditions, time of day, day of week, and holidays. Being able to anticipate both the **volume** of traffic and the resulting **congestion level** helps commuters and traffic planners make better decisions about route and timing.

## 🎯 Project Objectives

- Clean and understand the raw traffic sensor + weather dataset
- Engineer meaningful time-based and categorical features
- Train and compare multiple regression models to predict `traffic_volume`
- Train and compare multiple classification models to predict `congestion_level`
- Tune the best-performing model of each task
- Combine both predictions into a simple decision-support recommendation

---

## 📊 Dataset Description

**Source file:** `Metro_Interstate_Traffic_Volume_modified.csv`

| Property | Value |
|---|---|
| Raw shape | 48,204 rows × 13 columns |
| Shape after cleaning + feature engineering | 45,794 rows × 21 columns |

**Original columns:** `date_time`, `temp`, `rain_1h`, `snow_1h`, `clouds_all`, `fog_mm`, `wind_speed_ms`, `flood_mm`, `holiday`, `weather_main`, `weather_description`, `day_type`, `traffic_volume`

---

## 🧩 Features

**Final model input features (16, shared by both Regression and Classification):**

| Feature | Type | Description |
|---|---|---|
| `temp` | numeric | Temperature (Kelvin) |
| `rain_1h` | numeric | Rain volume in the last hour |
| `snow_1h` | numeric | Snow volume in the last hour |
| `clouds_all` | numeric | Cloud coverage percentage |
| `fog_mm` | numeric | Fog level (mm) |
| `wind_speed_ms` | numeric | Wind speed (m/s) |
| `flood_mm` | numeric | Flood level (mm) |
| `hour` | numeric | Hour of day (0–23) |
| `day_of_week` | numeric | Day of week (0=Mon … 6=Sun) |
| `month` | numeric | Month (1–12) |
| `year` | numeric | Year |
| `is_weekend` | numeric (0/1) | Weekend flag |
| `is_holiday` | numeric (0/1) | Holiday flag |
| `holiday` | categorical | Holiday name, or `No Holiday` |
| `weather_description` | categorical | Detailed weather description |
| `time_period` | categorical | `Night`, `Morning`, `Afternoon`, `Evening`, `Late_Night` |

**Targets**
- Regression target: `traffic_volume`
- Classification target: `congestion_level` (`Low` / `Medium` / `High`)

---

## 🔄 Project Workflow

```
Data Understanding → Data Cleaning → EDA → Feature Engineering
        → Preprocessing (Regression / Classification)
        → Modeling → Hyperparameter Tuning → Final Evaluation
        → Decision Support System → Streamlit App
```

---

## 🔍 Data Understanding

Notebook: `01_Data_Understanding.ipynb`

- Loaded the raw CSV (48,204 rows × 13 columns)
- Generated an automated `ydata-profiling` report (`Metro_Interstate_Traffic_Volume_modified_profile_report.html`)
- Inspected column types, missing values, and summary statistics

---

## 🧹 Data Cleaning

Notebook: `02_Data_Cleaning.ipynb`

- **Missing values:**
  - Dropped rows with missing `traffic_volume` (the target)
  - Filled missing values in `temp`, `rain_1h`, `snow_1h`, `clouds_all`, `flood_mm` with the column median
  - Filled missing `holiday` values with `"No Holiday"`
- **Duplicate records:** checked and quantified duplicate rows
- **Data types:** converted `date_time` to a proper datetime type
- **Outlier detection:** visualized outliers in numeric columns using boxplots

Output saved to `Data/traffic_cleaned.parquet`.

---

## 📈 EDA & Visualizations

Notebook: `03_EDA&Visualizations.ipynb`

### Univariate Analysis
Distribution analysis of individual numeric and categorical variables.

### Categorical Analysis
Distribution of `holiday`, `weather_main`, and `day_type`.

### Time-Based Analysis
- Traffic by hour of day
- Traffic by day of week
- Weekday vs. weekend traffic
- Traffic by month and year
- Hour × Day Type interaction

### Bivariate Analysis
- Weather vs. traffic: temperature, rain, snow, wind, fog vs. traffic volume
- Categorical vs. traffic: day type, weather main, holiday vs. traffic volume
- Hour vs. traffic by day type

### Multivariate Analysis
- Hour + weather + traffic relationships
- Traffic heatmap: hour × day of week

### Correlation Analysis
Correlation between numeric features and `traffic_volume`.

### Key Insights
- Traffic volume follows a clear hourly and weekly pattern (rush-hour peaks on weekdays)
- Weekends show a different, flatter traffic profile than weekdays
- Weather conditions (rain, snow, fog) show a visible relationship with traffic behavior

---

## 🛠️ Feature Engineering

Notebook: `04_Feature_Engineering.ipynb`

New features created from `date_time` and `traffic_volume`:

| New Feature | Logic |
|---|---|
| `hour`, `day_of_week`, `month`, `year` | Extracted from `date_time` |
| `is_weekend` | `1` if `day_of_week` is Saturday/Sunday |
| `is_holiday` | `1` if `holiday != "No Holiday"` |
| `time_period` | Binned from `hour` → `Night`, `Morning`, `Afternoon`, `Evening`, `Late_Night` |
| `congestion_level` | `traffic_volume` split into 3 equal-frequency bins (`pd.qcut`) → `Low`, `Medium`, `High` |

`weather_description` was also normalized (lower-cased and stripped).

Output saved to `Data/traffic_feature_engineered.parquet`.

---

## ⚙️ Regression Preprocessing

Notebook: `05_Preprocessing_Regression.ipynb`

- Time-based split: **train = years < 2018**, **test = year == 2018**
  - Train: 38,257 rows · Test: 7,537 rows
- `ColumnTransformer`:
  - `StandardScaler` on 13 numeric features
  - `OneHotEncoder(handle_unknown='ignore')` on `holiday`, `weather_description`, `time_period`
- Output: 66 processed features
- Saved: `Models/regression_preprocessor.pkl`, and processed train/test arrays in `Data/`

---

## ⚙️ Classification Preprocessing

Notebook: `06_Preprocessing_Classification.ipynb`

- Identical feature set and identical time-based split as regression
- Same `ColumnTransformer` structure (`StandardScaler` + `OneHotEncoder`)
- Output: 66 processed features
- Saved: `Models/classification_preprocessor.pkl`, and processed train/test arrays in `Data/`

---

## 🤖 Regression Modeling

Notebook: `07_Regression_Modeling.ipynb`

Baseline: `DummyRegressor` (mean strategy) → MAE 1728.06 · RMSE 1973.94 · R² ≈ 0

### Regression Model Comparison

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 782.53 | 999.57 | 0.7436 |
| Decision Tree | 366.52 | 654.13 | 0.8902 |
| Random Forest | 287.69 | 506.78 | 0.9341 |
| **XGBoost** | **280.06** | **480.67** | **0.9407** |

XGBoost was selected as the best-performing regression model.

## 🔧 Regression Hyperparameter Tuning

`RandomizedSearchCV` (10 iterations, 3-fold CV, scoring = `neg_root_mean_squared_error`) over `XGBRegressor`.

**Best parameters found:**
```
subsample=0.9, n_estimators=100, min_child_weight=3,
max_depth=5, learning_rate=0.1, gamma=0.1, colsample_bytree=1.0
```

**Tuned XGBoost test performance:** MAE = 278.06 · RMSE = 486.61 · R² = 0.9392

This tuned model is the one saved to `Models/best_xgb_reg.pkl` and used by the Streamlit app.

---

## 🤖 Classification Modeling

Notebook: `08_Classification_Modeling.ipynb`

Target labels encoded with `LabelEncoder` → classes: `High`, `Low`, `Medium`
Baseline (majority-class prediction): Accuracy = 0.3328

### Classification Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.7657 | 0.7610 | 0.7657 | 0.7624 |
| Decision Tree | 0.8547 | 0.8574 | 0.8547 | 0.8557 |
| Random Forest | 0.8901 | 0.8919 | 0.8901 | 0.8903 |
| **XGBoost** | **0.8981** | **0.8998** | **0.8981** | **0.8984** |

XGBoost was selected as the best-performing classification model.

## 🔧 Classification Hyperparameter Tuning

`RandomizedSearchCV` (20 iterations, 3-fold CV, scoring = `f1_weighted`) over `XGBClassifier`.

**Best parameters found:**
```
subsample=0.7, n_estimators=500, max_depth=3,
learning_rate=0.05, colsample_bytree=0.7
```

**Tuned XGBoost test performance:** Accuracy = 0.897 · Precision = 0.899 · Recall = 0.897 · F1 = 0.897

This tuned model is saved to `Models/best_xgb_cls.pkl`, and the `LabelEncoder` used to map `congestion_level` classes is saved to `Models/classification_label_encoder.pkl`. Both are used by the Streamlit app to serve congestion-level predictions.

---

## 🏆 Final Model Selection

| Task | Best Model | Main Metric | Score | Saved to disk? |
|---|---|---|---|---|
| Traffic Volume Prediction | Tuned XGBoost Regressor | R² | 0.9392 | ✅ `Models/best_xgb_reg.pkl` |
| Congestion Level Classification | Tuned XGBoost Classifier | F1 Score | 0.897 | ✅ `Models/best_xgb_cls.pkl` (+ `Models/classification_label_encoder.pkl`) |

---

## 🧭 Decision Support System

Notebook: `09_Final Analysis & Model Saving.ipynb`

Combines the regression and classification outputs into a plain-language recommendation:

| Congestion Level | Recommendation |
|---|---|
| High | Avoid peak hours or consider an alternative route. |
| Medium | Expect moderate delays and allow extra travel time. |
| Low | Traffic conditions are favorable. |

---

## 💻 Streamlit Application

`app.py` (project root) provides an interactive interface built strictly on the models and preprocessing objects that actually exist in this project. Both prediction tasks are fully supported:

**Regression pipeline:** User Input → `Models/regression_preprocessor.pkl` → `Models/best_xgb_reg.pkl` → **Predicted Traffic Volume**

**Classification pipeline:** User Input → `Models/classification_preprocessor.pkl` → `Models/best_xgb_cls.pkl` → `Models/classification_label_encoder.pkl` → **Predicted Congestion Level** (`Low` / `Medium` / `High`)

Both predictions are combined into the same style of recommendation used in `09_Final Analysis & Model Saving.ipynb`. The app performs no retraining — it only loads the already-trained artifacts and applies them to user-provided inputs, and will show a clear error/warning instead of a fabricated prediction if any artifact fails to load.

---

## 📁 Project Structure

```
Final_Project/
├── README.md
├── app.py
├── requirements.txt
├── Data/
│   ├── Metro_Interstate_Traffic_Volume_modified.csv
│   ├── traffic_cleaned.parquet
│   ├── traffic_feature_engineered.parquet
│   ├── X_train_reg.pkl
│   ├── X_test_reg.pkl
│   ├── y_train_reg.pkl
│   ├── y_test_reg.pkl
│   ├── X_train_cls.pkl
│   ├── X_test_cls.pkl
│   ├── y_train_cls.pkl
│   └── y_test_cls.pkl
├── Models/
│   ├── regression_preprocessor.pkl
│   ├── classification_preprocessor.pkl
│   ├── best_xgb_reg.pkl
│   ├── best_xgb_cls.pkl
│   └── classification_label_encoder.pkl
└── Notebooks/
    ├── 01_Data_Understanding.ipynb
    ├── 02_Data_Cleaning.ipynb
    ├── 03_EDA&Visualizations.ipynb
    ├── 04_Feature_Engineering.ipynb
    ├── 05_Preprocessing_Regression.ipynb
    ├── 06_Preprocessing_Classification.ipynb
    ├── 07_Regression_Modeling.ipynb
    ├── 08_Classification_Modeling.ipynb
    ├── 09_Final Analysis & Model Saving.ipynb
    ├── Metro_Interstate_Traffic_Volume_modified_profile_report.html
    ├── Project.ipynb                    (legacy, not part of the active pipeline)
    ├── Project_(1).ipynb                (legacy, not part of the active pipeline)
    └── Project_traffic_prediction.ipynb (legacy, not part of the active pipeline)
```

---

## 🧰 Technologies & Libraries

- **Data handling:** `pandas`, `numpy`, `pyarrow` (parquet)
- **Visualization:** `matplotlib`, `seaborn`
- **Modeling:** `scikit-learn` (Linear/Logistic Regression, Decision Tree, Random Forest, preprocessing), `xgboost`
- **Model persistence:** `joblib`
- **Profiling:** `ydata-profiling`
- **App:** `streamlit`

---

## 🔑 Key Findings

- Traffic volume is highly predictable from time-based and weather features — XGBoost achieved **R² ≈ 0.94** on unseen 2018 data.
- Congestion level classification reaches **~90% F1 score** with a tuned XGBoost classifier.
- Time-based features (`hour`, `day_of_week`, `time_period`) and weather conditions are the primary drivers behind both prediction tasks, consistent with the EDA findings.

---

## 🚀 Future Improvements

- Add cross-validated hyperparameter tuning with a wider search space.
- Add model explainability (e.g., SHAP) for feature importance in the app.
- Persist the "final" model variables from `09_Final Analysis & Model Saving.ipynb` for a single canonical source of truth per task.

---

## 🖥️ Installation

```bash
git clone <your-repo-url>
cd Final_Project
pip install -r requirements.txt
```

## ▶️ How to Run the Streamlit App

From the project root directory:

```bash
streamlit run app.py
```

---

## ✍️ Authors

Project author — update with your name/contact details.

## 📄 License

Add your preferred license (e.g., MIT) here.
