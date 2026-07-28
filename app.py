"""
Traffic Volume Prediction & Congestion Classification — Streamlit App
=======================================================================

This app is a thin interface over the ALREADY TRAINED artifacts produced by
the notebooks in Notebooks/. It does not retrain, refit, or invent any model.

Artifacts used (verified to exist in this project):
    Models/regression_preprocessor.pkl         -> ColumnTransformer (StandardScaler + OneHotEncoder)
    Models/best_xgb_reg.pkl                    -> Tuned XGBRegressor (traffic_volume prediction)
    Models/classification_preprocessor.pkl     -> ColumnTransformer (StandardScaler + OneHotEncoder)
    Models/best_xgb_cls.pkl                    -> Tuned XGBClassifier (congestion_level prediction)
    Models/classification_label_encoder.pkl    -> LabelEncoder mapping {0,1,2} -> {High, Low, Medium}

The app still checks that each file loads successfully at runtime and will
show a clear error/warning instead of a fake prediction if anything is
missing or fails to load.
"""

import os
from datetime import date

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths (relative to the project root, where this file lives)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "Models")

REGRESSION_PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "regression_preprocessor.pkl")
REGRESSION_MODEL_PATH = os.path.join(MODELS_DIR, "best_xgb_reg.pkl")

CLASSIFICATION_PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "classification_preprocessor.pkl")
CLASSIFICATION_MODEL_PATH = os.path.join(MODELS_DIR, "best_xgb_cls.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "classification_label_encoder.pkl")

# ---------------------------------------------------------------------------
# Feature schema — must exactly match the training features used in
# Notebooks/05_Preprocessing_Regression.ipynb and
# Notebooks/06_Preprocessing_Classification.ipynb
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "temp", "rain_1h", "snow_1h", "clouds_all", "fog_mm",
    "wind_speed_ms", "flood_mm", "hour", "day_of_week",
    "month", "year", "is_weekend", "is_holiday",
]
CATEGORICAL_FEATURES = ["holiday", "weather_description", "time_period"]
FEATURE_ORDER = [
    "temp", "rain_1h", "snow_1h", "clouds_all", "fog_mm", "wind_speed_ms",
    "flood_mm", "hour", "day_of_week", "month", "year", "is_weekend",
    "is_holiday", "holiday", "weather_description", "time_period",
]

# Values taken directly from Data/traffic_feature_engineered.parquet (the
# actual training data) — not invented.
HOLIDAY_OPTIONS = [
    "No Holiday", "Christmas Day", "Columbus Day", "Independence Day",
    "Labor Day", "Martin Luther King Jr Day", "Memorial Day",
    "New Years Day", "State Fair", "Thanksgiving Day", "Veterans Day",
    "Washingtons Birthday",
]

WEATHER_DESCRIPTION_OPTIONS = [
    "sky is clear", "few clouds", "scattered clouds", "broken clouds",
    "overcast clouds", "mist", "haze", "fog", "smoke",
    "light rain", "moderate rain", "heavy intensity rain", "very heavy rain",
    "light intensity drizzle", "drizzle", "heavy intensity drizzle",
    "shower drizzle", "light intensity shower rain",
    "proximity shower rain", "light rain and snow",
    "light snow", "snow", "heavy snow", "light shower snow", "sleet",
    "freezing rain", "proximity thunderstorm",
    "proximity thunderstorm with rain",
    "proximity thunderstorm with drizzle",
    "thunderstorm", "thunderstorm with rain", "thunderstorm with drizzle",
    "thunderstorm with light rain", "thunderstorm with light drizzle",
    "thunderstorm with heavy rain", "squalls",
]

# Same bin edges/labels used in Notebooks/04_Feature_Engineering.ipynb
TIME_PERIOD_BINS = [-1, 5, 11, 16, 20, 23]
TIME_PERIOD_LABELS = ["Night", "Morning", "Afternoon", "Evening", "Late_Night"]

# Recommendation logic copied verbatim from
# Notebooks/09_Final Analysis & Model Saving.ipynb (get_recommendation)
def get_recommendation(level: str) -> str:
    if level == "High":
        return "Avoid peak hours or consider an alternative route."
    elif level == "Medium":
        return "Expect moderate delays and allow extra travel time."
    else:
        return "Traffic conditions are favorable."


# ---------------------------------------------------------------------------
# Resource loading (cached, read-only — never retrains or overwrites files)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_resources():
    resources = {
        "regression_preprocessor": None,
        "regression_model": None,
        "classification_preprocessor": None,
        "classification_model": None,
        "label_encoder": None,
        "errors": [],
    }

    # Regression preprocessor
    try:
        resources["regression_preprocessor"] = joblib.load(REGRESSION_PREPROCESSOR_PATH)
    except Exception as e:
        resources["errors"].append(f"Regression preprocessor could not be loaded: {e}")

    # Regression model
    try:
        resources["regression_model"] = joblib.load(REGRESSION_MODEL_PATH)
    except Exception as e:
        resources["errors"].append(f"Regression model could not be loaded: {e}")

    # Classification preprocessor
    if os.path.exists(CLASSIFICATION_PREPROCESSOR_PATH):
        try:
            resources["classification_preprocessor"] = joblib.load(CLASSIFICATION_PREPROCESSOR_PATH)
        except Exception as e:
            resources["errors"].append(f"Classification preprocessor could not be loaded: {e}")
    else:
        resources["errors"].append("Classification preprocessor file not found.")

    # Classification model
    if os.path.exists(CLASSIFICATION_MODEL_PATH):
        try:
            resources["classification_model"] = joblib.load(CLASSIFICATION_MODEL_PATH)
        except Exception as e:
            resources["errors"].append(f"Classification model could not be loaded: {e}")
    else:
        resources["errors"].append("Classification model file not found.")

    # Label encoder
    if os.path.exists(LABEL_ENCODER_PATH):
        try:
            resources["label_encoder"] = joblib.load(LABEL_ENCODER_PATH)
        except Exception as e:
            resources["errors"].append(f"Label encoder could not be loaded: {e}")
    else:
        resources["errors"].append("Classification label encoder file not found.")

    return resources


def build_feature_row(inputs: dict) -> pd.DataFrame:
    """Assemble a single-row DataFrame with the exact columns/order the
    preprocessing pipelines expect."""
    row = {col: inputs[col] for col in FEATURE_ORDER}
    return pd.DataFrame([row], columns=FEATURE_ORDER)


def predict_traffic_volume(df_row: pd.DataFrame, resources: dict):
    preprocessor = resources["regression_preprocessor"]
    model = resources["regression_model"]
    if preprocessor is None or model is None:
        return None, "Regression preprocessor or model is not available."
    try:
        X = preprocessor.transform(df_row)
        prediction = model.predict(X)[0]
        return float(prediction), None
    except Exception as e:
        return None, f"Regression prediction failed: {e}"


def predict_congestion_level(df_row: pd.DataFrame, resources: dict):
    preprocessor = resources["classification_preprocessor"]
    model = resources["classification_model"]
    encoder = resources["label_encoder"]
    if preprocessor is None or model is None or encoder is None:
        return None, (
            "Congestion-level classification is unavailable in this project. "
            "No trained classification model / label encoder file was found "
            "under Models/ (see README.md for details)."
        )
    try:
        X = preprocessor.transform(df_row)
        pred_encoded = model.predict(X)[0]
        label = encoder.inverse_transform([pred_encoded])[0]
        return label, None
    except Exception as e:
        return None, f"Classification prediction failed: {e}"


# ---------------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Traffic Volume Prediction & Congestion Classification",
    page_icon="🚦",
    layout="wide",
)

resources = load_resources()

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🚦 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home / Prediction", "📊 Prediction Results", "🤖 Model Information", "📁 About the Project"],
)

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

# ---------------------------------------------------------------------------
# PAGE: Home / Prediction
# ---------------------------------------------------------------------------
if page == "🏠 Home / Prediction":
    st.title("🚦 Traffic Volume Prediction & Congestion Classification")
    st.write(
        "Predict hourly interstate traffic volume and (when available) the "
        "resulting congestion level, using the models trained in this project's "
        "notebooks."
    )

    if resources["errors"]:
        for err in resources["errors"]:
            st.warning(err)

    st.sidebar.header("📥 Input Features")

    # --- Date & time (derives hour, day_of_week, month, year, is_weekend) ---
    st.sidebar.subheader("Date & Time")
    input_date = st.sidebar.date_input("Date", value=date(2018, 6, 15))
    input_hour = st.sidebar.slider("Hour of day", min_value=0, max_value=23, value=8)

    day_of_week = input_date.weekday()  # 0 = Monday ... 6 = Sunday
    month = input_date.month
    year = input_date.year
    is_weekend = 1 if day_of_week in (5, 6) else 0
    time_period = pd.cut(
        [input_hour], bins=TIME_PERIOD_BINS, labels=TIME_PERIOD_LABELS
    )[0]

    st.sidebar.caption(
        f"Derived → day_of_week: {day_of_week}, month: {month}, year: {year}, "
        f"is_weekend: {is_weekend}, time_period: {time_period}"
    )

    # --- Holiday (derives is_holiday) ---
    st.sidebar.subheader("Holiday")
    holiday = st.sidebar.selectbox("Holiday", HOLIDAY_OPTIONS, index=0)
    is_holiday = 0 if holiday == "No Holiday" else 1

    # --- Weather ---
    st.sidebar.subheader("Weather Conditions")
    temp = st.sidebar.number_input("Temperature (Kelvin)", value=280.0, min_value=200.0, max_value=320.0, step=0.1)
    rain_1h = st.sidebar.number_input("Rain in last hour (mm)", value=0.0, min_value=0.0, step=0.1)
    snow_1h = st.sidebar.number_input("Snow in last hour (mm)", value=0.0, min_value=0.0, step=0.01, format="%.2f")
    clouds_all = st.sidebar.slider("Cloud coverage (%)", min_value=0, max_value=100, value=40)
    fog_mm = st.sidebar.number_input("Fog level (mm)", value=0.0, min_value=0.0, step=0.1)
    wind_speed_ms = st.sidebar.number_input("Wind speed (m/s)", value=3.0, min_value=0.0, step=0.1)
    flood_mm = st.sidebar.number_input("Flood level (mm)", value=0.0, min_value=0.0, step=0.1)
    weather_description = st.sidebar.selectbox("Weather description", WEATHER_DESCRIPTION_OPTIONS, index=0)

    inputs = {
        "temp": temp,
        "rain_1h": rain_1h,
        "snow_1h": snow_1h,
        "clouds_all": float(clouds_all),
        "fog_mm": fog_mm,
        "wind_speed_ms": wind_speed_ms,
        "flood_mm": flood_mm,
        "hour": input_hour,
        "day_of_week": day_of_week,
        "month": month,
        "year": year,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "holiday": holiday,
        "weather_description": weather_description,
        "time_period": str(time_period),
    }

    st.subheader("Review Input")
    df_row = build_feature_row(inputs)
    st.dataframe(df_row, use_container_width=True)

    if st.button("🔮 Predict Traffic", type="primary"):
        volume, vol_error = predict_traffic_volume(df_row, resources)
        congestion, cong_error = predict_congestion_level(df_row, resources)

        st.session_state["last_result"] = {
            "inputs": df_row,
            "volume": volume,
            "vol_error": vol_error,
            "congestion": congestion,
            "cong_error": cong_error,
        }

        st.subheader("📊 Results")
        col1, col2 = st.columns(2)

        with col1:
            if volume is not None:
                st.metric("Predicted Traffic Volume", f"{round(volume):,} vehicles/hour")
            else:
                st.error(vol_error)

        with col2:
            if congestion is not None:
                icon = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}.get(congestion, "⚪")
                st.metric("Predicted Congestion Level", f"{icon} {congestion}")
            else:
                st.info(cong_error)

        if volume is not None and congestion is not None:
            st.success(f"**Recommendation:** {get_recommendation(congestion)}")
        elif volume is not None:
            st.write(
                "A traffic-volume-only recommendation isn't produced by this "
                "project's decision-support logic, which requires both the "
                "predicted volume and congestion level. Congestion classification "
                "is currently unavailable — see the message above."
            )

# ---------------------------------------------------------------------------
# PAGE: Prediction Results
# ---------------------------------------------------------------------------
elif page == "📊 Prediction Results":
    st.title("📊 Prediction Results")
    result = st.session_state.get("last_result")

    if result is None:
        st.info("No prediction has been made yet. Go to '🏠 Home / Prediction' to run one.")
    else:
        st.write("Inputs used for the last prediction:")
        st.dataframe(result["inputs"], use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if result["volume"] is not None:
                st.metric("Predicted Traffic Volume", f"{round(result['volume']):,} vehicles/hour")
            else:
                st.error(result["vol_error"])
        with col2:
            if result["congestion"] is not None:
                icon = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}.get(result["congestion"], "⚪")
                st.metric("Predicted Congestion Level", f"{icon} {result['congestion']}")
            else:
                st.info(result["cong_error"])

        if result["volume"] is not None and result["congestion"] is not None:
            st.success(f"**Recommendation:** {get_recommendation(result['congestion'])}")

# ---------------------------------------------------------------------------
# PAGE: Model Information
# ---------------------------------------------------------------------------
elif page == "🤖 Model Information":
    st.title("🤖 Model Information")

    st.subheader("Regression — Traffic Volume Prediction")
    st.markdown(
        """
        - **Model:** Tuned XGBoost Regressor (`Models/best_xgb_reg.pkl`)
        - **Preprocessor:** `Models/regression_preprocessor.pkl`
          (StandardScaler on numeric features + OneHotEncoder on categorical features)
        - **Test performance:** MAE = 278.06 · RMSE = 486.61 · R² = 0.9392
        - **Best hyperparameters:** `subsample=0.9, n_estimators=100, min_child_weight=3,
          max_depth=5, learning_rate=0.1, gamma=0.1, colsample_bytree=1.0`
        """
    )

    st.subheader("Classification — Congestion Level")
    if resources["classification_model"] is not None and resources["label_encoder"] is not None:
        st.markdown(
            """
            - **Model:** Tuned XGBoost Classifier (`Models/best_xgb_cls.pkl`)
            - **Preprocessor:** `Models/classification_preprocessor.pkl`
            - **Label Encoder:** `Models/classification_label_encoder.pkl`
              (maps encoded classes back to `High`, `Low`, `Medium`)
            - **Test performance:** Accuracy ≈ 0.897 · F1 (weighted) ≈ 0.897
            - **Best hyperparameters:** `subsample=0.7, n_estimators=500,
              max_depth=3, learning_rate=0.05, colsample_bytree=0.7`
            """
        )
    else:
        st.warning(
            "⚠️ The classification model, preprocessor, or label encoder could not "
            "be loaded. Check that all three files exist under `Models/`: "
            "`classification_preprocessor.pkl`, `best_xgb_cls.pkl`, and "
            "`classification_label_encoder.pkl`."
        )

    st.subheader("Feature Schema")
    st.markdown("Both models share the same 16 input features:")
    st.code(", ".join(FEATURE_ORDER), language="text")

# ---------------------------------------------------------------------------
# PAGE: About the Project
# ---------------------------------------------------------------------------
elif page == "📁 About the Project":
    st.title("📁 About the Project")
    st.markdown(
        """
        This application is the interactive front-end for a Machine Learning
        project analyzing the **Metro Interstate Traffic Volume** dataset.

        The full pipeline — data cleaning, EDA, feature engineering,
        preprocessing, modeling, and hyperparameter tuning — is documented in
        the `Notebooks/` folder and summarized in `README.md`.

        **This app does not train any model.** It only loads the models and
        preprocessing objects that were already saved to `Models/` by the
        notebooks, and applies them to user-provided inputs.

        For full details on the dataset, methodology, and results, see
        `README.md` in the project root.
        """
    )
