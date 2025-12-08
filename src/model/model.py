# Complete SIA Forecasting System with Integrated Dashboard Generation
# Save this as models.py - it replaces your existing models.py
import warnings
import logging
import numpy as np
import pandas as pd
from math import sqrt
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.seasonal import STL
from prophet import Prophet
from pmdarima.arima import auto_arima
from statsmodels.tsa.ar_model import AutoReg
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from datetime import datetime

# --------------------------------------------------------
# CONFIG: Logging and warnings
# --------------------------------------------------------
loggers = ["cmdstanpy", "prophet", "pmdarima", "statsmodels", "fbprophet"]
for logger_name in loggers:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False
warnings.filterwarnings("ignore")

# Set professional SIA color palette
SIA_COLORS = {
    'primary_blue': '#1E3A8A',
    'secondary_blue': '#3B82F6', 
    'accent_gold': '#F59E0B',
    'light_gold': '#FEF3C7',
    'dark_gray': '#374151',
    'light_gray': '#F3F4F6',
    'success_green': '#10B981',
    'warning_red': '#EF4444',
    'bias_positive': '#60A5FA',  # Light blue for positive bias
    'bias_negative': '#F97316'   # Orange for negative bias
}

plt.style.use('default')
sns.set_palette([SIA_COLORS['primary_blue'], SIA_COLORS['accent_gold'], SIA_COLORS['secondary_blue']])

# --------------------------------------------------------
# Enhanced SMAPE function
def smape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denominator = (np.abs(y_pred) + np.abs(y_true))
    mask = denominator > 1e-8
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs(y_pred[mask] - y_true[mask]) / denominator[mask])

# --------------------------------------------------------
# Reading Universal Input Sheet with score benchmark
def read_universal_input_sheet(excel_file):
    raw = pd.read_excel(excel_file, header=None)
    forecast_horizon = int(raw.iloc[0, 3])
    score_benchmark = float(raw.iloc[0, 6])  # G1 cell
    timeline_cols = raw.iloc[5, 1:].astype(str).tolist()
    data_part = raw.iloc[6:, :]
    items = data_part.iloc[:, 0].astype(str).tolist()
    numeric_data = data_part.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").values
    df = pd.DataFrame(numeric_data, index=items, columns=timeline_cols)
    return forecast_horizon, score_benchmark, df

# --------------------------------------------------------
# Enhanced Seasonal Period Detection
def guess_seasonal_period(timeline_cols):
    if any("W" in col.upper() for col in timeline_cols):
        return 52
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    if any(any(m in col for m in months) for col in timeline_cols):
        return 12
    if any("Q" in col.upper() for col in timeline_cols):
        return 4
    if all(col.isdigit() and len(col) == 4 for col in timeline_cols):
        return 1
    return 1

# --------------------------------------------------------
# Enhanced Outlier smoothing
def smooth_series(series, method='iqr'):
    if method == 'iqr':
        Q1 = np.percentile(series, 25)
        Q3 = np.percentile(series, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return np.clip(series, lower_bound, upper_bound)
    else:
        q_low = np.quantile(series, 0.05)
        q_high = np.quantile(series, 0.95)
        return np.clip(series, q_low, q_high)

# --------------------------------------------------------
# Enhanced Forecasters Class
class EnhancedForecasters:
    def __init__(self, train_data, seasonal_period):
        self.train = smooth_series(train_data.dropna())
        self.seasonal_period = seasonal_period
        freq = self.seasonal_period
        if freq == 12:
            self.train.index = pd.date_range("2000-01-01", periods=len(self.train), freq="MS")
        elif freq == 52:
            self.train.index = pd.date_range("2000-01-01", periods=len(self.train), freq="W-SUN")
        elif freq == 4:
            self.train.index = pd.date_range("2000-01-01", periods=len(self.train), freq="QS")
        else:
            self.train.index = pd.Index(range(len(self.train)))

    def sarima(self):
        try:
            return auto_arima(
                self.train,
                start_p=0, max_p=2, start_q=0, max_q=2,
                d=None, max_d=1,
                start_P=0, max_P=1, start_Q=0, max_Q=1,
                D=None, max_D=1,
                seasonal=True if self.seasonal_period > 1 else False,
                m=self.seasonal_period,
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
                trace=False,
                information_criterion='aic'
            )
        except:
            return None

    def holt_winters(self):
        try:
            return ExponentialSmoothing(
                self.train,
                trend="add",
                seasonal=("add" if self.seasonal_period > 1 else None),
                seasonal_periods=(self.seasonal_period if self.seasonal_period > 1 else None),
                initialization_method="estimated",
                damped_trend=True
            ).fit()
        except:
            return None

    def prophet(self):
        try:
            df = self.train.reset_index()
            df.columns = ["ds", "y"]
            m = Prophet(
                yearly_seasonality=(self.seasonal_period == 12),
                weekly_seasonality=(self.seasonal_period == 52),
                daily_seasonality=False,
                seasonality_mode="multiplicative" if self.seasonal_period > 1 else "additive",
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10
            )
            if self.seasonal_period == 4:
                m.add_seasonality(name="quarterly", period=4, fourier_order=3)
            m.fit(df)
            return m
        except:
            return None

    def croston(self):
        try:
            demand = self.train.values
            non_zero_demand = demand[demand > 0]
            if len(non_zero_demand) == 0:
                return None
            
            intervals = []
            last_non_zero = -1
            for i, val in enumerate(demand):
                if val > 0:
                    if last_non_zero >= 0:
                        intervals.append(i - last_non_zero)
                    last_non_zero = i
            
            if len(intervals) == 0:
                return SimpleExpSmoothing(non_zero_demand).fit(optimized=True)
            
            alpha = 0.1
            avg_demand = np.mean(non_zero_demand)
            avg_interval = np.mean(intervals)
            
            for i in range(1, len(non_zero_demand)):
                avg_demand = alpha * non_zero_demand[i] + (1 - alpha) * avg_demand
            
            for i in range(1, len(intervals)):
                avg_interval = alpha * intervals[i] + (1 - alpha) * avg_interval
            
            forecast = avg_demand / avg_interval if avg_interval > 0 else avg_demand
            return forecast
        except:
            return None

    def autoreg(self):
        try:
            return AutoReg(self.train, lags=1, old_names=False).fit()
        except:
            return None
    
    def xgboost(self):
        try:
            df = self.train.to_frame("y")
            df["position"] = np.arange(len(df))
            df["sin_year"] = np.sin(2 * np.pi * df["position"] / 52)
            df["cos_year"] = np.cos(2 * np.pi * df["position"] / 52)
            df["lag1"] = df["y"].shift(1)
            df["lag2"] = df["y"].shift(2)
            df = df.dropna()
            if df.empty:
                return None
            X = df[["lag1", "lag2", "sin_year", "cos_year"]]
            y = df["y"]
            m = xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=300,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
            )
            m.fit(X, y)
            return m
        except:
            return None

    def lgbm(self):
        try:
            df = self.train.to_frame("y")
            df["position"] = np.arange(len(df))
            df["sin_season"] = np.sin(2 * np.pi * df["position"] / self.seasonal_period)
            df["cos_season"] = np.cos(2 * np.pi * df["position"] / self.seasonal_period)
            df["lag1"] = df["y"].shift(1)
            df["lag2"] = df["y"].shift(2)
            df = df.dropna()
            if df.empty:
                return None
            X = df[["lag1", "lag2", "sin_season", "cos_season"]]
            y = df["y"]
            m = lgb.LGBMRegressor(
                objective="regression",
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
                verbose=-1,
            )
            m.fit(X, y)
            return m
        except:
            return None
        
    def mlp(self):
        try:
            df = self.train.to_frame("y")
            df["position"] = np.arange(len(df))
            df["sin_season"] = np.sin(2 * np.pi * df["position"] / self.seasonal_period)
            df["cos_season"] = np.cos(2 * np.pi * df["position"] / self.seasonal_period)
            df["lag1"] = df["y"].shift(1)
            df["lag2"] = df["y"].shift(2)
            df = df.dropna()
            if df.empty:
                return None
            X = df[["lag1", "lag2", "sin_season", "cos_season"]]
            y = df["y"]
            m = MLPRegressor(
                hidden_layer_sizes=(100,),
                activation="relu",
                solver="adam",
                alpha=0.0001,
                max_iter=1000,
                random_state=42,
            )
            m.fit(X, y)
            return m
        except:
            return None

    def catboost(self):
        try:
            df = self.train.to_frame("y")
            df["position"] = np.arange(len(df))
            df["sin_season"] = np.sin(2 * np.pi * df["position"] / self.seasonal_period)
            df["cos_season"] = np.cos(2 * np.pi * df["position"] / self.seasonal_period)
            df["lag1"] = df["y"].shift(1)
            df["lag2"] = df["y"].shift(2)
            df = df.dropna()
            if df.empty:
                return None
            X = df[["lag1", "lag2", "sin_season", "cos_season"]]
            y = df["y"]
            m = CatBoostRegressor(
                iterations=300,
                depth=4,
                learning_rate=0.1,
                random_state=42,
                verbose=False,
            )
            m.fit(X, y)
            return m
        except:
            return None

    def sba(self):
        """Enhanced Syntetos-Boylan Approximation"""
        try:
            demand = self.train.values
            non_zero_demand = demand[demand > 0]
            if len(non_zero_demand) == 0:
                return None
            
            cv = np.std(non_zero_demand) / np.mean(non_zero_demand) if np.mean(non_zero_demand) > 0 else 0
            
            intervals = []
            last_non_zero = -1
            for i, val in enumerate(demand):
                if val > 0:
                    if last_non_zero >= 0:
                        intervals.append(i - last_non_zero)
                    last_non_zero = i
            
            adi = np.mean(intervals) if intervals else 1.0
            
            if cv < 0.49:
                correction = 1.0
            else:
                correction = 1 - (cv - 0.49) / cv
            
            croston_forecast = self.croston()
            if isinstance(croston_forecast, (int, float)):
                return croston_forecast * correction
            else:
                return croston_forecast.forecast(1)[0] * correction if croston_forecast else None
        except:
            return None

    def adidda(self):
        """Enhanced ADIDDA with STL decomposition"""
        try:
            if len(self.train) < 2 * self.seasonal_period:
                return ExponentialSmoothing(
                    self.train,
                    trend="add",
                    seasonal=None,
                    initialization_method="estimated",
                ).fit()
            
            stl = STL(self.train, seasonal=self.seasonal_period, robust=True)
            decomposition = stl.fit()
            
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            
            trend_model = ExponentialSmoothing(
                trend.dropna(), 
                trend="add", 
                seasonal=None
            ).fit()
            
            if self.seasonal_period > 1:
                seasonal_model = ExponentialSmoothing(
                    seasonal.dropna(), 
                    seasonal="add", 
                    seasonal_periods=self.seasonal_period
                ).fit()
            else:
                seasonal_model = None
            
            class ADIDDAModel:
                def __init__(self, trend_model, seasonal_model, seasonal_period, last_seasonal):
                    self.trend_model = trend_model
                    self.seasonal_model = seasonal_model
                    self.seasonal_period = seasonal_period
                    self.last_seasonal = last_seasonal
                
                def forecast(self, steps):
                    trend_forecast = self.trend_model.forecast(steps)
                    
                    if self.seasonal_model:
                        seasonal_forecast = self.seasonal_model.forecast(steps)
                    else:
                        seasonal_forecast = np.tile(
                            self.last_seasonal[-self.seasonal_period:], 
                            (steps // self.seasonal_period) + 1
                        )[:steps]
                    
                    return trend_forecast + seasonal_forecast
            
            return ADIDDAModel(trend_model, seasonal_model, self.seasonal_period, seasonal.values)
            
        except:
            return ExponentialSmoothing(
                self.train,
                trend="add",
                seasonal=("add" if self.seasonal_period > 1 else None),
                seasonal_periods=(self.seasonal_period if self.seasonal_period > 1 else None),
                initialization_method="estimated",
            ).fit()

    def stl(self):
        """Enhanced STL decomposition method with proper forecasting"""
        try:
            if len(self.train) < 2 * self.seasonal_period:
                # Fallback to simple exponential smoothing for short series
                return SimpleExpSmoothing(self.train).fit(optimized=True)
            
            stl = STL(self.train, seasonal=self.seasonal_period, robust=True)
            decomposition = stl.fit()
            
            # Get components
            trend = decomposition.trend.dropna()
            seasonal = decomposition.seasonal
            
            # Fit trend model
            trend_model = ExponentialSmoothing(trend, trend="add").fit()
            
            class EnhancedSTLModel:
                def __init__(self, trend_model, seasonal_pattern, seasonal_period, original_series):
                    self.trend_model = trend_model
                    self.seasonal_pattern = seasonal_pattern
                    self.seasonal_period = seasonal_period
                    self.original_series = original_series
                
                def forecast(self, steps):
                    try:
                        # Get trend forecast
                        trend_fc = self.trend_model.forecast(steps)
                        
                        # Get seasonal pattern for forecast periods
                        seasonal_fc = []
                        for i in range(steps):
                            seasonal_idx = (len(self.original_series) + i) % self.seasonal_period
                            seasonal_fc.append(self.seasonal_pattern[seasonal_idx])
                        
                        seasonal_fc = np.array(seasonal_fc)
                        
                        # Combine trend and seasonal
                        forecast = trend_fc + seasonal_fc
                        
                        # Ensure non-negative values
                        forecast = np.maximum(forecast, 0)
                        
                        return forecast
                        
                    except:
                        # Fallback: use exponential smoothing on recent values
                        recent_values = self.original_series[-min(12, len(self.original_series)):]
                        fallback_model = ExponentialSmoothing(
                            recent_values, 
                            trend="add", 
                            seasonal=None
                        ).fit()
                        return fallback_model.forecast(steps)
            
            return EnhancedSTLModel(trend_model, seasonal.values, self.seasonal_period, self.train.values)
            
        except:
            # Ultimate fallback
            try:
                return ExponentialSmoothing(
                    self.train,
                    trend="add",
                    seasonal=("add" if self.seasonal_period > 1 else None),
                    seasonal_periods=(self.seasonal_period if self.seasonal_period > 1 else None)
                ).fit()
            except:
                return SimpleExpSmoothing(self.train).fit(optimized=True)

# Updated model names
model_names = [
    "SARIMA", "Holt-Winters", "Prophet", "Croston", "AutoReg", 
    "XGBoost", "SBA", "ADIDDA", "LGBM", "MLP", "CatBoost", "STL"
]

# Enhanced prediction functions
def _prophet_one(prophet_model, train_series):
    if not prophet_model:
        return np.nan
    try:
        dates = pd.date_range("2000-01-01", periods=len(train_series), freq="MS")
        fut = pd.DataFrame({"ds": [dates[-1] + pd.DateOffset(months=1)]})
        return prophet_model.predict(fut)["yhat"].iloc[-1]
    except:
        return np.nan

def _xgb_one(xgb_model, train_series):
    if not xgb_model or len(train_series) < 2:
        return np.nan
    try:
        n = len(train_series)
        lag1 = train_series.iloc[-1]
        lag2 = train_series.iloc[-2]
        position = n
        sin_year = np.sin(2 * np.pi * position / 52)
        cos_year = np.cos(2 * np.pi * position / 52)
        X_new = pd.DataFrame(
            [[lag1, lag2, sin_year, cos_year]],
            columns=["lag1", "lag2", "sin_year", "cos_year"],
        )
        pred = xgb_model.predict(X_new)
        return pred[0]
    except:
        return np.nan
    
def _ml_model_one(model, train_series, seasonal_period):
    if not model or len(train_series) < 2:
        return np.nan
    try:
        n = len(train_series)
        lag1 = train_series.iloc[-1]
        lag2 = train_series.iloc[-2] if len(train_series) >= 2 else np.nan
        position = n
        sin_season = np.sin(2 * np.pi * position / seasonal_period)
        cos_season = np.cos(2 * np.pi * position / seasonal_period)
        X_new = pd.DataFrame(
            [[lag1, lag2, sin_season, cos_season]],
            columns=["lag1", "lag2", "sin_season", "cos_season"],
        )
        pred = model.predict(X_new)[0]
        if np.isfinite(pred):
            return pred
        else:
            return np.nan
    except:
        return np.nan

def _stl_one(stl_model, train_series):
    if not stl_model:
        return np.nan
    try:
        # Get the last few values for trend estimation
        if len(train_series) >= 3:
            # Simple trend-based forecast
            recent_values = train_series.tail(3).values
            trend = (recent_values[-1] - recent_values[0]) / 2
            forecast = recent_values[-1] + trend
            return max(0, forecast)
        else:
            return train_series.iloc[-1]
    except:
        return train_series.iloc[-1] if len(train_series) > 0 else 0

def _adidda_one(adidda_model, train_series):
    if not adidda_model:
        return np.nan
    try:
        if hasattr(adidda_model, 'forecast'):
            return adidda_model.forecast(1)[0]
        else:
            return np.nan
    except:
        return np.nan

def _sba_one(sba_result, train_series):
    if isinstance(sba_result, (int, float)):
        return sba_result
    return np.nan

# Get forecast functions
def get_forecast_functions(seasonal_period):
    return {
        "SARIMA": lambda m, t: m.predict(n_periods=int(1))[0] if m else np.nan,
        "Holt-Winters": lambda m, t: m.forecast(1)[0] if m else np.nan,
        "Prophet": lambda m, t: _prophet_one(m, t),
        "Croston": lambda m, t: m if isinstance(m, (int, float)) else (m.forecast(1)[0] if m else np.nan),
        "AutoReg": lambda m, t: m.predict(start=len(t), end=len(t))[0] if m else np.nan,
        "XGBoost": lambda m, t: _xgb_one(m, t),
        "SBA": lambda m, t: _sba_one(m, t),
        "ADIDDA": lambda m, t: _adidda_one(m, t),
        "LGBM": lambda m, t: _ml_model_one(m, t, seasonal_period),
        "MLP": lambda m, t: _ml_model_one(m, t, seasonal_period),
        "CatBoost": lambda m, t: _ml_model_one(m, t, seasonal_period),
        "STL": lambda m, t: _stl_one(m, t),
    }

# Rolling forecast function
def rolling_forecast_for_product(series, test_indices, seasonal_period, forecast_funcs):
    periods = []
    agg_smapes = {m: [] for m in model_names}
    agg_err = {m: 0.0 for m in model_names}
    agg_act = {m: 0.0 for m in model_names}

    for idx in test_indices:
        train = series[:idx]
        actual = series[idx]
        
        if len(train) < 3:
            continue
            
        forecaster = EnhancedForecasters(pd.Series(train), seasonal_period)
        fitted = {}
        
        for m in model_names:
            method_name = m.lower().replace("-", "_")
            fitted[m] = getattr(forecaster, method_name)()
        
        fc = {}
        sm = {}
        
        for m in model_names:
            try:
                p = forecast_funcs[m](fitted[m], pd.Series(train))
                p = int(round(max(0, p))) if not np.isnan(p) else np.nan
                fc[m] = p
                sm[m] = smape([actual], [p]) if not np.isnan(p) else np.nan
                
                if not np.isnan(sm[m]):
                    agg_smapes[m].append(sm[m])
                if not np.isnan(p):
                    agg_err[m] += p - actual
                    agg_act[m] += actual
            except:
                fc[m] = np.nan
                sm[m] = np.nan
        periods.append({"actual": actual, "fc": fc, "sm": sm})
    
    return periods, agg_smapes, agg_err, agg_act

# Enhanced model selection
def choose_best_model(agg_smapes, agg_err, agg_act):
    best, bs, bias, score = None, np.inf, 0.0, np.inf
    
    for m in model_names:
        if not agg_smapes[m] or agg_act[m] == 0:
            continue
            
        avg_smape = np.mean(agg_smapes[m])
        std_smape = np.std(agg_smapes[m]) if len(agg_smapes[m]) > 1 else 0
        bias_val = agg_err[m] / agg_act[m]
        confidence_penalty = std_smape * 0.1
        enhanced_score = avg_smape + abs(bias_val) + confidence_penalty
        
        if enhanced_score < score:
            best, bs, bias, score = m, avg_smape, bias_val, enhanced_score
        if enhanced_score < score:
            best, bs, bias, score = m, avg_smape, bias_val, enhanced_score

        # Fix inf values
        if np.isinf(score):
            score = np.nan
        if np.isinf(enhanced_score):
            enhanced_score = np.nan
    
    if not best:
        return "Prophet", None, None, None
    
    return best, bs, bias, score

# Forecast next periods
def forecast_next_periods(series, best, horizon, seasonal_period, forecast_funcs):
    if len(series) < 2:
        return [np.nan] * horizon
    
    # For models that can do multi-step forecasting directly
    if best in ["SARIMA", "Holt-Winters", "Prophet", "STL", "ADIDDA"]:
        try:
            f = EnhancedForecasters(pd.Series(series), seasonal_period)
            method_name = best.lower().replace("-", "_")
            m = getattr(f, method_name)()
            
            if m:
                if best == "SARIMA":
                    vals = m.predict(n_periods=horizon)
                elif best == "Holt-Winters":
                    vals = m.forecast(horizon)
                elif best == "Prophet":
                    freq_map = {12: "MS", 52: "W-SUN", 4: "QS", 1: "AS"}
                    fut = m.make_future_dataframe(horizon, freq=freq_map.get(seasonal_period, "D"))
                    vals = m.predict(fut)["yhat"].iloc[-horizon:].values
                elif best == "STL":
                    if hasattr(m, 'forecast'):
                        vals = m.forecast(horizon)
                    else:
                        # Fallback for simple models
                        vals = [m.forecast(1)[0]] * horizon
                elif best == "ADIDDA":
                    vals = m.forecast(horizon) if hasattr(m, 'forecast') else [np.nan] * horizon
                
                return [int(round(max(0, v))) if np.isfinite(v) else np.nan for v in vals]
        except:
            pass
    
    # For ML models and others: do proper iterative forecasting
    temp_series = list(series)
    results = []
    
    for step in range(horizon):
        try:
            # Create new forecaster with current extended series
            f = EnhancedForecasters(pd.Series(temp_series), seasonal_period)
            method_name = best.lower().replace("-", "_")
            m = getattr(f, method_name)()
            
            # Get one-step forecast
            if m:
                val = forecast_funcs[best](m, pd.Series(temp_series))
                val = int(round(max(0, val))) if np.isfinite(val) else temp_series[-1]
            else:
                val = temp_series[-1]  # Use last known value if model fails
            
            results.append(val)
            temp_series.append(val)  # CRITICAL: Add prediction to series for next step
            
        except:
            # Fallback to last known value
            val = temp_series[-1] if temp_series else 0
            results.append(val)
            temp_series.append(val)
    
    return results

# Enhanced ensemble forecasting
def forecast_all_future(series, horizon, ensemble_scores, seasonal_period, forecast_funcs):
    mf = {}
    
    for m in model_names:
        try:
            f = EnhancedForecasters(pd.Series(series), seasonal_period)
            method_name = m.lower().replace("-", "_")
            mod = getattr(f, method_name)()
            
            if mod and m in ["SARIMA", "Holt-Winters", "Prophet", "STL", "ADIDDA"]:
                if m == "SARIMA":
                    vals = mod.predict(n_periods=horizon)
                elif m == "Holt-Winters":
                    vals = mod.forecast(horizon)
                elif m == "STL":
                    vals = mod.forecast(horizon)
                elif m == "ADIDDA":
                    vals = mod.forecast(horizon) if hasattr(mod, 'forecast') else [np.nan] * horizon
                else:  # Prophet
                    fr = {"12": "MS", "52": "W-SUN", "4": "QS", "1": "AS"}[str(seasonal_period)]
                    fut = mod.make_future_dataframe(periods=horizon, freq=fr)
                    vals = mod.predict(fut)["yhat"].iloc[-horizon:].values
                
                mf[m] = [int(round(max(0, v))) if not np.isnan(v) else np.nan for v in vals]
            else:
                # For ML models, do proper iterative forecasting
                temp_series = list(series)
                model_forecasts = []
                
                for step in range(horizon):
                    try:
                        f = EnhancedForecasters(pd.Series(temp_series), seasonal_period)
                        method_name = m.lower().replace("-", "_")
                        mod = getattr(f, method_name)()
                        
                        if mod:
                            val = forecast_funcs[m](mod, pd.Series(temp_series))
                            val = int(round(max(0, val))) if np.isfinite(val) else temp_series[-1]
                        else:
                            val = temp_series[-1]
                        
                        model_forecasts.append(val)
                        temp_series.append(val)  # Update series for next step
                        
                    except:
                        val = temp_series[-1] if temp_series else 0
                        model_forecasts.append(val)
                        temp_series.append(val)
                
                mf[m] = model_forecasts
        except:
            mf[m] = [np.nan] * horizon
    
    # Enhanced ensemble
    ens = []
    for j in range(horizon):
        preds = {m: mf[m][j] for m in model_names if not np.isnan(mf[m][j])}
        
        valid_models = [m for m in preds if ensemble_scores.get(m, np.inf) < np.inf]
        if not valid_models:
            ens.append(np.nan)
            continue
        
        top_models = sorted(valid_models, key=lambda m: ensemble_scores.get(m, np.inf))[:3]
        
        if top_models:
            weights = {}
            for i, m in enumerate(top_models):
                score = ensemble_scores.get(m, np.inf)
                if score < np.inf:
                    weights[m] = 1 / (score + 0.01)
                else:
                    weights[m] = 0.01
            
            total_weight = sum(weights.values())
            if total_weight > 0:
                normalized_weights = {m: w / total_weight for m, w in weights.items()}
                weighted_sum = sum(preds[m] * normalized_weights[m] for m in top_models)
                
                if np.isfinite(weighted_sum):
                    ens.append(int(round(weighted_sum)))
                else:
                    ens.append(np.nan)
            else:
                ens.append(np.nan)
        else:
            ens.append(np.nan)
    
    return ens, mf

# Create KPI sheet
def create_kpi_sheet(excel_file, results_df, score_benchmark):
    """Create KPI sheet with overall and top 10 SKU metrics"""
    
    print("Creating KPI sheet...")
    print(f"Results DF shape: {results_df.shape}")
    print(f"Columns: {results_df.columns.tolist()[:10]}...")
    
    # Calculate using actual historical data (your exact method)
    # Get timeline columns to identify test periods
    timeline_cols = [col for col in results_df.columns 
                    if not col.startswith(('T', 'P', 'Best_', 'Ensembled_'))]

    # Calculate test period range (last 10% of data)
    n_periods = len(timeline_cols)
    test_size = max(1, round(n_periods * 0.10))
    test_start_idx = n_periods - test_size

    # Initialize totals for your exact calculation method
    total_abs_error = 0
    total_error = 0  
    total_demand = 0

    # Get test forecast columns
    test_forecast_cols = [col for col in results_df.columns if col.startswith('T') and 'Best_Forecast' in col]

    print(f"Using {len(test_forecast_cols)} test periods starting from index {test_start_idx}")

    # Calculate using actual historical data (your method)
    for _, row in results_df.iterrows():
        # For each test period
        for i, forecast_col in enumerate(test_forecast_cols):
            if pd.notna(row[forecast_col]):
                forecast = row[forecast_col]
                
                # Get actual from historical timeline data
                actual_col_idx = test_start_idx + i
                if actual_col_idx < len(timeline_cols):
                    actual = row[timeline_cols[actual_col_idx]]
                    
                    if pd.notna(actual) and actual >= 0:
                        error = actual - forecast
                        
                        # Add to totals using your exact method
                        total_demand += actual
                        total_error += error  
                        total_abs_error += abs(error)
    
    # Calculate using your exact formulas
    overall_mae_pct = (total_abs_error / total_demand * 100) if total_demand > 0 else 0
    overall_bias_pct = (total_error / total_demand * 100) if total_demand > 0 else 0
    overall_final_score = overall_mae_pct + abs(overall_bias_pct)

    # Calculate correct enhancement (benchmark - actual score)
    benchmark_pct = score_benchmark * 100 if score_benchmark < 1 else score_benchmark
    benchmark_delta = benchmark_pct - overall_final_score  # Simple difference

    print(f"Calculated KPIs: MAE={overall_mae_pct:.1f}%, Bias={overall_bias_pct:.1f}%, Score={overall_final_score:.1f}%")
    print(f"Totals: ABS_Error={total_abs_error:.0f}, Error={total_error:.0f}, Demand={total_demand:.0f}")
    
    # Get top 10 SKUs
    score_col = 'Best_Score%'
    if score_col in results_df.columns:
        top_10_skus = results_df.nsmallest(10, score_col)
    else:
        top_10_skus = results_df.head(10)  # Fallback
    
    # Create KPI data
    kpi_data = {
        'Metric': ['Overall MAE%', 'Overall Bias%', 'Overall Final Score', 'Benchmark Delta %'],
        'Value': [overall_mae_pct, overall_bias_pct, overall_final_score, benchmark_delta]
    }
    
    # Top 10 SKU data
    top10_data = []
    for idx, (sku, row) in enumerate(top_10_skus.iterrows()):
        sku_mae = row.get('Best_SMAPE', 0) * 100 if pd.notna(row.get('Best_SMAPE', np.nan)) else 0
        sku_bias = row.get('Best_Bias%', 0) * 100 if pd.notna(row.get('Best_Bias%', np.nan)) else 0
        sku_final_score = row.get('Best_Score%', 0) * 100 if pd.notna(row.get('Best_Score%', np.nan)) else 0
        
        top10_data.append({
            'Rank': idx + 1,
            'SKU': sku,
            'MAE%': sku_mae,
            'Bias%': sku_bias,
            'Final Score': sku_final_score,
            'Best Model': row.get('Best_Model', 'N/A')
        })
    
    # Write to Excel
    try:
        with pd.ExcelWriter(excel_file, mode='a', if_sheet_exists='replace', engine='openpyxl') as writer:
            # Overall KPIs
            kpi_main_df = pd.DataFrame(kpi_data)
            kpi_main_df.to_excel(writer, sheet_name='KPIs', index=False, startrow=0)
            
            # Top 10 SKUs
            top10_df = pd.DataFrame(top10_data)
            top10_df.to_excel(writer, sheet_name='KPIs', index=False, startrow=6)
            
            # Benchmark reference
            benchmark_df = pd.DataFrame({
                'Benchmark Score': [score_benchmark],
                'Source': ['Input Data Sheet G1'],
                'Generated': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            })
            benchmark_df.to_excel(writer, sheet_name='KPIs', index=False, startrow=20)
            
        print("KPI sheet created successfully!")
        
    except Exception as e:
        print(f"Error writing KPI sheet: {e}")
    
    return pd.DataFrame(kpi_data), pd.DataFrame(top10_data)

# INTEGRATED DASHBOARD GENERATION
def create_integrated_dashboard(results_df, kpi_df, top10_df, score_benchmark, output_path):
    """Create complete 3x4 dashboard layout in single PNG"""
    
    # Get timeline columns for historical data access
    timeline_cols = [col for col in results_df.columns 
                    if not col.startswith(('T', 'P', 'Best_', 'Ensembled_'))]
    
    # Set up the figure with GridSpec for precise control
    fig = plt.figure(figsize=(24, 32))  # Large figure for 3x4 layout
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.2)
    
    # Extract overall metrics with safety checks
    overall_mae = 0 if pd.isna(kpi_df.iloc[0]['Value']) else kpi_df.iloc[0]['Value']
    overall_bias = 0 if pd.isna(kpi_df.iloc[1]['Value']) else kpi_df.iloc[1]['Value']
    overall_final_score = 0 if pd.isna(kpi_df.iloc[2]['Value']) else kpi_df.iloc[2]['Value']
    benchmark_delta = 0 if pd.isna(kpi_df.iloc[3]['Value']) else kpi_df.iloc[3]['Value']
    
    # ROW 1: Main dashboard charts


    # Count actual wins (each item has only one winner)
    choice_counts = results_df['Best_Choice'].value_counts()

    labels = []
    sizes = []
    colors = []
    # ADD THIS LINE to create ax1 before using it:
    ax1 = fig.add_subplot(gs[0, 0])  # Create the first chart in the top-left position
    
    if 'BM' in choice_counts:
        labels.append('Best Model')
        sizes.append(choice_counts['BM'])
        colors.append(SIA_COLORS['primary_blue'])

    if 'EN' in choice_counts:
        labels.append('Ensemble') 
        sizes.append(choice_counts['EN'])
        colors.append(SIA_COLORS['accent_gold'])

    # Ensure we only show actual winners
    total_items = len(results_df)
    winning_items = sum(sizes)
    if winning_items < total_items:
        # Add "No Clear Winner" if some items don't have BM or EN
        labels.append('No Clear Winner')
        sizes.append(total_items - winning_items)
        colors.append(SIA_COLORS['dark_gray'])
    
    if sizes:
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, 
                                          autopct='%1.1f%%', startangle=90,
                                          textprops={'fontsize': 12, 'fontweight': 'bold'})
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(14)
            autotext.set_fontweight('bold')
    
    ax1.set_title('Model Selection Distribution\nBest Model vs Ensemble', 
                fontsize=16, fontweight='bold', pad=20)
    
    # 2. KPI CARD (Center)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    # Background
    background = patches.Rectangle((0.5, 1), 9, 8, facecolor=SIA_COLORS['light_gray'], 
                                  edgecolor=SIA_COLORS['primary_blue'], linewidth=3)
    ax2.add_patch(background)
    
    # Title
    ax2.text(5, 8.5, 'SIA FORECASTING KPIs', 
            fontsize=18, fontweight='bold', ha='center', 
            color=SIA_COLORS['primary_blue'])
    
    # KPI boxes
    kpi_data = [
        ('MAE%', f'{overall_mae:.1f}%', SIA_COLORS['primary_blue']),
        ('Bias%', f'{overall_bias:+.1f}%', SIA_COLORS['accent_gold']),
        ('Score', f'{overall_final_score:.1f}%', SIA_COLORS['success_green']),
        ('Enhancement', f'{benchmark_delta:+.1f}%' + (' ↑' if benchmark_delta > 0 else ' ↓'), 
        SIA_COLORS['success_green'] if benchmark_delta > 0 else SIA_COLORS['warning_red'])
    ]
    positions = [(2, 6.5), (8, 6.5), (2, 3.5), (8, 3.5)]
    
    for i, ((label, value, color), (x, y)) in enumerate(zip(kpi_data, positions)):
        box = patches.Rectangle((x-1.5, y-1), 3, 2, facecolor=color, alpha=0.8)
        ax2.add_patch(box)
        
        ax2.text(x, y+0.2, value, fontsize=16, fontweight='bold', 
                ha='center', va='center', color='white')
        ax2.text(x, y-0.5, label, fontsize=11, fontweight='bold', 
                ha='center', va='center', color='white')
    
    # Footer
    ax2.text(5, 0.5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
            fontsize=10, ha='center', style='italic', 
            color=SIA_COLORS['dark_gray'])
    
    # 3. SEPARATE COLUMN CHART (Right)
    ax3 = fig.add_subplot(gs[0, 2])

    # Prepare data using correctly calculated values
    mae_value = overall_mae
    bias_abs = abs(overall_bias)  # Always plot absolute value
    score_value = overall_final_score

    # Replace NaN values with 0 to avoid plotting errors
    values = []
    for val in [mae_value, bias_abs, score_value]:
        values.append(0 if pd.isna(val) else val)

    # Create proper bias label showing sign but plotting absolute
    bias_sign = "+ve" if overall_bias >= 0 else "-ve" 
    bias_label = f'Bias% ({bias_sign})'
    labels = ['MAE%', bias_label, 'Score']

    # Colors
    bias_color = SIA_COLORS['bias_positive'] if overall_bias >= 0 else SIA_COLORS['bias_negative']
    colors = [SIA_COLORS['primary_blue'], bias_color, SIA_COLORS['success_green']]

    # Create separate columns
    x_positions = [0, 1, 2]

    bars = ax3.bar(x_positions, values, color=colors, alpha=0.8, width=0.6)


    # Add benchmark line
    ax3.axhline(y=score_benchmark*100, color=SIA_COLORS['warning_red'], 
            linestyle='--', linewidth=3, label=f'Benchmark: {score_benchmark*100:.1f}%')

    # Customize chart
    ax3.set_xticks(x_positions)
    ax3.set_xticklabels(labels)
    ax3.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Performance Metrics vs Benchmark\n(Score = MAE% + |Bias%|)', 
                fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # ROWS 2-4: Top 10 SKU Charts
    sku_positions = [(i, j) for i in range(1, 4) for j in range(3)]  # 3x3 grid for exactly 9 charts
    
    # Get top 10 SKUs for detailed charts
    top_9_skus = results_df.nsmallest(9, 'Best_Score%').head(9)
    
    # Find test columns for time series
    test_forecast_cols = [col for col in results_df.columns if col.startswith('T') and 'Best_Forecast' in col]
    test_periods = sorted(list(set([col.split()[0] for col in test_forecast_cols])))
    
    # Future forecast columns
    future_cols = [col for col in results_df.columns if col.startswith('P') and 'Forecast_Best' in col]
    
    for idx, ((sku, row), (grid_row, grid_col)) in enumerate(zip(top_9_skus.iterrows(), sku_positions)):
        if idx >= 9:  # Limit to 9 charts
            break
            
        ax = fig.add_subplot(gs[grid_row, grid_col])
        
        # Extract test data
        test_forecasts = []
        test_smapes = []
        
        for period in test_periods:
            forecast_col = f"{period} Best_Forecast"
            smape_col = f"{period} Best_SMAPE"
            
            if forecast_col in row and pd.notna(row[forecast_col]):
                test_forecasts.append(row[forecast_col])
                smape_val = row[smape_col] if smape_col in row else 0
                test_smapes.append(smape_val)
            else:
                test_forecasts.append(np.nan)
                test_smapes.append(np.nan)
        
        # Create approximate actuals from forecasts and SMAPE
        test_actuals = []
        for i, (forecast, smape_val) in enumerate(zip(test_forecasts, test_smapes)):
            if not np.isnan(forecast) and not np.isnan(smape_val):
                # Approximate actual (this is simplified)
                error_approx = forecast * smape_val if smape_val > 0 else 0
                actual_approx = forecast - error_approx
                test_actuals.append(max(0, actual_approx))
            else:
                test_actuals.append(np.nan)
        
        # Extract future forecasts
        future_forecasts = []
        for col in future_cols:
            if col in row and pd.notna(row[col]):
                future_forecasts.append(row[col])
            else:
                future_forecasts.append(np.nan)
        
        # Plot the data
        x_range = range(len(test_periods))

        # Extended actual line (double the test periods for better view)
        extended_periods = len(test_periods) * 2
        start_idx = max(0, len(timeline_cols) - extended_periods)
        
        
        # Get extended actual data from historical columns
        extended_actuals = []
        extended_x = []

        # Add historical actual data (extended range)
        for i in range(start_idx, len(timeline_cols)):
            hist_value = row.get(timeline_cols[i], np.nan)
            if pd.notna(hist_value):
                extended_actuals.append(hist_value)
                extended_x.append(i - start_idx)

        # Plot extended actual line
        if extended_actuals:
            ax.plot(extended_x, extended_actuals, 'o-', color=SIA_COLORS['primary_blue'], 
                    linewidth=2, markersize=6, label='Actual', alpha=0.8)

        # Continuous forecast line: Test periods + Future periods as ONE line
        all_forecasts = test_forecasts + future_forecasts

        # Calculate continuous x-positions for the entire forecast
        # Start forecast at the beginning of test periods and continue through future
        test_start_x = max(extended_x) - len(test_periods) + 1 if extended_x else 0
        forecast_x_positions = list(range(test_start_x, test_start_x + len(all_forecasts)))

        if all_forecasts:
            valid_forecast_x = []
            valid_forecast_y = []
            for i, val in enumerate(all_forecasts):
                if pd.notna(val):
                    valid_forecast_x.append(forecast_x_positions[i])
                    valid_forecast_y.append(val)
                    
            if valid_forecast_x:  # Only plot if we have valid points
                ax.plot(valid_forecast_x, valid_forecast_y, 's-', color=SIA_COLORS['accent_gold'], 
                    linewidth=2, markersize=4, label='Forecast', alpha=0.8)
        # Customize individual SKU chart
        model_name = row.get("Best_Model", "Unknown")
        if pd.notna(row.get("Best_Score%")):
            # Convert to percentage if needed
            score_val = row['Best_Score%']
            if score_val < 1:  # If it's decimal (0.098), convert to percentage
                score_pct = score_val * 100
            else:  # If already percentage
                score_pct = score_val
            score_text = f"{score_pct:.1f}%"
        else:
            score_text = "N/A"
        ax.set_title(f'#{idx+1}: {sku}\n{model_name} | Score: {score_text}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Period', fontsize=10)
        ax.set_ylabel('Demand', fontsize=10)
        # Consistent legend position to avoid overlap
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Add error metrics
        if pd.notna(row.get('Best_SMAPE')):
            error_text = f"SMAPE: {row['Best_SMAPE']:.3f}"
        else:
            error_text = "SMAPE: N/A"
            
        if 'Best_Bias%' in row and pd.notna(row['Best_Bias%']):
            error_text += f" | Bias: {row['Best_Bias%']:.3f}%"
        
        ax.text(0.02, 0.98, error_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))
    
    # Add main title
    fig.suptitle('SIA FORECASTING PERFORMANCE DASHBOARD', 
                fontsize=24, fontweight='bold', y=0.98)
    
    # Save the complete dashboard
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Complete dashboard saved: {output_path}")

def get_app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

# MAIN EXECUTION WITH INTEGRATED DASHBOARD
def main(file_name):
    print("="*60)
    print("SIA FORECASTING SYSTEM - STARTING ANALYSIS")
    print("="*60)
    
    # Read input with score benchmark
    forecast_horizon, score_benchmark, raw_df = read_universal_input_sheet(file_name)
    timeline_cols = raw_df.columns.tolist()
    items = raw_df.index.tolist()
    
    # Seasonal period detection
    guessed_seasonal = guess_seasonal_period(timeline_cols)
    if guessed_seasonal == 52 and raw_df.shape[1] < 104:
        print("Insufficient weekly history detected. Forcing seasonal period to 12.")
        seasonal_period = 12
    else:
        seasonal_period = guessed_seasonal
    
    print(f"Detected Seasonal Period: {seasonal_period}")
    print(f"Forecast Horizon: {forecast_horizon} periods")
    print(f"Score Benchmark: {score_benchmark}")
    print(f"Score Benchmark from G1: {score_benchmark}")
    print(f"Score Benchmark as %: {score_benchmark * 100}%")
    
    # Get forecast functions
    forecast_funcs = get_forecast_functions(seasonal_period)
    
    # Setup output directory
    output_dir = Path(get_app_dir()) / "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    partial = output_dir / "Partial_Results.xlsx"
    if os.path.exists(partial):
        df0 = pd.read_excel(partial, index_col="ITEM")
        final = list(df0.to_dict("records"))
    else:
        final = []

    n = len(timeline_cols)
    test_size = max(1, round(n * 0.10))
    test_idx = list(range(n - test_size, n))

    print(f"Processing {len(items)} items with {test_size} test periods...")
    print("Model ensemble includes: " + ", ".join(model_names))

    for i, item in enumerate(items, 1):
        series = raw_df.loc[item].values
        
        # Skip items with insufficient data
        if len(series) < 5 or np.sum(~np.isnan(series)) < 3:
            print(f"{i:3d}: {item} —----> Skipped (insufficient data)")
            continue
        
        # Rolling forecast
        periods, agg_sm, agg_err, agg_act = rolling_forecast_for_product(
            series, test_idx, seasonal_period, forecast_funcs
        )
        
        # Best model selection
        best, bst_sm, bias, bst_score = choose_best_model(agg_sm, agg_err, agg_act)
        
        # Enhanced ensemble scoring
        ens_scores = {}
        for m in model_names:
            if agg_sm[m] and agg_act[m] > 0:
                avg_smape = np.mean(agg_sm[m])
                bias_val = agg_err[m] / agg_act[m] if agg_act[m] != 0 else 0
                std_smape = np.std(agg_sm[m]) if len(agg_sm[m]) > 1 else 0
                stability_penalty = std_smape * 0.05
                ens_scores[m] = avg_smape + abs(bias_val) + stability_penalty
            else:
                ens_scores[m] = np.inf
        
        # Build test results
        ens_smapes = []
        ens_err = 0.0
        ens_act = 0.0
        test_res = {}
        
        for pi, p in enumerate(periods, 1):
            # Model forecasts & smapes
            for m in model_names:
                test_res[f"T{pi} {m} Forecast"] = p["fc"][m]
                test_res[f"T{pi} {m} SMAPE"] = p["sm"][m]
            
            # Best model results
            tbest_fc = p["fc"][best]
            tbest_sm = p["sm"][best]
            test_res[f"T{pi} Best_Forecast"] = tbest_fc
            test_res[f"T{pi} Best_SMAPE"] = tbest_sm
            
            # Enhanced ensemble
            preds = {m: p["fc"][m] for m in model_names if not np.isnan(p["fc"][m])}
            valid_models = [m for m in preds if ens_scores[m] < np.inf]
            
            if valid_models:
                top_models = sorted(valid_models, key=lambda m: ens_scores[m])[:3]
                
                weights = {}
                for m in top_models:
                    score = ens_scores[m]
                    weights[m] = 1 / (score + 0.01)
                
                total_weight = sum(weights.values())
                if total_weight > 0:
                    normalized_weights = {m: w / total_weight for m, w in weights.items()}
                    weighted_sum = sum(preds[m] * normalized_weights[m] for m in top_models)
                    te = int(round(weighted_sum)) if np.isfinite(weighted_sum) else np.nan
                else:
                    te = np.nan
            else:
                te = np.nan
            
            test_res[f"T{pi} Ensemble_Forecast"] = te
            se = smape([p["actual"]], [te]) if not np.isnan(te) else np.nan
            test_res[f"T{pi} Ensemble_SMAPE"] = se
            
            # Aggregate ensemble metrics
            if not np.isnan(se):
                ens_smapes.append(se)
            if not np.isnan(te):
                ens_err += te - p["actual"]
                ens_act += p["actual"]
        
        # Calculate ensemble KPIs
        ens_avg_sm = np.mean(ens_smapes) if ens_smapes else np.nan
        ens_bias = ens_err / ens_act if ens_act else np.nan
        ens_score = (ens_avg_sm + abs(ens_bias)) if not np.isnan(ens_avg_sm) else np.inf
        
        # Future forecasts
        future_best = forecast_next_periods(series, best, forecast_horizon, seasonal_period, forecast_funcs)
        future_ens, _ = forecast_all_future(series, forecast_horizon, ens_scores, seasonal_period, forecast_funcs)
        
        # Build result row
        row = {
            "ITEM": item,
            "Ensembled": best,
            "Ensembled_Avg_SMAPE": ens_avg_sm,
            "Ensembled_Bias%": ens_bias,
            "Ensembled_Score%": ens_score,
            "Best_Model": best,
            "Best_Model_Avg_SMAPE": bst_sm,
            "Best_Model_Bias%": bias,
            "Best_Model_Score%": bst_score,
        }

        # Determine best choice
        row["Best_Choice"] = (
            "EN" if (np.isfinite(ens_score) and np.isfinite(bst_score) and ens_score < bst_score)
            else "BM" if (np.isfinite(ens_score) and np.isfinite(bst_score)) else np.nan
        )
        
        row["Best_SMAPE"] = (
            ens_avg_sm if row["Best_Choice"] == "EN"
            else bst_sm if row["Best_Choice"] == "BM" else np.nan
        )
        
        row["Best_Bias%"] = (
            ens_bias if row["Best_Choice"] == "EN"
            else bias if row["Best_Choice"] == "BM" else np.nan
        )
        
        row["Best_Score%"] = (
            np.nanmin([bst_score, ens_score])
            if (bst_score is not None and ens_score is not None and 
                not np.isnan(bst_score) and not np.isnan(ens_score))
            else np.nan
        )
        
        # Add test results and historical data
        # Add test results FIRST
        row.update(test_res)

        # Then add historical data
        for j, col in enumerate(timeline_cols):
            row[col] = series[j]

        # Then add future forecasts
        for j in range(forecast_horizon):
            row[f"P{j+1} Forecast_Best"] = future_best[j]
            row[f"P{j+1} Ensemble_Forecast"] = future_ens[j]
        # Replace inf values with None (will show as blank in Excel)
        for key, value in row.items():
            if isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
                row[key] = None

        final.append(row)
        # Replace it with this clean and reliable version:
        if isinstance(bst_score, (int, float)) and bst_score is not None:
            score_display = f"{bst_score:.4f}"
        else:
            score_display = "N/A"
        print(f"{i:3d}: {item} —----> Complete (Best: {best}, Score: {score_display})")

        
        # Save partial results
        pd.DataFrame(final).set_index("ITEM").to_excel(partial)

    # Save final results with cleaned data
    final_df = pd.DataFrame(final).set_index("ITEM")

    # Clean up inf and nan values before saving to Excel
    final_df = final_df.replace([np.inf, -np.inf], None)  # Replace infinity with None
    final_df = final_df.where(pd.notna(final_df), None)   # Replace NaN with None

    results_file = output_dir / "SIA_Forecasting_Results.xlsx"
    final_df.to_excel(results_file)
    
    # Create KPI sheet
    print("\n" + "="*60)
    print("CREATING KPI SHEET")
    print("="*60)
    kpi_df, top10_df = create_kpi_sheet(results_file, final_df, score_benchmark)
    print("KPI sheet creation completed.")
    
    print("\n" + "="*60)
    print("FORECASTING COMPLETE - GENERATING DASHBOARD")
    print("="*60)
    
    # Generate integrated dashboard
    dashboard_file = output_dir / "SIA_Complete_Dashboard.png"
    create_integrated_dashboard(final_df, kpi_df, top10_df, score_benchmark, dashboard_file)
    
    # Summary statistics
    total_items = len(final)
    ensemble_wins = len(final_df[final_df['Best_Choice'] == 'EN'])
    model_wins = len(final_df[final_df['Best_Choice'] == 'BM'])
    avg_score = final_df['Best_Score%'].mean()
    
    print(f"\n📊 FINAL RESULTS SUMMARY:")
    print(f"   • Total SKUs Processed: {total_items}")
    print(f"   • Ensemble: {ensemble_wins} ({ensemble_wins/total_items*100:.1f}%)")
    print(f"   • Best Model: {model_wins} ({model_wins/total_items*100:.1f}%)")
    print(f"   • Average Accuracy Score: {avg_score:.4f}")
    print(f"   • Score Benchmark: {score_benchmark}")
    
    print(f"\n📁 OUTPUT FILES GENERATED:")
    print(f"   • Forecasting Results: {results_file}")
    print(f"   • Complete Dashboard: {dashboard_file}")
    print(f"   • KPI Sheet: Included in results file")
    
    print("\n✅ SIA FORECASTING ANALYSIS COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main("SIA-Forecasting input.xlsx")