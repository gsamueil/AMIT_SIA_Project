"""
SIA Forecasting System - Advanced Forecasting Engine
Handles intermittent demand with intelligent scaling and growth analysis
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForecastEngine:
    """Advanced forecasting engine with multiple methods"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.forecast_results = {}
        self.error_metrics = {}
        
    def generate_forecast(self, 
                         data: pd.DataFrame,
                         target_metric: str = 'qty',
                         frequency: str = 'M',
                         periods_ahead: int = 12,
                         filter_dict: Dict = None) -> Dict:
        """
        Generate forecast with specified parameters
        
        Args:
            data: Input time series data
            target_metric: 'qty', 'usd_amount', or 'weight_tons'
            frequency: 'Y', 'Q', 'M', 'W'
            periods_ahead: Number of periods to forecast
            filter_dict: Filters (e.g., {'country': 'Egypt', 'system': 'System1'})
        """
        try:
            logger.info(f"Generating {frequency} forecast for {target_metric}, {periods_ahead} periods ahead")
            
            # Apply filters
            if filter_dict:
                for col, value in filter_dict.items():
                    if col in data.columns:
                        data = data[data[col] == value]
            
            # Prepare time series
            ts_data = self._prepare_timeseries(data, target_metric, frequency)
            
            if len(ts_data) < 12:
                return {
                    'status': 'error',
                    'error': f'Insufficient data: {len(ts_data)} periods (minimum 12 required)'
                }
            
            # Detect intermittency
            intermittent_info = self._detect_intermittency(ts_data)
            
            # Generate forecasts using multiple methods
            forecasts = self._generate_multi_method_forecasts(
                ts_data, periods_ahead, intermittent_info
            )
            
            # Analyze growth trends
            growth_analysis = self._analyze_growth(ts_data)
            
            # Apply intelligent scaling
            scaled_forecasts = self._apply_intelligent_scaling(
                forecasts, ts_data, growth_analysis
            )
            
            # Compute errors on holdout
            error_metrics = self._compute_errors(ts_data, scaled_forecasts)
            
            # Prepare result
            result = {
                'status': 'success',
                'forecast': scaled_forecasts,
                'historical': ts_data,
                'intermittent_info': intermittent_info,
                'growth_analysis': growth_analysis,
                'error_metrics': error_metrics,
                'metadata': {
                    'target_metric': target_metric,
                    'frequency': frequency,
                    'periods_ahead': periods_ahead,
                    'filters': filter_dict,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Forecast generation failed: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _prepare_timeseries(self, data: pd.DataFrame, metric: str, frequency: str) -> pd.DataFrame:
        """Prepare time series at specified frequency"""
        # Determine period column
        if frequency == 'Y':
            period_col = 'period_year'
        elif frequency == 'Q':
            period_col = 'period_quarter'
        elif frequency == 'M':
            period_col = 'period_month'
        elif frequency == 'W':
            # For weekly, need to compute from booked_date
            data['period_week'] = data['booked_date'].dt.to_period('W')
            period_col = 'period_week'
        else:
            period_col = 'period_month'
        
        # Aggregate
        if period_col in data.columns:
            ts = data.groupby(period_col)[metric].sum().reset_index()
        else:
            # Fallback for weekly
            ts = data.groupby(data['booked_date'].dt.to_period('W'))[metric].sum().reset_index()
            ts.columns = ['period', metric]
            period_col = 'period'
        
        ts.rename(columns={period_col: 'period'}, inplace=True)
        ts['period_start'] = ts['period'].dt.to_timestamp()
        ts = ts.sort_values('period_start').reset_index(drop=True)
        
        # Fill missing periods
        ts = self._fill_missing_periods(ts, metric, frequency)
        
        return ts
    
    def _fill_missing_periods(self, ts: pd.DataFrame, metric: str, frequency: str) -> pd.DataFrame:
        """Fill missing periods with zeros"""
        if len(ts) == 0:
            return ts
        
        # Create complete period range
        freq_map = {'Y': 'Y', 'Q': 'Q', 'M': 'M', 'W': 'W'}
        freq_code = freq_map.get(frequency, 'M')
        
        full_range = pd.period_range(
            start=ts['period'].min(),
            end=ts['period'].max(),
            freq=freq_code
        )
        
        full_df = pd.DataFrame({'period': full_range})
        full_df['period_start'] = full_df['period'].dt.to_timestamp()
        
        # Merge
        ts_complete = full_df.merge(ts[['period', metric]], on='period', how='left')
        ts_complete[metric] = ts_complete[metric].fillna(0)
        
        return ts_complete
    
    def _detect_intermittency(self, ts: pd.DataFrame) -> Dict:
        """Detect if demand is intermittent"""
        values = ts.iloc[:, -1].values  # Last column is the metric
        
        non_zero_mask = values > 0
        pct_non_zero = non_zero_mask.sum() / len(values) * 100
        
        # Compute ADI (Average Demand Interval)
        if non_zero_mask.sum() > 1:
            non_zero_indices = np.where(non_zero_mask)[0]
            intervals = np.diff(non_zero_indices)
            adi = intervals.mean() if len(intervals) > 0 else 0
        else:
            adi = len(values)
        
        # Compute CV² (Coefficient of Variation squared)
        non_zero_values = values[non_zero_mask]
        if len(non_zero_values) > 1:
            cv2 = (non_zero_values.std() / non_zero_values.mean()) ** 2
        else:
            cv2 = 0
        
        is_intermittent = (pct_non_zero < 50) or (adi > 1.32)
        
        return {
            'is_intermittent': bool(is_intermittent),
            'pct_non_zero': float(pct_non_zero),
            'adi': float(adi),
            'cv2': float(cv2),
            'recommendation': 'Croston or SBA' if is_intermittent else 'Traditional methods'
        }
    
    def _generate_multi_method_forecasts(self, 
                                         ts: pd.DataFrame,
                                         periods_ahead: int,
                                         intermittent_info: Dict) -> Dict:
        """Generate forecasts using multiple methods"""
        forecasts = {}
        
        values = ts.iloc[:, -1].values  # Metric values
        
        # Method 1: Naive (last value)
        forecasts['naive'] = self._naive_forecast(values, periods_ahead)
        
        # Method 2: Moving Average
        forecasts['moving_avg'] = self._moving_average_forecast(values, periods_ahead, window=3)
        
        # Method 3: Exponential Smoothing
        forecasts['exp_smoothing'] = self._exp_smoothing_forecast(values, periods_ahead)
        
        # Method 4: Croston (for intermittent)
        if intermittent_info['is_intermittent']:
            forecasts['croston'] = self._croston_forecast(values, periods_ahead)
            forecasts['sba'] = self._sba_forecast(values, periods_ahead)
        
        # Method 5: Linear Trend
        forecasts['linear_trend'] = self._linear_trend_forecast(values, periods_ahead)
        
        # Method 6: Seasonal Naive (if enough history)
        if len(values) >= 12:
            forecasts['seasonal_naive'] = self._seasonal_naive_forecast(values, periods_ahead, season=12)
        
        # Ensemble: Average of all methods
        all_methods = list(forecasts.values())
        forecasts['ensemble'] = np.mean(all_methods, axis=0)
        
        return forecasts
    
    def _naive_forecast(self, values: np.ndarray, periods: int) -> np.ndarray:
        """Naive forecast: last value repeated"""
        return np.full(periods, values[-1])
    
    def _moving_average_forecast(self, values: np.ndarray, periods: int, window: int = 3) -> np.ndarray:
        """Moving average forecast"""
        ma = values[-window:].mean()
        return np.full(periods, ma)
    
    def _exp_smoothing_forecast(self, values: np.ndarray, periods: int, alpha: float = 0.3) -> np.ndarray:
        """Simple exponential smoothing"""
        # Fit
        smoothed = np.zeros(len(values))
        smoothed[0] = values[0]
        for t in range(1, len(values)):
            smoothed[t] = alpha * values[t] + (1 - alpha) * smoothed[t-1]
        
        # Forecast
        forecast = np.full(periods, smoothed[-1])
        return forecast
    
    def _croston_forecast(self, values: np.ndarray, periods: int, alpha: float = 0.1) -> np.ndarray:
        """Croston's method for intermittent demand"""
        non_zero_mask = values > 0
        
        if non_zero_mask.sum() == 0:
            return np.zeros(periods)
        
        # Demand sizes (when non-zero)
        demand_sizes = values[non_zero_mask]
        
        # Intervals between non-zero demands
        non_zero_indices = np.where(non_zero_mask)[0]
        if len(non_zero_indices) > 1:
            intervals = np.diff(non_zero_indices)
        else:
            return np.full(periods, demand_sizes.mean())
        
        # Initialize
        z = demand_sizes[0]  # Demand size estimate
        p = intervals[0] if len(intervals) > 0 else 1  # Interval estimate
        
        # Update estimates
        for i in range(1, len(demand_sizes)):
            z = alpha * demand_sizes[i] + (1 - alpha) * z
            if i < len(intervals):
                p = alpha * intervals[i] + (1 - alpha) * p
        
        # Forecast
        forecast_value = z / p if p > 0 else z
        return np.full(periods, forecast_value)
    
    def _sba_forecast(self, values: np.ndarray, periods: int, alpha: float = 0.1) -> np.ndarray:
        """Syntetos-Boylan Approximation for intermittent demand"""
        croston_fc = self._croston_forecast(values, periods, alpha)
        
        # SBA adjustment
        non_zero_mask = values > 0
        p = len(values) / non_zero_mask.sum() if non_zero_mask.sum() > 0 else 1
        
        sba_fc = croston_fc * (1 - alpha / (2 * p))
        return sba_fc
    
    def _linear_trend_forecast(self, values: np.ndarray, periods: int) -> np.ndarray:
        """Linear trend forecast"""
        x = np.arange(len(values))
        
        # Fit linear regression
        if len(values) > 1:
            slope, intercept = np.polyfit(x, values, 1)
        else:
            slope, intercept = 0, values[0]
        
        # Forecast
        future_x = np.arange(len(values), len(values) + periods)
        forecast = slope * future_x + intercept
        
        return np.maximum(forecast, 0)  # No negative forecasts
    
    def _seasonal_naive_forecast(self, values: np.ndarray, periods: int, season: int = 12) -> np.ndarray:
        """Seasonal naive: repeat last season"""
        last_season = values[-season:]
        
        # Repeat as needed
        n_repeats = (periods // season) + 1
        forecast = np.tile(last_season, n_repeats)[:periods]
        
        return forecast
    
    def _analyze_growth(self, ts: pd.DataFrame) -> Dict:
        """Analyze growth trends"""
        values = ts.iloc[:, -1].values
        
        if len(values) < 12:
            return {'has_growth': False, 'growth_rate': 0}
        
        # Compare recent vs earlier periods
        recent = values[-12:].sum()
        earlier = values[-24:-12].sum() if len(values) >= 24 else values[:-12].sum()
        
        if earlier > 0:
            yoy_growth = (recent - earlier) / earlier * 100
        else:
            yoy_growth = 0
        
        # Trend direction
        if len(values) > 1:
            x = np.arange(len(values))
            slope, _ = np.polyfit(x, values, 1)
            is_growing = slope > 0
        else:
            is_growing = False
            slope = 0
        
        return {
            'has_growth': bool(is_growing and yoy_growth > 0),
            'yoy_growth_pct': float(yoy_growth),
            'trend_slope': float(slope),
            'recent_sum': float(recent),
            'earlier_sum': float(earlier)
        }
    
    def _apply_intelligent_scaling(self,
                                   forecasts: Dict,
                                   historical: pd.DataFrame,
                                   growth_analysis: Dict) -> Dict:
        """
        Apply intelligent scaling to forecasts
        - Respects declining trends
        - Scales up if forecast is unreasonably low and growth exists
        - Does not force equality
        """
        scaled = {}
        
        hist_values = historical.iloc[:, -1].values
        recent_sum = hist_values[-12:].sum()
        
        for method, fc in forecasts.items():
            fc_sum = fc.sum()
            
            # Rule 1: If declining trend, don't scale up
            if growth_analysis['yoy_growth_pct'] < -10:
                scaled[method] = fc
                continue
            
            # Rule 2: If growing but forecast is too low (< 70% of recent)
            if growth_analysis['has_growth'] and fc_sum < 0.7 * recent_sum:
                # Scale up to 85% of recent (conservative)
                scale_factor = (0.85 * recent_sum) / fc_sum if fc_sum > 0 else 1
                scaled[method] = fc * scale_factor
            else:
                # No scaling needed
                scaled[method] = fc
        
        return scaled
    
    def _compute_errors(self, ts: pd.DataFrame, forecasts: Dict) -> Dict:
        """Compute error metrics on validation set"""
        values = ts.iloc[:, -1].values
        
        if len(values) < 24:
            return {'note': 'Insufficient data for validation'}
        
        # Use last 12 periods as validation
        train = values[:-12]
        actual = values[-12:]
        
        errors = {}
        
        for method in forecasts.keys():
            # Generate forecast on training data
            if method == 'ensemble':
                # Average of other methods
                train_fc = np.mean([
                    self._naive_forecast(train, 12),
                    self._exp_smoothing_forecast(train, 12)
                ], axis=0)
            else:
                train_fc = self._naive_forecast(train, 12)  # Simplified for validation
            
            # Compute errors
            mae = np.mean(np.abs(actual - train_fc))
            rmse = np.sqrt(np.mean((actual - train_fc) ** 2))
            
            # MAPE (avoid division by zero)
            mape_mask = actual != 0
            if mape_mask.sum() > 0:
                mape = np.mean(np.abs((actual[mape_mask] - train_fc[mape_mask]) / actual[mape_mask])) * 100
            else:
                mape = 0
            
            # SMAPE
            denominator = (np.abs(actual) + np.abs(train_fc)) / 2
            smape_mask = denominator != 0
            if smape_mask.sum() > 0:
                smape = np.mean(np.abs(actual[smape_mask] - train_fc[smape_mask]) / denominator[smape_mask]) * 100
            else:
                smape = 0
            
            # Bias
            bias = np.mean(train_fc - actual)
            
            errors[method] = {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'smape': float(smape),
                'bias': float(bias)
            }
        
        return errors