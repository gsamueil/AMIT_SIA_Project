"""
SIA Forecasting System - Advanced Preprocessing Module
Enterprise-Grade Data Processing for Supply Chain Forecasting
"""

import pandas as pd
import numpy as np
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataProcessor:
    """Advanced data processor for SIA forecasting system"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.validation_results = {}
        self.processed_data = None
        self.metrics_summary = {}
        
    def process_file(self, filepath: str, progress_callback: Optional[Callable] = None) -> Dict:
        """Main processing pipeline"""
        try:
            self._report_progress(1, "Reading Excel file...", progress_callback)
            raw_data = self._read_excel(filepath)
            
            self._report_progress(2, "Loading item lists...", progress_callback)
            standard_items, prime_items = self._load_item_lists(filepath)
            
            self._report_progress(3, "Validating data quality...", progress_callback)
            validation = self._validate_data(raw_data, standard_items, prime_items)
            self.validation_results = validation
            
            self._report_progress(4, "Cleaning and transforming data...", progress_callback)
            clean_data = self._clean_data(raw_data, standard_items, prime_items)
            
            self._report_progress(5, "Aggregating split quantities...", progress_callback)
            aggregated = self._aggregate_split_quantities(clean_data)
            
            self._report_progress(6, "Computing metrics (Qty, USD, Tons)...", progress_callback)
            final_data = self._compute_metrics(aggregated)
            
            self._report_progress(7, "Creating multi-dimensional views...", progress_callback)
            dimensional_data = self._create_dimensional_views(final_data)
            
            self._report_progress(8, "Computing summary statistics...", progress_callback)
            summary = self._compute_summary(final_data, dimensional_data)
            self.metrics_summary = summary
            
            self._report_progress(9, "Saving processed data...", progress_callback)
            output_paths = self._save_outputs(final_data, dimensional_data, summary, validation)
            
            self._report_progress(10, "Processing complete!", progress_callback)
            
            return {
                'status': 'success',
                'output_paths': output_paths,
                'summary': summary,
                'validation': validation,
                'data': final_data
            }
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _report_progress(self, step: int, message: str, callback: Optional[Callable]):
        """Report progress to GUI"""
        logger.info(f"Step {step}/10: {message}")
        if callback:
            callback(step, message)
    
    def _read_excel(self, filepath: str) -> pd.DataFrame:
        """Read main data sheet"""
        try:
            df = pd.read_excel(filepath, sheet_name='SOs')
            logger.info(f"Loaded {len(df)} rows from SOs sheet")
            return df
        except Exception as e:
            raise Exception(f"Failed to read Excel file: {str(e)}")
    
    def _load_item_lists(self, filepath: str) -> Tuple[set, set]:
        """Load Standard and Prime item lists"""
        try:
            standard_df = pd.read_excel(filepath, sheet_name='Standard')
            prime_df = pd.read_excel(filepath, sheet_name='Prime')
            
            # Get first column as item codes
            standard_items = set(standard_df.iloc[:, 0].dropna().astype(str).str.strip().str.upper())
            prime_items = set(prime_df.iloc[:, 0].dropna().astype(str).str.strip().str.upper())
            
            logger.info(f"Loaded {len(standard_items)} Standard items, {len(prime_items)} Prime items")
            return standard_items, prime_items
            
        except Exception as e:
            logger.warning(f"Could not load item lists: {str(e)}")
            return set(), set()
    
    def _validate_data(self, df: pd.DataFrame, standard_items: set, prime_items: set) -> Dict:
        """Comprehensive data validation"""
        validation = {
            'timestamp': datetime.now().isoformat(),
            'total_rows': len(df),
            'quality_checks': {},
            'insights': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check required columns
        required_cols = [
            'Ordered Item', 'SO Remaining Qty', 'Booked Date',
            'SO Line Amount EGP', 'Shipped Amount [USD]', 'Not Shipped Amount [USD]'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            validation['warnings'].append(f"Missing columns: {missing_cols}")
        
        # Data quality metrics
        validation['quality_checks']['missing_sku'] = {
            'count': int(df['Ordered Item'].isna().sum()),
            'percentage': float(df['Ordered Item'].isna().sum() / len(df) * 100)
        }
        
        validation['quality_checks']['missing_qty'] = {
            'count': int(df['SO Remaining Qty'].isna().sum()),
            'percentage': float(df['SO Remaining Qty'].isna().sum() / len(df) * 100)
        }
        
        validation['quality_checks']['missing_date'] = {
            'count': int(df['Booked Date'].isna().sum()),
            'percentage': float(df['Booked Date'].isna().sum() / len(df) * 100)
        }
        
        # Duplicate detection
        duplicate_rows = df.duplicated().sum()
        validation['quality_checks']['duplicates'] = int(duplicate_rows)
        
        # Item classification
        if standard_items and prime_items:
            df_clean_items = df['Ordered Item'].dropna().astype(str).str.strip().str.upper()
            in_standard = df_clean_items.isin(standard_items).sum()
            in_prime = df_clean_items.isin(prime_items).sum()
            in_neither = len(df_clean_items) - in_standard
            
            validation['quality_checks']['item_classification'] = {
                'standard_items': int(in_standard),
                'prime_items': int(in_prime),
                'non_standard_items': int(in_neither),
                'standard_pct': float(in_standard / len(df_clean_items) * 100),
                'prime_pct': float(in_prime / len(df_clean_items) * 100)
            }
            
            validation['insights'].append(
                f"{in_standard:,} rows ({in_standard/len(df_clean_items)*100:.1f}%) are Standard items"
            )
            validation['insights'].append(
                f"{in_prime:,} rows ({in_prime/len(df_clean_items)*100:.1f}%) are Prime items"
            )
        
        # Date range
        valid_dates = pd.to_datetime(df['Booked Date'], errors='coerce').dropna()
        if len(valid_dates) > 0:
            date_min = valid_dates.min()
            date_max = valid_dates.max()
            date_span_days = (date_max - date_min).days
            
            validation['quality_checks']['date_range'] = {
                'min': date_min.isoformat(),
                'max': date_max.isoformat(),
                'span_days': int(date_span_days),
                'span_years': float(date_span_days / 365.25)
            }
            
            validation['insights'].append(
                f"Data spans {date_span_days/365.25:.1f} years ({date_min.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')})"
            )
        
        # Dimension cardinality
        dimensions = {
            'unique_items': df['Ordered Item'].nunique(),
            'unique_orders': df['Order Number'].nunique() if 'Order Number' in df else 0,
            'unique_countries': df['Country'].nunique() if 'Country' in df else 0,
            'unique_systems': df['Order System'].nunique() if 'Order System' in df else 0,
            'unique_factories': df['Factory Name'].nunique() if 'Factory Name' in df else 0,
            'unique_cells': df['Cell Columns'].nunique() if 'Cell Columns' in df else 0
        }
        
        validation['quality_checks']['dimensions'] = dimensions
        validation['insights'].append(f"{dimensions['unique_items']:,} unique SKUs")
        validation['insights'].append(f"{dimensions['unique_countries']} countries")
        validation['insights'].append(f"{dimensions['unique_systems']} systems")
        
        # Split quantity detection
        if 'Order Number' in df.columns:
            order_item_groups = df.groupby(['Order Number', 'Ordered Item']).size()
            split_groups = (order_item_groups > 1).sum()
            total_groups = len(order_item_groups)
            
            validation['quality_checks']['split_quantities'] = {
                'split_groups': int(split_groups),
                'total_groups': int(total_groups),
                'split_percentage': float(split_groups / total_groups * 100)
            }
            
            validation['insights'].append(
                f"{split_groups:,} order-item combinations have split quantities ({split_groups/total_groups*100:.1f}%)"
            )
        
        # Recommendations
        if validation['quality_checks']['missing_sku']['percentage'] > 5:
            validation['warnings'].append("⚠️ More than 5% missing SKUs - review data quality")
        
        if validation['quality_checks']['missing_date']['percentage'] > 10:
            validation['warnings'].append("⚠️ More than 10% missing dates - may affect forecasting")
        
        if dimensions['unique_items'] < 100:
            validation['recommendations'].append("Low number of items - consider expanding product catalog")
        
        logger.info(f"Validation complete: {len(validation['insights'])} insights, {len(validation['warnings'])} warnings")
        
        return validation
    
    def _clean_data(self, df: pd.DataFrame, standard_items: set, prime_items: set) -> pd.DataFrame:
        """Clean and transform data"""
        df_clean = df.copy()
        
        # Standardize SKU
        df_clean['sku'] = df_clean['Ordered Item'].astype(str).str.strip().str.upper()
        
        # Remove missing SKUs
        df_clean = df_clean[df_clean['sku'].notna()].copy()
        
        # Parse dates
        df_clean['booked_date'] = pd.to_datetime(df_clean['Booked Date'], errors='coerce')
        df_clean = df_clean[df_clean['booked_date'].notna()].copy()
        
        # Parse quantities
        df_clean['qty'] = pd.to_numeric(df_clean['SO Remaining Qty'], errors='coerce').fillna(0)
        df_clean['shipped_qty'] = pd.to_numeric(df_clean['Shipped Qty'], errors='coerce').fillna(0) if 'Shipped Qty' in df_clean else 0
        df_clean['cancelled_qty'] = pd.to_numeric(df_clean['Cancelled Qty'], errors='coerce').fillna(0) if 'Cancelled Qty' in df_clean else 0
        
        # Parse financial data
        df_clean['egp_amount'] = pd.to_numeric(df_clean['SO Line Amount EGP'], errors='coerce').fillna(0)
        df_clean['shipped_usd'] = pd.to_numeric(df_clean['Shipped Amount [USD]'], errors='coerce').fillna(0) if 'Shipped Amount [USD]' in df_clean else 0
        df_clean['not_shipped_usd'] = pd.to_numeric(df_clean['Not Shipped Amount [USD]'], errors='coerce').fillna(0) if 'Not Shipped Amount [USD]' in df_clean else 0
        df_clean['usd_amount'] = df_clean['shipped_usd'] + df_clean['not_shipped_usd']
        
        # Parse weight
        df_clean['calc_weight_kg'] = pd.to_numeric(df_clean['Calculated Weight (Kg)'], errors='coerce').fillna(0) if 'Calculated Weight (Kg)' in df_clean else 0
        df_clean['unit_weight_kg'] = pd.to_numeric(df_clean['Unit Weight (Kg)'], errors='coerce').fillna(0) if 'Unit Weight (Kg)' in df_clean else 0
        
        # Compute final weight
        df_clean['weight_kg'] = df_clean['calc_weight_kg'].where(
            df_clean['calc_weight_kg'] > 0,
            df_clean['unit_weight_kg'] * df_clean['qty']
        )
        df_clean['weight_tons'] = df_clean['weight_kg'] / 1000
        
        # Dimensions
        df_clean['country'] = df_clean['Country'].astype(str).str.strip() if 'Country' in df_clean else 'Unknown'
        df_clean['system'] = df_clean['Order System'].astype(str).str.strip() if 'Order System' in df_clean else 'Unknown'
        df_clean['factory'] = df_clean['Factory Name'].astype(str).str.strip() if 'Factory Name' in df_clean else 'Unknown'
        df_clean['cell'] = df_clean['Cell Columns'].astype(str).str.strip() if 'Cell Columns' in df_clean else 'Unknown'
        df_clean['item_class'] = df_clean['Item Class'].astype(str).str.strip() if 'Item Class' in df_clean else 'Unknown'
        df_clean['order_number'] = df_clean['Order Number'].astype(str).str.strip() if 'Order Number' in df_clean else 'Unknown'
        
        # Classify items
        df_clean['is_standard'] = df_clean['sku'].isin(standard_items)
        df_clean['is_prime'] = df_clean['sku'].isin(prime_items)
        df_clean['item_category'] = 'Non-Standard'
        df_clean.loc[df_clean['is_standard'], 'item_category'] = 'Standard'
        df_clean.loc[df_clean['is_prime'], 'item_category'] = 'Prime'
        
        # ✅ ADD TIME DIMENSIONS - THIS WAS MISSING!
        df_clean['year'] = df_clean['booked_date'].dt.year
        df_clean['quarter'] = df_clean['booked_date'].dt.quarter
        df_clean['month'] = df_clean['booked_date'].dt.month
        df_clean['week'] = df_clean['booked_date'].dt.isocalendar().week
        df_clean['period_month'] = df_clean['booked_date'].dt.to_period('M')
        df_clean['period_quarter'] = df_clean['booked_date'].dt.to_period('Q')
        df_clean['period_year'] = df_clean['booked_date'].dt.to_period('Y')
        
        logger.info(f"Cleaned data: {len(df_clean)} rows")
        return df_clean

    
    def _aggregate_split_quantities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Properly aggregate split quantities
        Step 1: Aggregate by order+item+date (handles splits)
        """
        # ✅ FIX: Include period columns in group_cols
        group_cols = [
            'order_number', 'sku', 'booked_date', 
            'country', 'system', 'factory', 'cell', 
            'item_class', 'item_category',
            'period_month', 'period_quarter', 'period_year',  # ✅ ADD THESE!
            'year', 'quarter', 'month', 'week'  # ✅ ADD THESE TOO!
        ]
        
        order_level = df.groupby(group_cols, dropna=False).agg({
            'qty': 'sum',
            'egp_amount': 'sum',
            'usd_amount': 'sum',
            'weight_tons': 'sum',
            'shipped_qty': 'sum',
            'cancelled_qty': 'sum'
        }).reset_index()
        
        logger.info(f"Aggregated splits: {len(order_level)} unique order-item combinations")
        return order_level

    
    def _compute_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute additional metrics"""
        df_metrics = df.copy()
        
        # Fulfillment rate
        df_metrics['total_ordered'] = df_metrics['qty'] + df_metrics['shipped_qty'] + df_metrics['cancelled_qty']
        df_metrics['fulfillment_rate'] = (df_metrics['shipped_qty'] / df_metrics['total_ordered'] * 100).fillna(0)
        
        # Unit prices
        df_metrics['unit_price_egp'] = (df_metrics['egp_amount'] / df_metrics['qty']).replace([np.inf, -np.inf], 0).fillna(0)
        df_metrics['unit_price_usd'] = (df_metrics['usd_amount'] / df_metrics['qty']).replace([np.inf, -np.inf], 0).fillna(0)
        
        return df_metrics
    
    def _create_dimensional_views(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create aggregated views for each dimension"""
        views = {}
        
        # Monthly by SKU
        views['monthly_sku'] = df.groupby(['sku', 'period_month', 'item_category']).agg({
            'qty': 'sum',
            'egp_amount': 'sum',
            'usd_amount': 'sum',
            'weight_tons': 'sum'
        }).reset_index()
        views['monthly_sku']['period_start'] = views['monthly_sku']['period_month'].dt.to_timestamp()
        
        # By Country
        views['country'] = df.groupby(['country', 'period_month']).agg({
            'qty': 'sum',
            'egp_amount': 'sum',
            'usd_amount': 'sum',
            'weight_tons': 'sum',
            'sku': 'nunique'
        }).reset_index()
        views['country'].rename(columns={'sku': 'unique_skus'}, inplace=True)
        
        # By System
        views['system'] = df.groupby(['system', 'period_month']).agg({
            'qty': 'sum',
            'egp_amount': 'sum',
            'usd_amount': 'sum',
            'weight_tons': 'sum',
            'sku': 'nunique'
        }).reset_index()
        views['system'].rename(columns={'sku': 'unique_skus'}, inplace=True)
        
        # By Factory
        views['factory'] = df.groupby(['factory', 'period_month']).agg({
            'qty': 'sum',
            'egp_amount': 'sum',
            'usd_amount': 'sum',
            'weight_tons': 'sum',
            'sku': 'nunique'
        }).reset_index()
        views['factory'].rename(columns={'sku': 'unique_skus'}, inplace=True)
        
        # By Cell
        views['cell'] = df.groupby(['cell', 'period_month']).agg({
            'qty': 'sum',
            'egp_amount': 'sum',
            'usd_amount': 'sum',
            'weight_tons': 'sum',
            'sku': 'nunique'
        }).reset_index()
        views['cell'].rename(columns={'sku': 'unique_skus'}, inplace=True)
        
        logger.info(f"Created {len(views)} dimensional views")
        
        return views
    
    def _compute_summary(self, df: pd.DataFrame, views: Dict) -> Dict:
        """Compute summary statistics"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'overview': {},
            'items': {},
            'dimensions': {}
        }
        
        # Overview
        summary['overview'] = {
            'total_rows': int(len(df)),
            'total_qty': float(df['qty'].sum()),
            'total_egp': float(df['egp_amount'].sum()),
            'total_usd': float(df['usd_amount'].sum()),
            'total_tons': float(df['weight_tons'].sum()),
            'date_range': {
                'min': df['booked_date'].min().isoformat(),
                'max': df['booked_date'].max().isoformat(),
                'span_days': int((df['booked_date'].max() - df['booked_date'].min()).days)
            }
        }
        
        # Items
        item_stats = df.groupby('sku').agg({
            'qty': ['sum', 'mean', 'std'],
            'usd_amount': 'sum',
            'booked_date': ['min', 'max', 'count']
        })
        
        summary['items'] = {
            'total_unique': int(df['sku'].nunique()),
            'standard_items': int(df[df['item_category'] == 'Standard']['sku'].nunique()),
            'prime_items': int(df[df['item_category'] == 'Prime']['sku'].nunique()),
            'non_standard_items': int(df[df['item_category'] == 'Non-Standard']['sku'].nunique())
        }
        
        # Dimensions
        summary['dimensions'] = {
            'countries': int(df['country'].nunique()),
            'systems': int(df['system'].nunique()),
            'factories': int(df['factory'].nunique()),
            'cells': int(df['cell'].nunique())
        }
        
        # Top performers
        summary['top_items_by_qty'] = df.groupby('sku')['qty'].sum().nlargest(10).to_dict()
        summary['top_items_by_usd'] = df.groupby('sku')['usd_amount'].sum().nlargest(10).to_dict()
        summary['top_countries_by_usd'] = df.groupby('country')['usd_amount'].sum().nlargest(10).to_dict()
        summary['top_systems_by_qty'] = df.groupby('system')['qty'].sum().nlargest(10).to_dict()
        
        return summary
    
    def _save_outputs(self, df: pd.DataFrame, views: Dict, summary: Dict, validation: Dict) -> Dict:
        """Save all outputs"""
        output_dir = Path(self.config.get('output_dir', 'outputs'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        paths = {}
        
        # Save main data
        data_path = output_dir / f'processed_data_{timestamp}.parquet'
        df.to_parquet(data_path, index=False)
        paths['processed_data'] = str(data_path)
        
        # Save dimensional views
        for view_name, view_df in views.items():
            view_path = output_dir / f'view_{view_name}_{timestamp}.parquet'
            view_df.to_parquet(view_path, index=False)
            paths[f'view_{view_name}'] = str(view_path)
        
        # Save summary
        summary_path = output_dir / f'summary_{timestamp}.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        paths['summary'] = str(summary_path)
        
        # Save validation
        validation_path = output_dir / f'validation_{timestamp}.json'
        with open(validation_path, 'w') as f:
            json.dump(validation, f, indent=2, default=str)
        paths['validation'] = str(validation_path)
        
        logger.info(f"Saved all outputs to {output_dir}")
        
        return paths