'''
    Data Merging

    This module provides the DataMerging class for merging all preprocessed
    DataFrames into a single unified DataFrame for feature engineering.

    Strategy:
    - Join power data with client data on business keys
    - Join with electricity and gas prices
    - Join with weather data (forecast and historical) via county mapping
    - Create derived features (capacity_factor)

    Input:
    - Dictionary of preprocessed DataFrames from DataPreparation

    Output:
    - Single merged Polars DataFrame ready for feature engineering

    Usage:
    python -m src.preds.data_merging
'''

# ---------- Imports ---------- #
import os
import gc
import copy
import polars as pl
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from sklearn.base import BaseEstimator, TransformerMixin

from src.logging_setup import setup_logging


# ---------- Logging setup ---------- #
logger = setup_logging('data_merging')


# ---------- Config ---------- #
load_dotenv()
PROJECT_ROOT = Path(__file__).parent.parent.parent
os.chdir(PROJECT_ROOT)


# ---------- DataMerging Class ---------- #
class DataMerging(BaseEstimator, TransformerMixin):
    '''
        Merge all preprocessed DataFrames into a single unified DataFrame.
        
        Performs joins on:
        - Power + Client (on product_type, county, is_business, date)
        - + Electricity prices (on datetime)
        - + Gas prices (on datetime)
        - + Forecast weather (on county, datetime)
        - + Historical weather (on county, datetime)
    '''
        
    def __init__(self, mode: str = 'train'):
        '''
            Initialize the DataMerging transformer.
            
            Args:
                mode: One of 'train', 'retrain', or 'test'
        '''
        if mode not in ['train', 'retrain', 'test']:
            raise ValueError(f"Mode must be 'train', 'retrain', or 'test', got: {mode}")
        
        self.mode = mode
        logger.info(f'DataMerging initialized with mode: {mode}')
    
    def _merge_data(self, processed_data: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        '''
            Merge all DataFrames into a single unified DataFrame.
            
            Args:
                processed_data: Dictionary containing all preprocessed DataFrames
                
            Returns:
                Merged Polars DataFrame
        '''
        self.power_df = processed_data['power_df']
        self.client_df = processed_data['client_df']
        self.electricity_df = processed_data['electricity_df']
        self.gas_df = processed_data['gas_df']
        self.historical_weather_df = processed_data['historical_weather_df']
        self.forecast_weather_df = processed_data['forecast_weather_df']
        self.mapping_df = processed_data['mapping_df']

        # Process power data: set as main DataFrame, rename target to power
        self.df = (
            self.power_df
                .rename({'target': 'power'})
                .with_columns([
                    (pl.col('datetime').dt.truncate('1d').cast(pl.Date)).alias('date')
                ])
        )
        logger.info(f'Power data processed: {self.df.shape}')

        # Join with client data
        self.df = (
            self.df
                .join(
                    self.client_df,
                    how='left',
                    on=['product_type', 'county', 'is_business', 'date']
                )
        )
        logger.info(f'After client join: {self.df.shape}')

        # Prepare electricity prices
        self.electricity_df = (
            self.electricity_df
                .with_columns([
                    pl.col('forecast_date').alias('datetime'),
                    pl.col('euros_per_mwh').abs().alias('electricity_price')
                ])
        )
        
        # Join with electricity prices
        self.df = (
            self.df
                .join(
                    self.electricity_df.select(['electricity_price', 'datetime']),
                    how='left',
                    on=['datetime']
                )
        )
        logger.info(f'After electricity join: {self.df.shape}')

        # Prepare gas prices (expand to hourly)
        self.gas_df = (
            self.gas_df
                .with_columns([
                    pl.col('forecast_date').alias('date_only'),
                    ((pl.col('lowest_price_per_mwh') + pl.col('highest_price_per_mwh')) / 2).alias('gas_price')
                ])
        )
        
        # Create hourly gas prices
        hours_df = pl.DataFrame({'hour': range(24)})
        self.gas_df = (
            self.gas_df
                .join(hours_df, how='cross')
                .with_columns([
                    (pl.col('date_only') + pl.col('hour') * pl.duration(hours=1)).alias('datetime'),
                ])
                .select(['datetime', 'gas_price'])  
        )

        # Join with gas prices
        self.df = (
            self.df
                .join(
                    self.gas_df,
                    how='left',
                    on=['datetime']
                )
        )
        logger.info(f'After gas join: {self.df.shape}')

        # Process forecast weather
        forecast_weather_cols = [
            'temperature', 'cloudcover_total', 'total_precipitation', 
            '10_metre_u_wind_component', '10_metre_v_wind_component', 
            'direct_solar_radiation', 'surface_solar_radiation_downwards'
        ]
        
        self.forecast_weather_df = (
            self.forecast_weather_df
                .filter(pl.col('hours_ahead') <= 24)
                .with_columns(
                    pl.col('forecast_datetime').alias('datetime')
                )
                .join(
                    self.mapping_df,
                    how='left',
                    on=['longitude', 'latitude']
                )
        )
        
        # Aggregate forecast weather by county and datetime
        self.forecast_weather_df = (
            self.forecast_weather_df
                .group_by(['county', 'datetime'])
                .agg([pl.mean(col).alias(f'fw_{col}') for col in forecast_weather_cols])  
                .with_columns([
                    # Cap cloud cover at 100
                    pl.when(pl.col('fw_cloudcover_total') * 100 > 100)
                        .then(100)
                        .otherwise(pl.col('fw_cloudcover_total') * 100),
                    pl.col('fw_direct_solar_radiation').abs(),
                    # Compute diffuse solar radiation
                    (pl.when((pl.col('fw_surface_solar_radiation_downwards').abs() - pl.col('fw_direct_solar_radiation').abs()) < 0)
                        .then(0)
                        .otherwise(pl.col('fw_surface_solar_radiation_downwards').abs() - pl.col('fw_direct_solar_radiation').abs()))
                        .alias('fw_diffuse_solar_radiation'),
                    (pl.col('fw_total_precipitation') * 1000).abs()
                ])
        ) 
        
        # Join with forecast weather
        self.df = (
            self.df
                .join(
                    self.forecast_weather_df, 
                    how='left',
                    on=['county', 'datetime']
                )
                .with_columns(
                    [pl.col(f'fw_{col}').fill_null(pl.col(f'fw_{col}').median().over('datetime')) 
                     for col in forecast_weather_cols] + 
                    [pl.col('fw_diffuse_solar_radiation').fill_null(
                        pl.col('fw_diffuse_solar_radiation').median().over('datetime'))]
                )  
        )
        logger.info(f'After forecast weather join: {self.df.shape}')

        # Process historical weather
        historical_weather_cols = [
            'temperature', 'surface_pressure', 'rain', 'snowfall', 
            'cloudcover_total', 'windspeed_10m', 'winddirection_10m', 
            'direct_solar_radiation', 'diffuse_radiation'
        ]
        
        self.historical_weather_df = (
            self.historical_weather_df
                .join(
                    self.mapping_df,
                    how='left',
                    on=['longitude', 'latitude']
                )
        )
        
        # Aggregate historical weather by county and datetime
        self.historical_weather_df = (
            self.historical_weather_df
                .group_by(['county', 'datetime'])
                .agg([pl.mean(col).alias(f'hw_{col}') for col in historical_weather_cols])  
                .with_columns([
                    (pl.col('hw_rain') + pl.col('hw_snowfall')).alias('hw_total_precipitation')
                ])
        )
        
        # Join with historical weather
        self.df = (
            self.df
                .join(
                    self.historical_weather_df,
                    how='left',
                    on=['county', 'datetime']
                )
                .with_columns(
                    [pl.col(f'hw_{col}').fill_null(pl.col(f'hw_{col}').median().over('datetime')) 
                     for col in historical_weather_cols] +
                    [pl.col('hw_total_precipitation').fill_null(
                        pl.col('hw_total_precipitation').median().over('datetime'))]
                ) 
        )
        logger.info(f'After historical weather join: {self.df.shape}')
  
        # Generate capacity_factor and define column order
        self.df = (
            self.df
                .with_columns([
                    (pl.col('power') / pl.col('installed_capacity')).alias('capacity_factor')
                ])
                .select(
                    'row_id', 'datetime', 'prediction_unit_id', 'is_consumption', 
                    'county', 'is_business', 'product_type',
                    'capacity_factor', 'power', 'installed_capacity', 
                    'electricity_price', 'gas_price',
                    'fw_temperature', 'fw_cloudcover_total', 
                    'fw_total_precipitation', 
                    'fw_10_metre_u_wind_component', 'fw_10_metre_v_wind_component',
                    'fw_direct_solar_radiation', 'fw_diffuse_solar_radiation',
                    'hw_temperature', 'hw_surface_pressure',
                    'hw_cloudcover_total', 'hw_total_precipitation',
                    'hw_windspeed_10m', 'hw_winddirection_10m',
                    'hw_direct_solar_radiation', 'hw_diffuse_radiation'
                ) 
                .sort('datetime')
        )

        logger.info(f'DONE: Data merging completed. Final shape: {self.df.shape}')
        return self.df
    
    def fit(self, X: Dict[str, pl.DataFrame], y=None) -> 'DataMerging':
        '''Fit method (required for sklearn pipeline compatibility)'''
        return self

    def transform(self, X: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        '''Transform by merging all DataFrames'''
        df = self._merge_data(X)
        return df

    def fit_transform(self, X: Dict[str, pl.DataFrame], y=None) -> pl.DataFrame:
        '''Fit and transform in one step'''
        self.fit(X)
        return self.transform(X)


# ---------- Helper Functions ---------- #
def run_merging(
    processed_data: Dict[str, pl.DataFrame],
    mode: str = 'train'
) -> pl.DataFrame:
    '''
        Run data merging pipeline.
        
        Args:
            processed_data: Dictionary of preprocessed DataFrames
            mode: Processing mode ('train', 'retrain', 'test')
            
        Returns:
            Merged Polars DataFrame
    '''
    logger.info('Starting data merging pipeline')
    
    merger = DataMerging(mode=mode)
    merged_df = merger.fit_transform(processed_data)
    
    logger.info('Data merging pipeline completed successfully')
    
    gc.collect()
    return merged_df


# ---------- Main function ---------- #
if __name__ == '__main__':
    # For standalone testing, load preprocessed data first
    from src.preds.data_preprocessing import run_preprocessing
    processed_data = run_preprocessing()
    merged_df = run_merging(processed_data)
    print(f'Merged DataFrame shape: {merged_df.shape}')


# ---------- All exports ---------- #
__all__ = ['DataMerging', 'run_merging']
