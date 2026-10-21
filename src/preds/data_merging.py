'''
    Data Merging

    This module provides the DataMerging class for merging all preprocessed
    DataFrames into a single unified DataFrame for feature engineering.

    Strategy:
    - Join client capacity shifted forward by 2 days
    - Join electricity and gas prices shifted forward by 1 day
    - Join the day-ahead forecast from the same data_block_id
    - Join observed weather shifted forward by 48 hours
    - Create capacity_factor from that known capacity

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
        - Power + Client shifted +2 days (capacity known for the target day)
        - + Electricity prices shifted +1 day
        - + Gas prices shifted +1 day, joined on the date
        - + Forecast weather from the same data_block_id whose valid time is the target hour
        - + Historical weather shifted +48 hours
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

        # Join with client data:
        # Client rows are dated 2 days before the target day of the same batch.
        # Moving the date forward attaches the capacity that is already known according to the competition rules.
        self.client_df = (
            self.client_df
                .with_columns(
                    (pl.col('date') + pl.duration(days=2)).alias('date')
                )
                # A repeated segment-day would multiply every target hour.
                .unique(
                    subset=['product_type', 'county', 'is_business', 'date'],
                    keep='last'
                )
                .rename({'data_block_id': 'client_block_id'})
        )
        self.df = (
            self.df
                .join(
                    self.client_df,
                    how='left',
                    on=['product_type', 'county', 'is_business', 'date']
                )
        )
        logger.info(f'After client join: {self.df.shape}')

        # The price file's timestamp is the previous day. Shift it onto the target hour.
        self.electricity_df = (
            self.electricity_df
                .with_columns([
                    (pl.col('forecast_date') + pl.duration(days=1)).alias('datetime'),
                    pl.col('euros_per_mwh').abs().alias('electricity_price')
                ])
                # One price per hour; a duplicate hour would multiply the target rows.
                .unique(subset=['datetime'], keep='last')
                .rename({'data_block_id': 'electricity_block_id'})
        )
        
        self.df = (
            self.df
                .join(
                    self.electricity_df.select(
                        ['electricity_price', 'datetime', 'electricity_block_id']
                    ),
                    how='left',
                    on=['datetime']
                )
        )
        logger.info(f'After electricity join: {self.df.shape}')

        # Gas is one price per day, revealed for the previous day according to the comprtition rules.
        self.gas_df = (
            self.gas_df
                .with_columns([
                    (pl.col('forecast_date') + pl.duration(days=1)).cast(pl.Date).alias('date'),
                    ((pl.col('lowest_price_per_mwh') + pl.col('highest_price_per_mwh')) / 2)
                        .alias('gas_price')
                ])
                .unique(subset=['date'], keep='last')
                .rename({'data_block_id': 'gas_block_id'})
                .select(['date', 'gas_price', 'gas_block_id'])
        )

        self.df = (
            self.df
                .join(
                    self.gas_df,
                    how='left',
                    on=['date']
                )
        )
        logger.info(f'After gas join: {self.df.shape}')

        # Process forecast weather
        forecast_weather_cols = [
            'temperature', 'cloudcover_total', 'total_precipitation', 
            '10_metre_u_wind_component', '10_metre_v_wind_component', 
            'direct_solar_radiation', 'surface_solar_radiation_downwards'
        ]

        # The day each batch is asked to predict. One batch is one calendar day.
        block_day = (
            self.power_df
                .select(
                    'data_block_id',
                    pl.col('datetime').dt.date().alias('block_date')
                )
                .unique(subset=['data_block_id'])
        )
        
        self.forecast_weather_df = (
            self.forecast_weather_df
                .join(block_day, on='data_block_id', how='inner')
                # The forecast is for the target day according to the competition rules.
                .filter(pl.col('forecast_datetime').dt.date() == pl.col('block_date'))
                .with_columns(
                    pl.col('forecast_datetime').alias('datetime')
                )
                .join(
                    self.mapping_df,
                    how='left',
                    on=['longitude', 'latitude']
                )
        )
        
        # Average stations inside one county, one target hour, one forecast issue.
        self.forecast_weather_df = (
            self.forecast_weather_df
                .group_by(['county', 'datetime', 'data_block_id'])
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
                # Copy of the issue batch, kept after the join so it can be audited.
                .with_columns(pl.col('data_block_id').alias('fw_block_id'))
        ) 
        
        # Join with forecast weather:
        # Same batch and same valid time, the day-ahead forecast according to the competition rules.
        self.df = (
            self.df
                .join(
                    self.forecast_weather_df, 
                    how='left',
                    on=['county', 'datetime', 'data_block_id']
                )
                .with_columns(
                    [pl.col(f'fw_{col}').fill_null(pl.col(f'fw_{col}').median().over('datetime')) 
                     for col in forecast_weather_cols] + 
                    [pl.col('fw_diffuse_solar_radiation').fill_null(
                        pl.col('fw_diffuse_solar_radiation').median().over('datetime'))]
                    # A county gap is filled from other counties at the same hour, which is the same batch.
                    + [pl.col('fw_block_id').fill_null(pl.col('fw_block_id').max().over('datetime'))]
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
                # 48h is old enough to have been revealed for every hour of the target day.
                .with_columns(
                    (pl.col('datetime') + pl.duration(hours=48)).alias('datetime')
                )
                .join(
                    self.mapping_df,
                    how='left',
                    on=['longitude', 'latitude']
                )
        )
        
        # Average stations. The batch id is the batch that revealed the observation, not the target.
        self.historical_weather_df = (
            self.historical_weather_df
                .group_by(['county', 'datetime'])
                .agg(
                    [pl.mean(col).alias(f'hw_{col}') for col in historical_weather_cols]
                    + [pl.col('data_block_id').max().alias('hw_block_id')]
                )  
                .with_columns([
                    (pl.col('hw_rain') + pl.col('hw_snowfall')).alias('hw_total_precipitation')
                ])
        )
        
        # Join with historical weather:
        # Join on the shifted timestamp only. The observation's own batch is an earlier one.
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
                    # Same-hour fill stays inside observations that were already shifted by 48h.
                    + [pl.col('hw_block_id').fill_null(pl.col('hw_block_id').max().over('datetime'))]
                ) 
        )
        logger.info(f'After historical weather join: {self.df.shape}')

        # Generate capacity_factor and define column order:
        # capacity_factor uses the capacity sitting on the row, which is the 2-day-old one according to the competition rules.
        # Zero or missing capacity would make the ratio infinite, so those rows stay null and are dropped later.
        self.df = (
            self.df
                .with_columns([
                    pl.when(pl.col('installed_capacity') > 0)
                        .then(pl.col('power') / pl.col('installed_capacity'))
                        .otherwise(None)
                        .alias('capacity_factor')
                ])
                .select(
                    'row_id', 'datetime', 'data_block_id', 'prediction_unit_id', 'is_consumption', 
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
                    'hw_direct_solar_radiation', 'hw_diffuse_radiation',
                    # Source batches. Feature engineering drops these before the model sees the frame.
                    'client_block_id', 'electricity_block_id', 'gas_block_id',
                    'fw_block_id', 'hw_block_id'
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
