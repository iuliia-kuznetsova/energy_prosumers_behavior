'''
    Data Preprocessing

    This module provides the DataPreparation class for preprocessing raw data
    into a format suitable for merging and model training.

    Strategy:
    - Define schema for each DataFrame (power, client, electricity, gas, weather)
    - Validate required columns exist
    - Convert data types according to schema
    - Handle train/test mode differences

    Input:
    - Dictionary of raw DataFrames from Kaggle competition data

    Output:
    - Dictionary of preprocessed DataFrames with proper schemas

    Usage:
    python -m src.preds.data_preprocessing
'''

# ---------- Imports ---------- #
import os
import gc
import copy
import polars as pl
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from sklearn.base import BaseEstimator, TransformerMixin

from src.logging_setup import setup_logging


# ---------- Logging setup ---------- #
logger = setup_logging('data_preprocessing')


# ---------- Config ---------- #
load_dotenv()
PROJECT_ROOT = Path(__file__).parent.parent.parent
os.chdir(PROJECT_ROOT)


# ---------- Constants ---------- #
RAW_DATA_DIR = os.getenv('RAW_DATA_DIR', './data/raw')
PROCESSED_DATA_DIR = os.getenv('PROCESSED_DATA_DIR', './data/processed')


# ---------- DataPreparation Class ---------- #
class DataPreparation(BaseEstimator, TransformerMixin):
    '''
        Prepare raw data by applying schema validation and type conversion.
        
        Handles three modes:
        - 'train': Initial training data processing
        - 'retrain': Processing revealed targets during inference
        - 'test': Processing test data for predictions
    '''

    def __init__(self, mode: str = 'train'):
        '''
            Initialize the DataPreparation transformer.
            
            Args:
                mode: One of 'train', 'retrain', or 'test'
        '''
        if mode not in ['train', 'retrain', 'test']:
            raise ValueError(f"Mode must be 'train', 'retrain', or 'test', got: {mode}")
        
        self.mode = mode
        self._define_schemas()
        logger.info(f'DataPreparation initialized with mode: {mode}')
        
    def _define_schemas(self) -> None:
        '''Define schema for each DataFrame'''

        self.power_schema = {
            'row_id': pl.Int32,
            'datetime': pl.Datetime,
            'prediction_unit_id': pl.Int16,
            'is_consumption': pl.Boolean,
            'county': pl.Int8,
            'is_business': pl.Boolean,
            'product_type': pl.Int8,
            'target': pl.Float64
        }

        self.client_schema = {
            'date': pl.Date,
            'county': pl.Int8,
            'is_business': pl.Boolean,
            'product_type': pl.Int8,
            'installed_capacity': pl.Float64
        }

        self.electricity_schema = {
            'forecast_date': pl.Datetime,
            'euros_per_mwh': pl.Float64
        }

        self.gas_schema = {
            'forecast_date': pl.Datetime,
            'lowest_price_per_mwh': pl.Float64,
            'highest_price_per_mwh': pl.Float64
        }

        self.forecast_weather_schema = {
            'latitude': pl.Float32,
            'longitude': pl.Float32,
            'forecast_datetime': pl.Datetime,
            'hours_ahead': pl.Int8,
            'temperature': pl.Float32,
            'cloudcover_total': pl.Float32,
            'total_precipitation': pl.Float32,
            '10_metre_u_wind_component': pl.Float32,
            '10_metre_v_wind_component': pl.Float32,
            'direct_solar_radiation': pl.Float32,
            'surface_solar_radiation_downwards': pl.Float32
        }

        self.historical_weather_schema = {
            'datetime': pl.Datetime,
            'latitude': pl.Float32,
            'longitude': pl.Float32,
            'temperature': pl.Float32,
            'surface_pressure': pl.Float32,
            'rain': pl.Float32,
            'snowfall': pl.Float32,
            'cloudcover_total': pl.Float32,
            'windspeed_10m': pl.Float32,
            'winddirection_10m': pl.Float32,
            'direct_solar_radiation': pl.Float32,
            'diffuse_radiation': pl.Float32
        }

        self.mapping_schema = {
            'county': pl.Int8,
            'latitude': pl.Float32,
            'longitude': pl.Float32,
        }

    def _check_columns(self, df: pl.DataFrame, required_columns: List[str]) -> pl.DataFrame:
        '''Check if all required columns exist in the DataFrame'''
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")
        return df

    def _convert_to_schema(self, data: Dict[str, pl.DataFrame]) -> Dict[str, pl.DataFrame]:
        '''Apply schema conversion to all DataFrames'''

        self.power_df = data['power_df']
        self.client_df = data['client_df']
        self.electricity_df = data['electricity_df']
        self.gas_df = data['gas_df']
        self.historical_weather_df = data['historical_weather_df']
        self.forecast_weather_df = data['forecast_weather_df']
        self.mapping_df = data['mapping_df']

        # Power DataFrame
        if self.mode == 'test':
            self.power_df = (
                self.power_df
                    .with_columns([
                        pl.col('prediction_datetime').alias('datetime'),
                        pl.lit(None, dtype=pl.Float64).alias('target')
                    ])
            )
        
        self.power_df = self._check_columns(
            self.power_df, 
            ['row_id', 'datetime', 'prediction_unit_id', 'is_consumption', 
             'county', 'is_business', 'product_type', 'target']
        )
        
        self.power_df = (
            self.power_df
                .with_columns([
                    (pl.col('datetime').str.strptime(pl.Datetime(time_unit="us"), '%Y-%m-%d %H:%M:%S') 
                     if self.mode == 'train' else pl.col('datetime'))
                ])
                .select([col for col in self.power_schema.keys()])
                .with_columns([
                    *(pl.col(col).cast(dtype) for col, dtype in self.power_schema.items() if col != 'datetime')
                ])
        )

        # Client DataFrame
        self.client_df = self._check_columns(
            self.client_df, 
            ['date', 'county', 'is_business', 'product_type', 'installed_capacity']
        )
        
        self.client_df = (
            self.client_df
                .with_columns([
                    (pl.col('date').str.strptime(pl.Date, '%Y-%m-%d') 
                     if self.mode == 'train' else pl.col('date'))
                ])
                .select([col for col in self.client_schema.keys()])
                .with_columns([
                    *(pl.col(col).cast(dtype) for col, dtype in self.client_schema.items() if col != 'date')
                ])
        )

        # Electricity DataFrame
        self.electricity_df = self._check_columns(
            self.electricity_df, 
            ['forecast_date', 'euros_per_mwh']
        )
        
        self.electricity_df = (
            self.electricity_df
                .with_columns([
                    (pl.col('forecast_date').str.strptime(pl.Datetime(time_unit="us"), '%Y-%m-%d %H:%M:%S') 
                     if self.mode == 'train' else pl.col('forecast_date'))
                ])
                .select(self.electricity_schema.keys())
                .with_columns([
                    *(pl.col(col).cast(dtype) for col, dtype in self.electricity_schema.items() if col != 'forecast_date')    
                ])
        )
    
        # Gas DataFrame
        self.gas_df = self._check_columns(
            self.gas_df, 
            ['forecast_date', 'lowest_price_per_mwh', 'highest_price_per_mwh']
        )
        
        self.gas_df = (
            self.gas_df
                .with_columns([
                    (pl.col('forecast_date').str.strptime(pl.Date, '%Y-%m-%d').cast(pl.Datetime) 
                     if self.mode == 'train' else pl.col('forecast_date'))
                ])
                .select(self.gas_schema.keys())
                .with_columns([
                    *(pl.col(col).cast(dtype) for col, dtype in self.gas_schema.items() if col != 'forecast_date')
                ])
        )

        # Forecast Weather DataFrame
        self.forecast_weather_df = self._check_columns(
            self.forecast_weather_df, 
            ['latitude', 'longitude', 'forecast_datetime', 'hours_ahead', 
             'temperature', 'cloudcover_total', 'total_precipitation', 
             '10_metre_u_wind_component', '10_metre_v_wind_component', 
             'direct_solar_radiation', 'surface_solar_radiation_downwards']
        )
        
        self.forecast_weather_df = (
            self.forecast_weather_df
                .with_columns([
                    (pl.col('forecast_datetime').str.strptime(pl.Datetime(time_unit="us"), '%Y-%m-%d %H:%M:%S') 
                     if self.mode == 'train' else pl.col('forecast_datetime'))
                ])
                .select(self.forecast_weather_schema.keys())
                .with_columns([
                    *(pl.col(col).cast(dtype) for col, dtype in self.forecast_weather_schema.items() if col != 'forecast_datetime') 
                ])
        )

        # Historical Weather DataFrame
        self.historical_weather_df = self._check_columns(
            self.historical_weather_df, 
            ['datetime', 'latitude', 'longitude', 
             'temperature', 'surface_pressure', 'rain', 'snowfall', 
             'cloudcover_total', 'windspeed_10m', 'winddirection_10m', 
             'direct_solar_radiation', 'diffuse_radiation']
        )
        
        self.historical_weather_df = (
            self.historical_weather_df
                .with_columns([
                    (pl.col('datetime').str.strptime(pl.Datetime(time_unit="us"), '%Y-%m-%d %H:%M:%S') 
                     if self.mode == 'train' else pl.col('datetime'))
                ])
                .select(self.historical_weather_schema.keys())
                .with_columns([
                    *(pl.col(col).cast(dtype) for col, dtype in self.historical_weather_schema.items() if col != 'datetime') 
                ])
        )

        # Mapping DataFrame
        self.mapping_df = self._check_columns(
            self.mapping_df, 
            ['county', 'latitude', 'longitude']
        )
        
        self.mapping_df = (
            self.mapping_df
                .select([col for col in self.mapping_schema.keys()])
                .with_columns([
                    pl.col(col).cast(dtype) for col, dtype in self.mapping_schema.items()
                ])
        )

        # Save intermediate data for train mode (needed for retrain/test)
        if self.mode == 'train':
            processed_dir = Path(PROCESSED_DATA_DIR)
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            self.client_df.write_parquet(processed_dir / 'client_train.parquet')
            self.electricity_df.write_parquet(processed_dir / 'electricity_train.parquet') 
            self.gas_df.write_parquet(processed_dir / 'gas_train.parquet')
            self.forecast_weather_df.write_parquet(processed_dir / 'forecast_weather_train.parquet') 
            self.historical_weather_df.write_parquet(processed_dir / 'historical_weather_train.parquet')
            logger.info(f'Intermediate data saved to {processed_dir}')
        else:
            # Load and concatenate with training data for retrain/test modes
            processed_dir = Path(PROCESSED_DATA_DIR)
            
            client_train = pl.read_parquet(processed_dir / 'client_train.parquet')
            electricity_train = pl.read_parquet(processed_dir / 'electricity_train.parquet')
            gas_train = pl.read_parquet(processed_dir / 'gas_train.parquet')
            forecast_weather_train = pl.read_parquet(processed_dir / 'forecast_weather_train.parquet')
            historical_weather_train = pl.read_parquet(processed_dir / 'historical_weather_train.parquet')
            
            self.client_df = (
                pl.concat([client_train, self.client_df], how='vertical')
                  .unique(subset=['date', 'county', 'is_business', 'product_type'], keep='last')
            )
            self.electricity_df = (
                pl.concat([electricity_train, self.electricity_df], how='vertical')
                  .unique(subset=['forecast_date'], keep='last')
            )
            self.gas_df = (
                pl.concat([gas_train, self.gas_df], how='vertical')
                  .unique(subset=['forecast_date'], keep='last')
            )
            self.forecast_weather_df = (
                pl.concat([forecast_weather_train, self.forecast_weather_df], how='vertical')
                  .unique(subset=['latitude', 'longitude', 'forecast_datetime', 'hours_ahead'], keep='last')
            )
            self.historical_weather_df = (
                pl.concat([historical_weather_train, self.historical_weather_df], how='vertical')
                  .unique(subset=['datetime', 'latitude', 'longitude'], keep='last')
            )
            logger.info('Concatenated with training data for retrain/test mode')

        processed_data = {
            'power_df': self.power_df,
            'client_df': self.client_df,
            'electricity_df': self.electricity_df,
            'gas_df': self.gas_df,
            'forecast_weather_df': self.forecast_weather_df,
            'historical_weather_df': self.historical_weather_df,
            'mapping_df': self.mapping_df
        }

        logger.info('DONE: Schema conversion completed')
        return processed_data

    def fit(self, X: Dict[str, pl.DataFrame], y=None) -> 'DataPreparation':
        '''Fit method (required for sklearn pipeline compatibility)'''
        return self

    def transform(self, X: Dict[str, pl.DataFrame]) -> Dict[str, pl.DataFrame]:
        '''Transform data by applying schema validation and type conversion'''
        processed_data = self._convert_to_schema(X)
        return processed_data

    def fit_transform(self, X: Dict[str, pl.DataFrame], y=None) -> Dict[str, pl.DataFrame]:
        '''Fit and transform in one step'''
        self.fit(X, y)
        return self.transform(X)


# ---------- Helper Functions ---------- #
def load_raw_data(data_dir: str = RAW_DATA_DIR) -> Dict[str, pl.DataFrame]:
    '''
        Load raw CSV files from data directory.
        
        Returns dictionary with all DataFrames needed for processing.
    '''
    data_path = Path(data_dir)
    
    power_df = pl.read_csv(data_path / 'train.csv')
    client_df = pl.read_csv(data_path / 'client.csv')
    electricity_df = pl.read_csv(data_path / 'electricity_prices.csv')
    gas_df = pl.read_csv(data_path / 'gas_prices.csv')
    historical_weather_df = pl.read_csv(data_path / 'historical_weather.csv')
    forecast_weather_df = pl.read_csv(data_path / 'forecast_weather.csv')
    mapping_df = pl.read_csv(data_path / 'weather_station_to_county_mapping.csv')
    
    logger.info(f'Raw data loaded from {data_path}')
    logger.info(f'Power data: {power_df.shape}')
    logger.info(f'Client data: {client_df.shape}')
    
    return {
        'power_df': power_df,
        'client_df': client_df,
        'electricity_df': electricity_df,
        'gas_df': gas_df,
        'forecast_weather_df': forecast_weather_df,
        'historical_weather_df': historical_weather_df,
        'mapping_df': mapping_df
    }


def run_preprocessing(
    data_dir: str = RAW_DATA_DIR,
    mode: str = 'train'
) -> Dict[str, pl.DataFrame]:
    '''
        Run data preprocessing pipeline.
        
        Args:
            data_dir: Directory with raw CSV files
            mode: Processing mode ('train', 'retrain', 'test')
            
        Returns:
            Dictionary of preprocessed DataFrames
    '''
    logger.info('Starting data preprocessing pipeline')
    
    # Load raw data
    raw_data = load_raw_data(data_dir)
    
    # Preprocess
    preprocessor = DataPreparation(mode=mode)
    processed_data = preprocessor.fit_transform(raw_data)
    
    logger.info('Data preprocessing pipeline completed successfully')
    
    gc.collect()
    return processed_data


# ---------- Main function ---------- #
if __name__ == '__main__':
    run_preprocessing()


# ---------- All exports ---------- #
__all__ = ['DataPreparation', 'load_raw_data', 'run_preprocessing']
