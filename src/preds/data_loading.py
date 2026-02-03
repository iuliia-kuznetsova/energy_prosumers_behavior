'''
    Data Loading

    This module downloads data from Kaggle competition 
    'predict-energy-behavior-of-prosumers' and extracts it to data/raw directory.

    Requirements:
    - Kaggle API credentials (~/.kaggle/kaggle.json or environment variables)
    - kaggle package installed (pip install kaggle)

    Input:
    - Competition name: predict-energy-behavior-of-prosumers

    Output:
    - Extracted CSV files in data/raw/

    Usage:
    python -m src.preds.data_loading
'''

# ----------- Imports ----------- #
import os
import gc
import subprocess
from pathlib import Path
from dotenv import load_dotenv

from src.logging_setup import setup_logging

# ---------- Logging setup ---------- #
logger = setup_logging('data_loading')

# ---------- Config ---------- #
load_dotenv()
PROJECT_ROOT = Path(__file__).parent.parent.parent
os.chdir(PROJECT_ROOT)

# ---------- Constants ---------- #
# Competition name on Kaggle
COMPETITION_NAME = 'predict-energy-behavior-of-prosumers'
# Data directory for raw data
RAW_DATA_DIR = os.getenv('RAW_DATA_DIR', './data/raw')

# Expected data files from the competition
EXPECTED_FILES = [
    'train.csv',
    'client.csv', 
    'electricity_prices.csv',
    'gas_prices.csv',
    'historical_weather.csv',
    'forecast_weather.csv',
    'weather_station_to_county_mapping.csv'
]


# ---------- Functions ---------- #
def check_kaggle_credentials() -> bool:
    '''
        Check if Kaggle API credentials are available.
        Returns True if credentials are found, False otherwise.
    '''
    # Check for kaggle.json file
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    if kaggle_json.exists():
        logger.info('Kaggle credentials found at ~/.kaggle/kaggle.json')
        return True
    
    # Check for environment variables
    if os.getenv('KAGGLE_USERNAME') and os.getenv('KAGGLE_KEY'):
        logger.info('Kaggle credentials found in environment variables')
        return True
    
    logger.warning('Kaggle credentials not found')
    return False


def download_from_kaggle(
    competition: str = COMPETITION_NAME,
    output_dir: str = RAW_DATA_DIR
) -> None:
    '''
        Download competition data from Kaggle using the Kaggle API.
        
        Args:
            competition: Kaggle competition name
            output_dir: Directory to save downloaded files
    '''
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f'Downloading data from Kaggle competition: {competition}')
    logger.info(f'Output directory: {output_path.absolute()}')
    
    try:
        # Use kaggle CLI to download competition data
        cmd = [
            'kaggle', 'competitions', 'download',
            '-c', competition,
            '-p', str(output_path),
            '--force'  # Overwrite existing files
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f'Kaggle download output: {result.stdout}')
        
        # Extract zip file if downloaded
        zip_file = output_path / f'{competition}.zip'
        if zip_file.exists():
            logger.info(f'Extracting {zip_file}')
            import zipfile
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(output_path)
            
            # Remove zip file after extraction
            zip_file.unlink()
            logger.info('Zip file extracted and removed')
        
        logger.info('DONE: Kaggle data download completed')
        
    except subprocess.CalledProcessError as e:
        logger.error(f'Kaggle download failed: {e.stderr}')
        raise RuntimeError(f'Failed to download from Kaggle: {e.stderr}')
    except FileNotFoundError:
        logger.error('Kaggle CLI not found. Install with: pip install kaggle')
        raise RuntimeError('Kaggle CLI not installed')


def verify_downloaded_files(
    data_dir: str = RAW_DATA_DIR,
    expected_files: list = EXPECTED_FILES
) -> bool:
    '''
        Verify that all expected files were downloaded.
        
        Returns True if all files exist, False otherwise.
    '''
    data_path = Path(data_dir)
    missing_files = []
    
    for file_name in expected_files:
        file_path = data_path / file_name
        if not file_path.exists():
            missing_files.append(file_name)
        else:
            # Log file size
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f'Found: {file_name} ({size_mb:.2f} MB)')
    
    if missing_files:
        logger.warning(f'Missing files: {missing_files}')
        return False
    
    logger.info('DONE: All expected files verified')
    return True


def run_data_loading(
    competition: str = COMPETITION_NAME,
    output_dir: str = RAW_DATA_DIR,
    skip_if_exists: bool = True
) -> None:
    '''
        Run data loading pipeline.
        
        Downloads competition data from Kaggle and verifies all files are present.
        
        Args:
            competition: Kaggle competition name
            output_dir: Directory to save downloaded files
            skip_if_exists: Skip download if all files already exist
    '''
    logger.info('Starting data loading pipeline')
    
    # Check if files already exist
    if skip_if_exists and verify_downloaded_files(output_dir):
        logger.info('All data files already exist, skipping download')
        return
    
    # Check credentials
    if not check_kaggle_credentials():
        raise RuntimeError(
            'Kaggle credentials not found. Please either:\n'
            '1. Create ~/.kaggle/kaggle.json with your API credentials, or\n'
            '2. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables\n'
            'Get your API key from: https://www.kaggle.com/settings'
        )
    
    # Download data
    download_from_kaggle(competition, output_dir)
    
    # Verify download
    if not verify_downloaded_files(output_dir):
        raise RuntimeError('Downloaded files verification failed')
    
    logger.info('Data loading pipeline completed successfully')
    
    gc.collect()


# ---------- Main function ---------- #
if __name__ == '__main__':
    run_data_loading()


# ---------- All exports ---------- #
__all__ = ['run_data_loading', 'download_from_kaggle', 'verify_downloaded_files']
