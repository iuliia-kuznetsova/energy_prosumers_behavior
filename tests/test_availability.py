'''Checks that features respect the competition's reveal delay, and that folds do not split a day.'''

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# Point writes at a temp dir before the pipeline modules read the env var.
os.environ['PROCESSED_DATA_DIR'] = tempfile.mkdtemp(prefix='enefit_processed_')
os.environ['MODELS_DIR'] = tempfile.mkdtemp(prefix='enefit_models_')

import pandas as pd
import polars as pl
import unittest

from src.preds.data_merging import DataMerging
from src.preds.feature_engineering import FeatureEngineering
from src.preds.modelling_catboost import (
    DEBUG_PREDICTION_UNITS,
    HOLDOUT_START,
    block_day,
    holdout_mask,
    limit_to_prediction_units,
    rolling_origin_masks,
    scaled_tree_count,
    validation_month_starts,
)


def _dt(text: str) -> datetime:
    return datetime.strptime(text, '%Y-%m-%d %H:%M:%S')


class FoldTests(unittest.TestCase):
    def test_validation_months_are_the_three_months_before_april(self):
        starts = validation_month_starts(3)
        # January, February, March. April is the holdout, so it is not a fold.
        self.assertEqual(
            [start.date().isoformat() for start in starts],
            ['2023-01-01', '2023-02-01', '2023-03-01'],
        )

    def test_holdout_keeps_a_batch_on_one_side(self):
        df = pd.DataFrame({
            'data_block_id': [1, 1, 2, 2],
            'datetime': pd.to_datetime([
                '2023-03-31 00:00:00', '2023-03-31 01:00:00',
                '2023-04-01 00:00:00', '2023-04-01 01:00:00',
            ]),
        })
        mask = holdout_mask(df)
        # March stays in training. April is the holdout. The two hours of a day agree.
        self.assertEqual(mask.tolist(), [False, False, True, True])
        self.assertTrue(block_day(df).min() < HOLDOUT_START)

    def test_scored_month_does_not_overlap_the_early_stopping_slice(self):
        days = pd.date_range('2023-01-01', '2023-03-31', freq='D')
        df = pd.DataFrame({
            'data_block_id': range(len(days)),
            'datetime': days,
        })
        masks = rolling_origin_masks(df, pd.Timestamp('2023-03-01'))
        # fit, inner and val are three different sets of batches.
        self.assertFalse((masks['fit'] & masks['inner']).any())
        self.assertFalse((masks['fit'] & masks['val']).any())
        self.assertFalse((masks['inner'] & masks['val']).any())
        # The model that gets scored is trained on fit plus the inner slice, not on March.
        self.assertTrue((masks['train'] == (masks['fit'] | masks['inner'])).all())
        self.assertEqual(df.loc[masks['val'], 'datetime'].min(), pd.Timestamp('2023-03-01'))
        self.assertEqual(df.loc[masks['inner'], 'datetime'].min(), pd.Timestamp('2023-02-15'))

    def test_tree_count_uses_the_best_index_plus_one_and_scales(self):
        # Index 9 means 10 trees. 150/100 scales that to 15.
        self.assertEqual(scaled_tree_count(9, 100, 150, fallback=50), 15)
        # No reported iteration: scale the search budget instead.
        self.assertEqual(scaled_tree_count(None, 100, 150, fallback=50), 75)
        # Index 9 is 10 trees. Without scaling, the larger train set does not add trees.
        self.assertEqual(scaled_tree_count(9, 100, 150, fallback=50, scale_trees=False), 10)
        self.assertEqual(scaled_tree_count(None, 100, 150, fallback=50, scale_trees=False), 50)


class CliFlagTests(unittest.TestCase):
    def test_main_preds_accepts_debug_and_no_tree_scale(self):
        from src.preds.main_preds import parse_args

        args = parse_args(['--debug', '--no-tree-scale'])
        self.assertTrue(args.debug)
        self.assertTrue(args.no_tree_scale)

    def test_modelling_accepts_debug_and_no_tree_scale(self):
        from src.preds.modelling_catboost import parse_args

        args = parse_args(['--debug', '--no-tree-scale'])
        self.assertTrue(args.debug)
        self.assertTrue(args.no_tree_scale)


class DebugUnitTests(unittest.TestCase):
    def test_debug_list_is_the_four_requested_units(self):
        self.assertEqual(DEBUG_PREDICTION_UNITS, [3, 19, 48, 60])

    def test_limit_keeps_only_the_debug_units_that_exist(self):
        df = pd.DataFrame({
            'prediction_unit_id': [3, 19, 2, 48, 48],
            'value': [1, 2, 3, 4, 5],
        })
        limited = limit_to_prediction_units(df, DEBUG_PREDICTION_UNITS)
        # Unit 60 is absent, so it is skipped. Unit 2 is not in the debug list.
        self.assertEqual(sorted(limited['prediction_unit_id'].tolist()), [3, 19, 48, 48])


class BaselineTests(unittest.TestCase):
    def test_lag_48_baseline_is_the_power_from_two_days_ago(self):
        from src.preds import modelling_catboost as modelling

        rows = []
        start = pd.Timestamp('2023-04-01')
        for hour in range(72):
            rows.append({
                'row_id': hour,
                'datetime': start + pd.Timedelta(hours=hour),
                'data_block_id': 100 + hour // 24,
                'prediction_unit_id': 0,
                'is_consumption': 1,
            })
        df = pd.DataFrame(rows)
        # Saved target: the power value equals the hour index, so the error of a 48h lag is exactly 48.
        power = pd.DataFrame({'row_id': list(range(72)), 'power': [float(hour) for hour in range(72)]})
        power_path = Path(modelling.PROCESSED_DATA_DIR) / 'power.parquet'
        power_path.parent.mkdir(parents=True, exist_ok=True)
        power.to_parquet(power_path)

        scored = df['datetime'] >= pd.Timestamp('2023-04-03')
        stats = modelling.baseline_power_mae(df, scored)
        self.assertAlmostEqual(stats['baseline_lag_48_mae_power'], 48.0)
        self.assertEqual(stats['baseline_lag_48_n'], 24)


class LagTests(unittest.TestCase):
    def test_short_target_lag_is_rejected(self):
        with self.assertRaises(ValueError):
            FeatureEngineering(mode='train', n_lag=[12, 48])

    def test_lag_follows_the_clock_when_an_hour_is_missing(self):
        # Hour 60 is missing. A row shift of 48 from hour 100 would land on hour 51.
        hours = [hour for hour in range(0, 101) if hour != 60]
        base = datetime(2023, 1, 1)
        frame = pl.DataFrame({
            'prediction_unit_id': [0] * len(hours),
            'is_consumption': [True] * len(hours),
            'datetime': [base + timedelta(hours=hour) for hour in hours],
            'capacity_factor': [float(hour) for hour in hours],
            'electricity_price': [1.0] * len(hours),
            'gas_price': [1.0] * len(hours),
            'hw_temperature': [1.0] * len(hours),
            'hw_surface_pressure': [1.0] * len(hours),
            'hw_cloudcover_total': [1.0] * len(hours),
            'hw_total_precipitation': [1.0] * len(hours),
            'hw_windspeed_10m': [1.0] * len(hours),
            'hw_winddirection_10m': [1.0] * len(hours),
            'hw_direct_solar_radiation': [1.0] * len(hours),
            'hw_diffuse_radiation': [1.0] * len(hours),
        }).with_columns(pl.col('datetime').cast(pl.Datetime('us')))

        engineered = FeatureEngineering(mode='train', n_lag=[48])._create_lag_features(frame)
        at_100 = engineered.filter(
            pl.col('datetime') == pl.lit(base + timedelta(hours=100)).cast(pl.Datetime('us'))
        )
        # 48 clock hours before hour 100 is hour 52, on the far side of the gap.
        self.assertEqual(at_100['capacity_factor_lag_48'][0], 52.0)
        self.assertNotIn('capacity_factor_lag_12', engineered.columns)
        self.assertNotIn('capacity_factor_lag_24', engineered.columns)
        # 24h is legal for a price, because the merged price is already yesterday's price.
        self.assertIn('electricity_price_lag_24', engineered.columns)


class MergeTests(unittest.TestCase):
    def _merged(self) -> pl.DataFrame:
        power = pl.DataFrame({
            'row_id': [1, 2],
            'datetime': [_dt('2023-03-03 00:00:00'), _dt('2023-03-03 12:00:00')],
            'data_block_id': [10, 10],
            'prediction_unit_id': [0, 0],
            'is_consumption': [True, True],
            'county': [0, 0],
            'is_business': [False, False],
            'product_type': [1, 1],
            'target': [50.0, 60.0],
        })
        client = pl.DataFrame({
            # Known capacity: date is 2 days behind the batch that reveals it.
            'date': [datetime(2023, 3, 1).date(), datetime(2023, 3, 3).date()],
            'data_block_id': [10, 12],
            'county': [0, 0],
            'is_business': [False, False],
            'product_type': [1, 1],
            'installed_capacity': [100.0, 999.0],
        })
        electricity = pl.DataFrame({
            'forecast_date': [_dt('2023-03-02 00:00:00'), _dt('2023-03-02 12:00:00'), _dt('2023-03-03 00:00:00')],
            'data_block_id': [10, 10, 11],
            'euros_per_mwh': [40.0, 30.0, -5.0],
        })
        gas = pl.DataFrame({
            'forecast_date': [_dt('2023-03-02 00:00:00'), _dt('2023-03-03 00:00:00')],
            'data_block_id': [10, 11],
            'lowest_price_per_mwh': [10.0, 80.0],
            'highest_price_per_mwh': [20.0, 80.0],
        })
        weather_values = {
            'temperature': [1.0, 9.0, 1.5],
            'cloudcover_total': [0.2, 0.2, 0.2],
            'total_precipitation': [0.0, 0.0, 0.0],
            '10_metre_u_wind_component': [0.0, 0.0, 0.0],
            '10_metre_v_wind_component': [0.0, 0.0, 0.0],
            'direct_solar_radiation': [0.0, 0.0, 0.0],
            'surface_solar_radiation_downwards': [0.0, 0.0, 0.0],
        }
        forecast = pl.DataFrame({
            'latitude': [59.0, 59.0, 59.0],
            'longitude': [24.0, 24.0, 24.0],
            'data_block_id': [10, 11, 10],
            # Block 11 is the next day's issue for the same valid time, and must not be used.
            'forecast_datetime': [
                _dt('2023-03-03 00:00:00'), _dt('2023-03-03 00:00:00'), _dt('2023-03-03 12:00:00'),
            ],
            'hours_ahead': [22, 10, 34],
            **weather_values,
        })
        historical = pl.DataFrame({
            'datetime': [_dt('2023-03-01 00:00:00'), _dt('2023-03-01 12:00:00'), _dt('2023-03-03 00:00:00')],
            'data_block_id': [8, 8, 10],
            'latitude': [59.0, 59.0, 59.0],
            'longitude': [24.0, 24.0, 24.0],
            'temperature': [4.0, 6.0, 8.0],
            'surface_pressure': [1000.0, 1000.0, 1000.0],
            'rain': [0.0, 0.0, 0.0],
            'snowfall': [0.0, 0.0, 0.0],
            'cloudcover_total': [10.0, 10.0, 10.0],
            'windspeed_10m': [1.0, 1.0, 1.0],
            'winddirection_10m': [1.0, 1.0, 1.0],
            'direct_solar_radiation': [0.0, 0.0, 0.0],
            'diffuse_radiation': [0.0, 0.0, 0.0],
        })
        mapping = pl.DataFrame({'county': [0], 'latitude': [59.0], 'longitude': [24.0]})
        frames = {
            'power_df': power,
            'client_df': client,
            'electricity_df': electricity,
            'gas_df': gas,
            'forecast_weather_df': forecast,
            'historical_weather_df': historical,
            'mapping_df': mapping,
        }
        for name, frame in frames.items():
            casts = []
            if 'datetime' in frame.columns:
                casts.append(pl.col('datetime').cast(pl.Datetime('us')))
            if 'forecast_date' in frame.columns:
                casts.append(pl.col('forecast_date').cast(pl.Datetime('us')))
            if 'forecast_datetime' in frame.columns:
                casts.append(pl.col('forecast_datetime').cast(pl.Datetime('us')))
            if 'data_block_id' in frame.columns:
                casts.append(pl.col('data_block_id').cast(pl.Int32))
            frames[name] = frame.with_columns(casts)
        return DataMerging(mode='train').fit_transform(frames)

    def test_joined_values_are_the_delayed_ones(self):
        merged = self._merged()
        midnight = merged.filter(pl.col('datetime') == pl.lit(_dt('2023-03-03 00:00:00')).cast(pl.Datetime('us')))
        # Capacity 100 is from two days earlier. 999 is the same-day capacity and must not be joined.
        self.assertEqual(midnight['installed_capacity'][0], 100.0)
        self.assertEqual(midnight['client_block_id'][0], 10)
        # Price 40 is yesterday. -5 is today's price and must not be joined.
        self.assertEqual(midnight['electricity_price'][0], 40.0)
        self.assertEqual(midnight['electricity_block_id'][0], 10)
        # Gas 15 is yesterday's midpoint. 80 is today's price.
        self.assertEqual(midnight['gas_price'][0], 15.0)
        self.assertEqual(midnight['gas_block_id'][0], 10)
        # Forecast temperature 1 is this batch's issue. 9 is the next batch's update.
        self.assertEqual(midnight['fw_temperature'][0], 1.0)
        self.assertEqual(midnight['fw_block_id'][0], 10)
        # Observed temperature 4 is 48h earlier. 8 is the same hour and must not be joined.
        self.assertEqual(midnight['hw_temperature'][0], 4.0)
        self.assertEqual(midnight['hw_block_id'][0], 8)
        # capacity_factor uses the known capacity: 50 / 100.
        self.assertAlmostEqual(midnight['capacity_factor'][0], 0.5)

    def test_source_batches_are_not_newer_than_the_target(self):
        merged = self._merged()
        for column in ['client_block_id', 'electricity_block_id', 'gas_block_id', 'fw_block_id', 'hw_block_id']:
            ok = merged.select(
                (pl.col(column).is_null() | (pl.col(column) <= pl.col('data_block_id'))).alias('ok')
            )
            self.assertTrue(ok['ok'].all(), column)


class RealBlockTests(unittest.TestCase):
    def test_real_rows_only_see_their_batch_or_an_older_one(self):
        raw = Path('data/raw')
        if not (raw / 'train.csv').exists():
            self.skipTest('raw competition files are not in data/raw')

        from src.preds.data_preprocessing import DataPreparation

        def _load(name, low, high):
            # Only a few batches, so the check does not read the whole weather archive into memory.
            return (
                pl.scan_csv(raw / name)
                .filter(pl.col('data_block_id').is_between(low, high))
                .collect()
            )

        raw_data = {
            'power_df': _load('train.csv', 40, 48),
            'client_df': _load('client.csv', 40, 48),
            'electricity_df': _load('electricity_prices.csv', 40, 48),
            'gas_df': _load('gas_prices.csv', 40, 48),
            'forecast_weather_df': _load('forecast_weather.csv', 40, 48),
            'historical_weather_df': _load('historical_weather.csv', 38, 48),
            'mapping_df': pl.read_csv(raw / 'weather_station_to_county_mapping.csv'),
        }
        processed = DataPreparation(mode='train').fit_transform(raw_data)
        merged = DataMerging(mode='train').fit_transform(processed)

        for column in ['client_block_id', 'electricity_block_id', 'gas_block_id', 'fw_block_id', 'hw_block_id']:
            ok = merged.select(
                (pl.col(column).is_null() | (pl.col(column) <= pl.col('data_block_id'))).alias('ok')
            )
            self.assertTrue(bool(ok['ok'].all()), column)

        # The day-ahead forecast, when it matched, comes from the same batch as the target.
        same_issue = merged.filter(pl.col('fw_block_id').is_not_null()).select(
            (pl.col('fw_block_id') == pl.col('data_block_id')).alias('ok')
        )
        self.assertGreater(same_issue.height, 0)
        self.assertTrue(bool(same_issue['ok'].all()))


if __name__ == '__main__':
    unittest.main()
