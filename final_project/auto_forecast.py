from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from autogluon.timeseries.splitter import ExpandingWindowSplitter

def date_columns_fit_transform(df):
    for column in df.columns:
        try:
            converted = pd.to_datetime(df[column], errors='coerce')
            if converted.notna().all() and (converted > datetime(2010, 1, 1)).all():
                df[column] = converted
        except Exception as e:
            print(f"Erro ao processar a coluna {column}: {e}")

    return df

def get_datetime_column_name(df):
    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            return column
    return None

def get_target_col(df):
    while True:
        target = input('Provide the target column name: ')
        if target in df.columns:
            print(df[target].head())
            option = input(f'Chosen target column name: {target}. Is that correct? (Y/n): ')
            if option in ('Y', 'y', ''):
                return target
        else:
            print('Column name not found. Try again.')

def get_dataframe_from_filename():
    while True:
        try:
            filename = input('Provide the filename with historical data: ')
            df = pd.read_csv(filename)
            df = date_columns_fit_transform(df)
            return df
        except FileNotFoundError:
            print('File not found. Try again.')
        except Exception as e:
            print(f"Error loading file: {e}")

def get_identifier(df): 

    while True:
        option = input('Would you like to add an identifier column? (N/y): ').strip().lower()
        
        if option in ('n', ''):
            print('No identifier column selected. A default identifier will be used.')
            return 'identifier'
        
        elif option == 'y':
            identifier = input('Provide the identifier column name: ').strip()
            
            if identifier in df.columns:
                print(df[identifier].head())
                confirm = input(f'Chosen identifier column: {identifier}. Confirm? (Y/n): ').strip().lower()
                
                if confirm in ('y', ''):
                    return identifier
                else:
                    print('Try selecting the identifier column again.')
            else:
                print(f'Column "{identifier}" not found. Trying again...')
        
        else:
            print('Invalid option. Please enter "Y" for yes or "N" for no.')

def get_predictions(df, target, prediction_length, use_covariates):

    identifier = get_identifier(df)

    if identifier == 'identifier':
        df['identifier'] = 'standard'

    date_column = get_datetime_column_name(df)
    if date_column is None:
        raise ValueError("No valid datetime column found in the DataFrame.")

    freq = pd.infer_freq(df[date_column])
    if freq is None:
        raise ValueError("Could not infer frequency. Ensure timestamps are evenly spaced.")

    if use_covariates:
        future_df = get_future_dataframe_from_filename()
        covariates_list = get_covariates_list(future_df)
        future_df['identifier'] = 'standard'
    
    timeseries_df = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column= identifier,
        timestamp_column=date_column
    )

    predictor_args = {
        "prediction_length": prediction_length,
        "target": target,
        "eval_metric": "RMSE",
        "freq": freq
    }
    
    if use_covariates:
        future_timeseries_df = TimeSeriesDataFrame.from_data_frame(
            future_df,
            id_column = identifier,
            timestamp_column = date_column
        )

        predictor_args["known_covariates_names"] = covariates_list

    predictor = TimeSeriesPredictor(**predictor_args)

    predictor.fit(
        timeseries_df,
        presets='best_quality',
        val_step_size=prediction_length,
        random_seed=42,
        num_val_windows=2,
        time_limit=60
    )

    if use_covariates:
        predictions = predictor.predict(timeseries_df, 
                                        random_seed=42, 
                                        known_covariates = future_timeseries_df)
    else:
        predictions = predictor.predict(timeseries_df, 
                                        random_seed=42)

    return timeseries_df, predictor, predictions

def use_future_covariates():
    
    while True:
        option = input('Include future covariates? (N/y): ')
        if option in ('Y', 'y'):
            use_covariates = True
            break
        elif option in ('N', 'n', ''):
            use_covariates = False
            break
        else:
            print('Invalid option. Try again.')

    return use_covariates
    
def get_future_dataframe_from_filename():

    while True:
        try:
            future_filename = input('Provide the filename with future covariates: ')
            df_future = pd.read_csv(future_filename)
            df_future = date_columns_fit_transform(df_future)
            return df_future
        except FileNotFoundError:
            print('Invalid filename. Try again: ')
        except Exception as e:
            print(f'Error loading file: {e}. Try again.')

def get_covariates_list(future_df):

    covariates_list = []
    for column in future_df.columns:

        while True:
            use_column = input(f'Column {column} found. Use it as a future covariate? (Y/n)')
            if use_column in ('Y', 'y', ''):
                covariates_list.append(column)
                print(f'Column {column} added to future covariates!')
                break
            elif use_column in ('N', 'n'):
                print(f'Column {column} ignored.')
                break
            else:
                print(f'Invalid option. Trying again.')
    
    return covariates_list

if __name__ == '__main__':
    df = get_dataframe_from_filename()

    print('\nGenerated DataFrame:')
    print(df.head())
    print(f'DataFrame has the following columns: {df.columns.to_list()}')

    target = get_target_col(df)
    use_covariates = use_future_covariates()

    timeseries_df, predictor, predictions = get_predictions(df, target, 24, use_covariates)

    print(predictions)

    predictor.plot(timeseries_df, predictions, max_history_length=120)

    plt.savefig("forecast_plot.png")
    plt.show()
