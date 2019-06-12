import pandas as pd
import numpy as np
import os
from os import path

#from featureEngineering import create_dirs

INPUT_DIR = "../data/"
OUTPUT_DIR = "../data/aggregated_data/"

def read_data(file_name, drop_na=False):
    """
    Read credit data in the .csv file and data types from the .json file.

    Inputs:
        - data_file (string): name of the data file.
        - drop_na (bool): whether to drop rows with any missing values

    Returns:
        (DataFrame) clean data set with correct data types

    """
    data = pd.read_csv(file_name)

    if drop_na:
        data.dropna(axis=0, inplace=True)

    data['date'] = pd.to_datetime(data['date'])

    return data


def slice_data(data, var, years):
    
    (start_year, end_year) = years
    early_start_date = str(start_year - 5) + "-01-01"
    end_date = str(end_year + 1) + "-01-01"

    full_data = data[(data[var] >= early_start_date) & (data[var] < end_date)]

    return full_data


def clean(df):
    df.date = pd.to_datetime(df.date)
    df = df.drop(columns=['eventid', 'city', 'loc_id'])
    df['year'] = df.date.dt.year
    df = df.assign(loc_id=(
                df['latitude'].astype(str) + '_' + df['longitude'].astype(
            str)).astype('category').cat.codes)
    df['unique_id'] = df[['loc_id', 'year']].apply(tuple, axis=1)

    for col in ['elevation', 'DIS_LAKE',
                'DIS_MAJOR_RIVER', 'DIS_OCEAN', 'DIS_RIVER', 'MER1990_40',
                'MER1995_40',
                'MER2000_40', 'MER2005_40', 'POPGPW_1990_40', 'POPGPW_1995_40',
                'POPGPW_2000_40', 'POPGPW_2005_40', 'PRECAVNEW80_08',
                'TEMPAV_8008']:
        df[col] = df[col].astype(float)

    return df


def aggregate(df):
    df_list = list()
    col_names = list()
    for col in ['attacktype', 'targettype', 'targetsubtype', 'group_name']:
        temp_series = df.groupby('unique_id')[col].nunique()
        df_list.append(temp_series)
        col_names.append(col)

    for col in ['nkill', 'nwound']:
        temp_series = df.groupby('unique_id')[col].sum()
        df_list.append(temp_series)
        col_names.append(col)

    for col in ['elevation', 'DIS_LAKE', 'DIS_MAJOR_RIVER', 'DIS_OCEAN',
                'DIS_RIVER',
                'PRECAVNEW80_08', 'TEMPAV_8008', 'ethin_div', 'HighRelig',
                'ChrCatP',
                'ReligCatP', 'year', 'loc_id']:
        temp_series = df.groupby('unique_id')[col].unique().apply(
            lambda x: x[0])
        df_list.append(temp_series)
        col_names.append(col)

    for col in ['MER{}_40', 'POPGPW_{}_40']:
        if df.year.max() < 1995:
            year = 1990
        elif df.year.max() < 2000:
            year = 1995
        elif df.year.max() < 2005:
            year = 2000
        else:
            year = 2005
        temp_series = df.groupby('unique_id')[col.format(year)].unique().apply(
            lambda x: x[0])
        df_list.append(temp_series)

    col_names += ['MER_40', 'POPGPW_40']

    final_df = pd.concat(df_list, axis=1, keys=col_names)

    final_df['attacked'] = 1

    return final_df


def fill_year_gaps(df, start_year, end_year):
    new_df = pd.DataFrame()

    for block_id in df.loc_id.values:
        block = df[df['loc_id'] == block_id]

        exclude = []
        for year in range(start_year, end_year + 1):
            if year not in block.year.values:
                # append an empty row
                block = block.append(pd.Series(), ignore_index=True)

                # fill in year, target, and other values
                block.iloc[-1, -5] = year
                block.iloc[-1, -1] = 0
                block.fillna(method="ffill", inplace=True)

        for to_exclude in exclude:
            block.loc[block.year == to_exclude, ['attacktype', 'targettype',
                                                 'targetsubtype', 'group_name',
                                                 'nkill', 'nwound']] = np.nan

        new_df = pd.concat([new_df, block])

    return new_df


def sum_past_k_year_data(df, col, loc_id, year, k):
    agg_num = df.groupby(['loc_id', 'year']).sum().loc[loc_id].loc[
              year - k + 1:year - 1].loc[:, col].sum()

    return agg_num


def count_past_k_year_data(df, col, loc_id, year, k):
    count = len(set(df.groupby(['loc_id', 'year', col]).count().loc[loc_id].loc[
                    int(year) - k + 1:int(year) - 1].index.get_level_values(
        col).values.tolist()))

    return count


def add_past_k(df, full_df, k):
    """

    :param df:
    :param full_df:
    :param k:
    :return:
    """
    for col in ['nkill', 'nwound']:
        col_name = col + '_past_{}'.format(k)
        df[col_name] = df.apply(
            lambda row: sum_past_k_year_data(full_df, col, row['loc_id'],
                                             row['year'], k), axis=1)

    for col in ['attacktype', 'targettype', 'group_name']:
        col_name = col + '_past_{}'.format(k)
        df[col_name] = df.apply(
            lambda row: count_past_k_year_data(full_df, col, row['loc_id'],
                                               row['year'], k), axis=1)

    return df


#----------------------------------------------------------------------------#
if __name__ == "__main__":

    #gtd = read_data("rev_gtd4.csv")

    for i in range(2002, 2006, 2):
        data = read_data("rev_gtd4.csv")
        test_year = (i, i + 1)
        train_year = (i - 6, i - 1)
        #print("Train year: {}, test year: {}".format(train_year, test_year))
        test_train = [test_year, train_year]
        counter = 0
        name = ["test", "train"]
        for i, year in enumerate(test_train):
            folder_name = "./train test sets/Batch {}".format(counter)
            if not path.exists(folder_name):
                os.mkdir(folder_name)
            start_year = year[0]
            end_year = year[1]
            # CHECKING IF THE FILE EXISTS
            save_file = folder_name + "/{}.csv".format(name[i])
            if path.exists(save_file):
                continue
            data = slice_data(data, "date", year)
            cleaned = clean(data)
            pre_ag = cleaned[(cleaned.year <= end_year) & (cleaned.year >= start_year)]
            aggregated = aggregate(pre_ag)
            filled = fill_year_gaps(aggregated, start_year, end_year)
            Ks = [2, 5]
            for col in ['nkill', 'nwound']:
                for k in Ks:
                    col_name = col + '_past_{}'.format(k)
                    filled[col_name] = filled.apply(lambda row : sum_past_k_year_data(cleaned, col, row['loc_id'], row['year'], k), axis=1)

            for col in ['attacktype', 'targettype', 'group_name']:
                for k in Ks:
                    col_name = col + '_past_{}'.format(k)
                    filled[col_name] = filled.apply(lambda row : count_past_k_year_data(cleaned, col, row['loc_id'], row['year'], k), axis=1)
            filled.drop(columns=['nkill', 'nwound', 'attacktype', 'targettype', 'group_name', 'targetsubtype'], inplace=True)
            # SAVING THE FILE
            filled.to_csv(save_file, index=False)
            #print("{} DONE!".format(year))
        counter += 1
