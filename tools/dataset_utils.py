import asyncio
import datetime
import decimal
import json
import os
import random
import string
from abc import ABC, abstractmethod
from collections import deque, namedtuple
from dataclasses import dataclass
import re
import httpx
import numpy as np
import openai
import pandas as pd
from loguru import logger
from tenacity import RetryError, retry, stop_after_attempt, wait_random_exponential


class MyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.float64):
            return round(obj, 3)
        elif isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(obj, (decimal.Decimal, float)):
            return round(float(obj), 3)
        return str(obj)


class DatasetUtils:
    MISSING_THRESHOLD = 0.7
    MAX_LEN_THRESHOLD = 50
    MEAN_LEN_THRESHOLD = 25

    @staticmethod
    def get_csv_dataframe(ref_dir, csv, columns=None, nrows=10_000):
        csv_path = os.path.join(ref_dir, csv)
        return pd.read_csv(csv_path, usecols=columns, low_memory=False, nrows=nrows)

    @staticmethod
    def get_column_availability(dataframe):
        cols = dataframe.columns
        missing_values = dataframe.isnull().mean()
        max_lengths = dataframe.astype(str).applymap(len).max()
        mean_lengths = dataframe.astype(str).applymap(len).mean()

        availabilities = {}
        available_cols = []
        for col in cols:
            if col.strip() == '' or "Unnamed: " in col.lower():
                availabilities[col] = "not-meaningful-column-name"
            elif missing_values[col] > DatasetUtils.MISSING_THRESHOLD:
                availabilities[col] = "too-much-missing"
            elif max_lengths[col] > DatasetUtils.MAX_LEN_THRESHOLD:
                availabilities[col] = "too-long-max-length"
            elif mean_lengths[col] > DatasetUtils.MEAN_LEN_THRESHOLD:
                availabilities[col] = "too-long-mean-length"
            else:
                availabilities[col] = "available"
                available_cols.append(col)
        return available_cols, availabilities

    @staticmethod
    def get_filtered_dataframe(dataframe, available_cols):
        return dataframe[available_cols]

    @staticmethod
    def get_meta_info(dataframe, n_typical_vals=5):
        cols = dataframe.columns
        meta_info = []
        for col in cols:
            num_nans = dataframe[col].isnull().sum()
            num_uniques = dataframe[col].nunique()
            typical_values = dataframe[col].value_counts().head(n_typical_vals).index.tolist()
            meta_info.append({
                "column_name": col,
                "typical_values": typical_values,
                "num_uniques": num_uniques,
                "num_nans": num_nans
            })
        return meta_info

    @staticmethod
    def get_sampled_csv_raw(dataframe, n=5):
        cols = dataframe.columns
        weights = pd.Series(1e-5, index=dataframe.index)
        for col in cols:
            weights += dataframe[[col]].apply(lambda col: col.map(1 / col.value_counts())).sum(axis=1)
        sampled_dataframe = dataframe.sample(min(n, len(dataframe)), weights=weights)
        return sampled_dataframe.to_csv(index=False)

class CSV:
    column_meta_type = namedtuple("ColumnMeta", ['column_name', 'refined_name', 'description', 'dtype', 'data_type', 'parse_type', 'size', 'num_nan', 'nan_ratio', 'num_unique', 'unique_ratio'])

    def __init__(self, ref, csv, src_table_root, src_meta_root):
        self.ref = ref
        self.csv = csv
        self.src_table_root = src_table_root
        self.src_meta_root = src_meta_root
        self.csv_path = os.path.join(src_table_root, ref, csv)
        self.col_annot_path = os.path.join(src_meta_root, ref, csv + ".annot.json")
        self.meta_info_path = os.path.join(src_meta_root, ref, "dataset_meta_info.json")
        self.load_()
        self.make_meta_()
        self.make_pool_()

    def load_(self):
        with open(self.meta_info_path, 'r') as fp:
            self.meta_info = json.load(fp)
        with open(self.col_annot_path, 'r') as fp:
            self.col_annots = json.load(fp)
        col_annot_mapping = {col_annot['column_name']: col_annot for col_annot in self.col_annots if col_annot['column_name'] != "" and "Unnamed" not in col_annot['column_name']}
        self.cols = list(col_annot_mapping.keys())
        self.refined_name_mapping = {k: v['refined_name'] for k, v in col_annot_mapping.items()}
        self.refined_cols = [self.refined_name_mapping[col] for col in self.cols]
        self.date_cols = [col for col in self.cols if col_annot_mapping[col]['parse_type'] == 'date' or 'date' in col.lower() or 'date' in self.refined_name_mapping[col].lower()]
        self.dataframe = pd.read_csv(self.csv_path, usecols=self.cols, parse_dates=self.date_cols, low_memory=False).dropna()

    def make_meta_(self):
        self.dataset_title = self.meta_info['title']
        self.dataset_subtitle = self.meta_info['subtitle']
        self.dataset_description = self.meta_info['description']
        self.col_meta = {}
        for col_annot in self.col_annots:
            column_name = col_annot['column_name']
            if column_name == "" or "Unnamed" in column_name:
                logger.debug(f"empty or unnamed column detected in {self.ref}:{self.csv}")
                continue
            refined_name = col_annot['refined_name']
            description = col_annot['description']
            data_type = col_annot['data_type']
            parse_type = col_annot['parse_type']
            dtype = self.dataframe[column_name].dtype
            size = len(self.dataframe[column_name])
            num_nan = self.dataframe[column_name].isna().sum()
            nan_ratio = num_nan / size
            num_unique = self.dataframe[column_name].nunique()
            unique_ratio = num_unique / size
            self.col_meta[column_name] = CSV.column_meta_type(
                column_name=column_name,
                refined_name=refined_name,
                description=description,
                data_type=data_type,
                parse_type=parse_type,
                dtype=dtype,
                size=size,
                num_nan=num_nan,
                nan_ratio=nan_ratio,
                num_unique=num_unique,
                unique_ratio=unique_ratio
            )

    def make_pool_(self):
        self.numerical_pool = [
            key for key, val in self.col_meta.items()
            if val.data_type in ('discrete', 'continuous') and val.dtype in (np.dtype('int64'), np.dtype('float64')) and val.nan_ratio < 0.3
        ]
        self.classable_pool = [
            key for key, val in self.col_meta.items()
            if val.unique_ratio < 0.3 and val.num_unique <= 5 and val.data_type in ("nominal", "ordinal")
        ]
        self.unique_pool = [
            key for key, val in self.col_meta.items()
            if ((val.data_type in ('discrete', 'continuous') and val.dtype in (np.dtype('int64'), np.dtype('float64'))) or (val.dtype.kind == 'M')) and val.unique_ratio > 0.95 and val.nan_ratio < 0.1
        ]
        self.enumerable_pool = [
            key for key, val in self.col_meta.items()
            if val.data_type in ('ordinal', 'nominal') and val.num_unique <= 30
        ]

    def sample_subtable(self, types, batch=50):
        pools = {
            'n': self.numerical_pool,
            'c': self.classable_pool,
            'e': self.enumerable_pool,
            'u': self.unique_pool
        }
        cols = []
        dataframe = self.dataframe
        for t in types:
            pool = pools[t.lower()].copy()
            pool = list(set(pool) - set(cols))
            if len(pool) == 0:
                raise ValueError(f"No enough columns for {types} in {self.ref}:{self.csv}")
            cols.append(random.choice(pool))
            if t == 'u':
                dataframe = dataframe.drop_duplicates(cols[-1])
        weights = pd.Series(1e-5, index=dataframe.index)
        for t, col in zip(types, cols):
            if t in 'CE':
                weights += dataframe[[col]].apply(lambda col: col.map(1 / col.value_counts())).sum(axis=1)
        if len(dataframe) < batch:
            raise ValueError(f"No enough rows for {types} in {self.ref}:{self.csv}")
        return dataframe[cols].sample(batch, weights=weights)