import datetime
import decimal
import json
import os
import pickle
import zipfile

import requests
from bs4 import BeautifulSoup
from kaggle.api.kaggle_api_extended import KaggleApi
from langdetect import detect
from loguru import logger
from tqdm import tqdm


class KaggleUtils:
    def __init__(self, data_root, meta_root, max_pages=100, max_data_size_mb=5, log_path="../log/download.log"):
        self.DATA_ROOT = data_root
        self.META_ROOT = meta_root
        self.META_OBJ = "src_table_meta.obj"
        self.MAX_PAGES = max_pages
        self.MAX_DATA_SIZE_MB = max_data_size_mb
        self.api = KaggleApi()
        self.api.authenticate()
        logger.remove(handler_id=None)
        logger.add(log_path)

    def load_or_fetch_datasets(self):
        if self.META_OBJ not in os.listdir(self.META_ROOT):
            logger.info("Dataset meta information not found. Attempting retrieval.")
            datasets_total = []
            for page in tqdm(range(1, self.MAX_PAGES + 1)):
                datasets = self.api.dataset_list(sort_by='votes', page=page, file_type='csv')
                datasets = [d for d in datasets if d.totalBytes < self.MAX_DATA_SIZE_MB * 1024 * 1024]
                datasets_total.extend(datasets)
            logger.info(f"Retrieved {len(datasets_total)} datasets.")
            with open(os.path.join(self.META_ROOT, self.META_OBJ), 'wb') as fp:
                pickle.dump(datasets_total, fp)
        else:
            with open(os.path.join(self.META_ROOT, self.META_OBJ), 'rb') as fp:
                datasets_total = pickle.load(fp)
            logger.info(f"Loaded {len(datasets_total)} datasets from disk.")
        return datasets_total

    @staticmethod
    def get_description(url):
        response = requests.get(url)
        html_content = response.content
        soup = BeautifulSoup(html_content, 'html.parser')
        script_tag = soup.find('script', {'type': 'application/ld+json'})
        json_data = json.loads(script_tag.contents[0])
        return json_data['description']

    @staticmethod
    def detect_language(text):
        try:
            return detect(text)
        except Exception as e:
            logger.warning(e)
            return None

    def download_and_unzip(self, data):
        data_id = data.ref.split('/')[1]
        self.api.dataset_download_files(data.ref, path=os.path.join(self.DATA_ROOT, data_id))
        zip_file = os.path.join(self.DATA_ROOT, data_id, data_id + '.zip')
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(self.DATA_ROOT, data_id))

    @staticmethod
    def save_data_meta_info(data, data_root):
        description = KaggleUtils.get_description(data.url)
        lang = KaggleUtils.detect_language(description)
        data_id = data.ref.split('/')[1]
        meta_info = {
            "id": data_id,
            "title": data.titleNullable,
            "subtitle": data.subtitleNullable,
            "description": description,
            "language": lang
        }
        with open(os.path.join(data_root, data_id, data_id + '.meta.json'), 'w') as fp:
            json.dump(meta_info, fp, ensure_ascii=False, indent=4)

    def is_data_downloaded(self, data):
        data_id = data.ref.split('/')[1]
        return data_id in os.listdir(self.DATA_ROOT)

    def download_datasets(self, datasets_total):
        logger.info(f"Starting download process for {len(datasets_total)} datasets.")
        for data in tqdm(datasets_total):
            if self.is_data_downloaded(data):
                logger.info(f"Dataset {data.ref} already downloaded. Skipping.")
                continue
            try:
                self.download_and_unzip(data)
                self.save_data_meta_info(data, self.DATA_ROOT)
                logger.info(f"Downloaded and unzipped dataset {data.ref}.")
            except zipfile.BadZipFile:
                logger.warning(f"Error unzipping {data.ref}.")
            except Exception as e:
                logger.warning(e)
        logger.info(f"Download process completed. {len(os.listdir(self.DATA_ROOT))} datasets downloaded.")

    def prepare_dataset(self, ref, src_meta_root, src_table_root):
        if ref not in os.listdir(src_meta_root):
            with open(os.path.join(src_table_root, ref, f"{ref}.meta.json"), 'r') as fp:
                meta_info = json.load(fp)
            if meta_info['language'] == 'en':
                os.mkdir(os.path.join(src_meta_root, ref))
                with open(os.path.join(src_meta_root, ref, "dataset_meta_info.json"), 'w') as fp:
                    json.dump(meta_info, fp)
                logger.info(f"Dataset {ref} prepared.")
            else:
                logger.warning(f"Dataset {ref} not in English. Skipping.")
        else:
            logger.info(f"Dataset {ref} already prepared. Skipping.")

    def prepare_all_datasets(self, src_meta_root, src_table_root):
        logger.info(f"Starting preparation process for {len(os.listdir(self.DATA_ROOT))} datasets.")
        for ref in tqdm(os.listdir(self.DATA_ROOT)):
            self.prepare_dataset(ref, src_meta_root, src_table_root)
        logger.info(f"Preparation process completed. {len(os.listdir(src_meta_root))} datasets prepared.")
