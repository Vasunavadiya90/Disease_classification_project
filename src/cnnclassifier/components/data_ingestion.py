from cnnclassifier.constants import *


from cnnclassifier.entity.config_entity import DataIngestionConfig
import zipfile
import os
from pathlib import Path
import urllib.request as request

class DataIngestion:
    def __init__(self , config:DataIngestionConfig):
        self.config = config


    def download_file(self):

            source_url = self.config.source_URL
            local_file = self.config.local_data_file

            # CREATE DIRECTORY FIRST
            os.makedirs(os.path.dirname(local_file), exist_ok=True)

            print("Downloading file from:")
            print(source_url)

            filename, headers = request.urlretrieve(
                url=source_url,
                filename=local_file
            )

            print("Downloaded file:")
            print(filename)

            print("Download completed successfully")
    def extract_zip_file(self):

        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)

        print("Checking zip validity...")

        if not zipfile.is_zipfile(self.config.local_data_file):
            raise Exception(
                f"Invalid ZIP file: {self.config.local_data_file}"
            )

        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

        print("Extraction completed")