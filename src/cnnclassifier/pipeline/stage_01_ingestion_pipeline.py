import os
import sys
from pathlib import Path

# Automatically resolve and change to project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))

from cnnclassifier.config.configuration import configurationManager
from cnnclassifier.components.data_ingestion import DataIngestion
from cnnclassifier import logger

STAGE_NAME = "data_Ingestion_stage"


class dataIngestionTrainingPipeline:
    def __init__(self):
        
        pass

    def main(self):
        config = configurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config = data_ingestion_config)
        
        data_ingestion.download_file()

        print("\nFILE EXISTS:")
        print(os.path.exists(data_ingestion.config.local_data_file))

        print("\nFILE SIZE:")
        print(os.path.getsize(data_ingestion.config.local_data_file))

        print("\nFIRST 200 BYTES:")

        with open(data_ingestion.config.local_data_file, "rb") as f:
            print(f.read(200))

        data_ingestion.extract_zip_file()

        print("Data ingestion completed successfully!")



if __name__ == "__main__":
    try:

        logger.info(f">>>>>>>> {STAGE_NAME} started <<<<<<<<")
        obj = dataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")


    except Exception as e:
        logger.exception(e)
        raise e