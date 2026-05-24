from src.cnnclassifier import logger
from cnnclassifier.pipeline.stage_01_ingestion_pipeline import dataIngestionTrainingPipeline

logger.info("Intro to Disease classification")


import os
from pathlib import Path

STAGE_NAME = "data_Ingestion_stage"


try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started >>>>>>>>>>>>>>>")
    dataingestion = dataIngestionTrainingPipeline()
    dataingestion.main()
    logger.info(f">>>>>>>> stage {STAGE_NAME} Completed >>>>>>>>>>>>>")

except Exception as e:
    logger.exception(e)
    raise e
