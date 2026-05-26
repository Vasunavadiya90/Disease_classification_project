from src.cnnclassifier import logger
from cnnclassifier.pipeline.stage_01_ingestion_pipeline import dataIngestionTrainingPipeline
from cnnclassifier.pipeline.stage_02_prepare_base_model_pipeline import PrepareBaseModelTrainingPipeline

logger.info("Intro to Disease classification")


import os
from pathlib import Path

STAGE_NAME_01 = "data_Ingestion_stage"
STAGE_NAME_02 = "Prepare Base Model Stage"


try:
    # Stage 1: Data Ingestion
    logger.info(f">>>>>>> stage {STAGE_NAME_01} started >>>>>>>>>>>>>>>")
    dataingestion = dataIngestionTrainingPipeline()
    dataingestion.main()
    logger.info(f">>>>>>>> stage {STAGE_NAME_01} Completed >>>>>>>>>>>>>")

except Exception as e:
    logger.exception(e)
    raise e

try:
    # Stage 2: Prepare Base Model
    logger.info(f"\n\n>>>>>>> stage {STAGE_NAME_02} started >>>>>>>>>>>>>>>")
    prepare_base_model = PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()
    logger.info(f">>>>>>>> stage {STAGE_NAME_02} Completed >>>>>>>>>>>>>")

except Exception as e:
    logger.exception(e)
    raise e
