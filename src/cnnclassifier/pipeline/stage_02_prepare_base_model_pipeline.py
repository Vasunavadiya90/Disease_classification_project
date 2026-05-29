import os
import sys
from pathlib import Path

# Automatically resolve and change to project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))

from cnnclassifier.config.configuration import configurationManager
from cnnclassifier.components.prepare_base_model import PrepareBaseModel
from cnnclassifier import logger

STAGE_NAME = "Prepare Base Model Stage"


class PrepareBaseModelTrainingPipeline:
    """
    Pipeline for preparing the base model using transfer learning.
    
    This pipeline orchestrates the complete process of:
    1. Loading configuration
    2. Creating the base model component
    3. Executing the base model preparation
    """
    
    def __init__(self):
        """Initialize the pipeline."""
        pass

    def main(self):
        """
        Main function to execute the prepare base model pipeline.
        
        This function:
        1. Initializes the configuration manager
        2. Gets the prepare base model config
        3. Creates a PrepareBaseModel instance
        4. Executes the prepare_and_save_base_model process
        """
        config = configurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.prepare_and_save_base_model()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>> {STAGE_NAME} started <<<<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

    except Exception as e:
        logger.exception(e)
        raise e
