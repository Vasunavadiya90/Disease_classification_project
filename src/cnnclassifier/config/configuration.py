# Import required modules
from cnnclassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from cnnclassifier.utils.common import read_yaml, create_directories
from cnnclassifier.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig ,PrepareCallbacksConfig ,TrainingConfig
from pathlib import Path
import os



class configurationManager:
    def __init__(self,
                 config_filepath = CONFIG_FILE_PATH,
                 params_file_path = PARAMS_FILE_PATH):
                 
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_file_path)

        create_directories([self.config.artifact_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
    
    def get_prepare_callbacks_config(self)->PrepareCallbacksConfig:
       
        create_directories([
            Path(os.path.dirname(self.config.prepare_callbacks.checkpoint_model_filepath)),
            Path(self.config.prepare_callbacks.tensorboard_root_log_dir)
        ])

        prepare_callback_config = PrepareCallbacksConfig(
            root_dir = Path(self.config.prepare_callbacks.root_dir),
            tensorboard_root_log_dir=Path(self.config.prepare_callbacks.tensorboard_root_log_dir),
            checkpoint_model_filepath=Path(self.config.prepare_callbacks.checkpoint_model_filepath)
        )

        return prepare_callback_config

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "Chicken-fecal-images")
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate = params.LEARNING_RATE
        )

        return training_config
    
    
