import os
import tensorflow as tf
from pathlib import Path
from cnnclassifier import logger
from cnnclassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    """
    Prepare base model using transfer learning with VGG16.
    
    This class loads a pre-trained VGG16 model from ImageNet, removes the top layers,
    freezes the base weights, adds custom layers for the specific classification task,
    and saves the model.
    """
    
    def __init__(self, config: PrepareBaseModelConfig):
        """
        Initialize PrepareBaseModel with configuration.
        
        Args:
            config (PrepareBaseModelConfig): Configuration for base model preparation
        """
        self.config = config
        logger.info(f"PrepareBaseModel initialized with config: {config}")

    def get_base_model(self):
        """
        Load pre-trained VGG16 model from ImageNet.
        
        Returns:
            tf.keras.Model: Pre-trained VGG16 model
        """
        logger.info(f"Loading VGG16 model with weights: {self.config.params_weights}")
        
        base_model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )
        
        logger.info(f"Base model loaded successfully")
        logger.info(f"Base model summary:")
        base_model.summary()
        
        return base_model

    def prepare_full_model(self, base_model, learning_rate):
        """
        Prepare the full model by adding custom top layers to the base model.
        
        This function:
        1. Freezes the base model weights
        2. Adds custom layers for classification
        3. Compiles the model
        
        Args:
            base_model (tf.keras.Model): Pre-trained base model
            learning_rate (float): Learning rate for the optimizer
            
        Returns:
            tf.keras.Model: Complete model ready for training
        """
        logger.info(f"Preparing full model with custom top layers")
        
        # Freeze the base model weights
        base_model.trainable = False
        logger.info(f"Base model weights frozen")
        
        # Create the full model by adding custom layers
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                units=512,
                activation='relu',
                name='dense_1'
            ),
            tf.keras.layers.Dropout(rate=0.3, name='dropout_1'),
            tf.keras.layers.Dense(
                units=256,
                activation='relu',
                name='dense_2'
            ),
            tf.keras.layers.Dropout(rate=0.2, name='dropout_2'),
            tf.keras.layers.Dense(
                units=self.config.params_classes,
                activation='softmax',
                name='output_layer'
            )
        ])
        
        logger.info(f"Full model created with custom top layers")
        logger.info(f"Full model summary:")
        model.summary()
        
        # Compile the model
        logger.info(f"Compiling model with learning rate: {learning_rate}")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        logger.info(f"Model compiled successfully")
        
        return model

    def save_model(self, model, file_path):
        """
        Save the model to the specified path.
        
        Args:
            model (tf.keras.Model): Model to save
            file_path (Path): Path where to save the model
            
        Returns:
            bool: True if saved successfully
        """
        logger.info(f"Saving model to: {file_path}")
        model.save(file_path)
        logger.info(f"Model saved successfully at: {file_path}")
        return True

    def prepare_and_save_base_model(self):
        """
        Main function to prepare and save the base model.
        
        This function:
        1. Loads the VGG16 base model
        2. Saves the base model (without top layers)
        3. Prepares the full model with custom layers
        4. Saves the full model
        """
        logger.info("=" * 80)
        logger.info("Starting Base Model Preparation Process")
        logger.info("=" * 80)
        
        try:
            # Step 1: Load base model
            logger.info("\n[STEP 1] Loading base model...")
            base_model = self.get_base_model()
            
            # Step 2: Save base model
            logger.info("\n[STEP 2] Saving base model...")
            self.save_model(
                model=base_model,
                file_path=self.config.base_model_path
            )
            
            # Step 3: Prepare full model with custom layers
            logger.info("\n[STEP 3] Preparing full model with custom layers...")
            full_model = self.prepare_full_model(
                base_model=base_model,
                learning_rate=self.config.params_learning_rate
            )
            
            # Step 4: Save updated/full model
            logger.info("\n[STEP 4] Saving updated full model...")
            self.save_model(
                model=full_model,
                file_path=self.config.updated_base_model_path
            )
            
            logger.info("\n" + "=" * 80)
            logger.info("Base Model Preparation Process Completed Successfully!")
            logger.info("=" * 80)
            
            return full_model
            
        except Exception as e:
            logger.error(f"Error in prepare_and_save_base_model: {str(e)}")
            raise e
