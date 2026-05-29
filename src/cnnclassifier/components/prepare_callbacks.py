import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from cnnclassifier.config.configuration import PrepareCallbacksConfig





class PrepareCallBack:

    def __init__(self,config:PrepareCallbacksConfig):

        self.config = config

    @property
    def _create_tb_callbacks(self):
        timestemp = time.strftime("%Y-%m-%d-%H-%H-%S")

        tb_running_log_dir = os.path.join(self.config.tensorboard_root_log_dir,
                                          f"tb_log_dir {timestemp}")
        return tf.keras.callbacks.TensorBoard(log_dir = tb_running_log_dir)

    @property
    def _create_ckpt_callbacks(self):
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config.checkpoint_model_filepath,
            save_best_only=True)

    @property
    def _create_es_callbacks(self):
        return tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True)

    @property
    def _create_lr_callbacks(self):
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6)
    
    def get_tb_ckpt_callbacks(self):
        return [
            self._create_tb_callbacks,
            self._create_ckpt_callbacks,
            self._create_es_callbacks,
            self._create_lr_callbacks
        ]
    
    


