import os
from pathlib import Path
from box.exceptions import BoxValueError
from cnnclassifier import logger
import json
import yaml
import joblib
from ensure import ensure_annotations
from box import config_box
from pathlib import Path
from typing import Any
import base64

@ensure_annotations  # "Validate function inputs and outputs using type hints."
def read_yaml(path_to_yaml:Path) -> config_box:
    """reads the yaml file and returns 
    
    Args:
        path_to_yaml(str):input as path

    Raises:
        ValueError: if yaml file is empty
        e : empty file


    returns:
        configBox:Configuration Box

    """

    try:

        with open(path_to_yaml) as yaml_file:
            content = yaml.load(yaml_file)
            logger.info(f"yaml file : {path_to_yaml} loaded successfully")
            return config_box(content)
    except BoxValueError:
        raise ValueError("Yaml file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories:list , verbose = True):
    '''
    create list of directories

    Args:
    path_to_directories (list): list of path of directories
    ignore_log (bool,optional) : ignore if multiple dirs is to be created. Default to False.
    
    '''

    for path in path_to_directories:
        os.makedirs(path , exist_ok=True)

        if verbose:
            logger.info(f"Created Directory at :{path}")

@ensure_annotations
def save_json(path:Path , data:dict):
    '''
    save to json


    Args:
    path = path to json file
    data = data to be saved in json file
    
    '''

    with open(path , 'w') as f:
        json.dump(data , f, indent=4)

    logger.info(f"json file saved at {path}")

@ensure_annotations
def load_json(path:Path):
    '''
    jaon load


    Args:
    PATH = PATH to Json file


    Returns:
        configBox:data as class attributes instead of dict
    '''

    with open(path) as f:
        content = json.load(f)

        logger.info(f"json file loaded succesfully loaded at {path}")

        return config_box(content)
    

@ensure_annotations
def save_bin(data :Any , path:Path):
    '''
    save binary file


    Args:
        data(Any): data to be saved as binary
        path(Path): path to binary file
    '''

    joblib.dump(value= data ,filename=path)
    logger.info(f"binary file saved at {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"



def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
