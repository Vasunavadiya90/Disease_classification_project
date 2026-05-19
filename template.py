import os
from pathlib import Path
import logging


logging.basicConfig(level = logging.INFO , format ='[%(asctime)s]: %(message)s')

project_name = 'cnnclassifier'

list_of_files = [

        "setup.py",
        "requirements.txt",
        "app.py",
        f"src/{project_name}/__init__.py",
        f"src/{project_name}/components/__init__.py",
        f"src/{project_name}/utils/__init__.py",
        f"src/{project_name}/config/__init__.py",
        f"src/{project_name}/config/configuration.py",
        f"src/{project_name}/utils/__init__.py",
        f"src/{project_name}/pipeline/__init__.py",
        f"src/{project_name}/entity/__init__.py",
        f"src/{project_name}/constants/__init__.py",

        ".github/workflows/.gitkeep",
        "config/config.yaml",
        "params.yaml",
        "research/trials.ipynb",
        "templates/index.html"

]


for filepath in list_of_files:
    file_path  = Path(filepath)
    file_dir , file_name = os.path.split(file_path)


    if file_dir!="":
        os.makedirs(file_dir , exist_ok=True)
        logging.info(f"Initiate Directory {file_dir} for the file is {file_name}")

    if (not os.path.exists(file_path)) or (os.path.getsize(file_path)==0):
        with open(file_path,'w') as f:
            pass

        logging.info(f"Creating the file : {file_path}")

    else:
        logging.info(f"{file_name} is already exists")        