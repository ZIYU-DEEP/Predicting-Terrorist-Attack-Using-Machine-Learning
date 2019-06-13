#!/bin/bash

py -m venv env
.\env\Scripts\activate

pip install --user -r requirements.txt


cd ./codes
python clean.py
python featureEngineering.py --ask_user 0 --verbose 1
python train.py --clean_start 1 --ask_user 0 --verbose 1 --plot 1

deactivate
