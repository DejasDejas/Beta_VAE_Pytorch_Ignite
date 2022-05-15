# -*- coding: utf-8 -*-
""" Configuration module to define the paths to the data and the root. """
import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
processed_path: str = os.path.join(ROOT_DIR, 'data/processed')
raw_path: str = os.path.join(ROOT_DIR, 'data/raw')


if __name__ == '__main__':
    print(ROOT_DIR)
    print(processed_path)
    print(raw_path)
