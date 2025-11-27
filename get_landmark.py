import cv2
import dlib
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

DATASET_DIR = "dataset/emotion"   # Đường dẫn đến thư mục dataset
MODEL_PATH = "shape_predictor_68_face_landmarks.dat" # File model của Dlib
OUTPUT_CSV = "fer2013_aam_landmarks.csv"