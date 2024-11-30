import cv2
import pytesseract
import re
import numpy as np
from PIL import Image
import streamlit as st

pytessearct.pytesseract.tesseract_cmd= r"C:\Program Files\Tesseract-OCR\tesseract.exe"
def extract_aadhaar_number (text):
  aadhaar_pattern =r"\b\d{4}[- ]?\d{4}[- ]?d{4}\b"
  match = re.search(aadhaar_pattern,text)
  return match.group (0) if match else none

def mask_aadhaar_number (image,aadhaar_number):
  text_data = pytesseract.image_to_date ( img, output_type =pytesseract.output.DICT)
  
    

















  





