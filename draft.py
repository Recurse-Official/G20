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
  for i,word in enumerate (text_data{"text"})
  if word in aadhar_no & len(word)==4:
    x,y,w,h= (
      text_data["left"][i],
      text_data["top"][i],
      text_data["width"][i],
      text_data["height"][i],
    )
img =cv2.rectangle(img, (x,y) , (x+w,y+h), (0,0,0), -1)
return img
    

















  





