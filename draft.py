import cv2
import pytesseract
import re
import numpy as np
from PIL import Image
import streamlit as st

pytessearct.pytesseract.tesseract_cmd= r"C:\Program Files\Tesseract-OCR\tesseract.exe"
def extract_aadhaar_number (text):
  aadhaar_pattern =r "\b\d{4}[- ]?\d{4}[- ]?d{4}\b"
  match = re.search(aadhaar_pattern,text)
  return match.group (0) if match else None

def mask_aadhaar_number (image,aadhaar_number):
  text_data = pytesseract.image_to_date ( img, output_type =pytesseract.output.DICT)
  for i,word in enumerate (text_data{"text"})
  if word in aadhar_no & len(word)==4:
    x,y,w,h= 
    (
      text_data["left"][i],
      text_data["top"][i],
      text_data["width"][i],
      text_data["height"][i],
    )
    img =cv2.rectangle(img, (x,y) , (x+w,y+h), (0,0,0), -1)
return img

def extract_pan_number(text):
  pan_pattern= r "\b[A-Z]{5}\d{4}[A-Z]\b"
  match= re.search(pan_pattern,text)
  return match.group (0) if match else None 

def mask_pan_number (image,pan_number):
  text_data = pytesseract.image_to_date ( img, output_type =pytesseract.output.DICT)
  for i,word in enumerate (text_data{"text"})
  if word ==pan_number:
    x,y,w,h= 
    (
      text_data["left"][i],
      text_data["top"][i],
      text_data["width"][i],
      text_data["height"][i],
    )
    img =cv2.rectangle(img , (x,y) , (x+w,y+h) , (0,0,0) , -1)
return img

def extract_dob(text):
  dob_pattern = r "\b\d{2}/\d{2}/\d{4}\b"
  match = re.search(dob_pattern,text) 
  return match. group (0) if match else None

def mask_text (image,text_to_mask):
  text_data = pytesseract.image_to_date ( img, output_type =pytesseract.output.DICT)
  for i, word in enumerate (text_data{"text"})
  if text_to_mask in word:
    x,y,w,h= 
    (
      text_data["left"][i],
      text_data["top"][i],
      text_data["width"][i],
      text_data["height"][i],
    )
    img =cv2.rectangle(img , (x,y) , (x+w,y+h) , (0,0,0) , -1)
    break
return img
   
  def extract_driving_license(text):
 driving_license_pattern = r "\b[A-Z]{2}[- ]?d{2}[- ]?\d{7,13}\b"
  match = re.search(driving_license_pattern,text) 
  return match. group (0) if match else None

  def mask_driving_license (image,driving_license):
  text_data = pytesseract.image_to_date ( img, output_type =pytesseract.output.DICT)
  for i,word in enumerate (text_data{"text"})
  if word ==driving_license:
    x,y,w,h= 
    (
      text_data["left"][i],
      text_data["top"][i],
      text_data["width"][i],
      text_data["height"][i],
    )
    img =cv2.rectangle(img , (x,y) , (x+w,y+h) , (0,0,0) , -1)
return img

      def extract_voter_id(text):
 driving_license_pattern = r "\b[A-Z]{3}\b{7}\b"
  match = re.search(driving_license_pattern,text) 
  return match. group (0) if match else None

  def mask_voter_id (image,voter_id):
  text_data = pytesseract.image_to_date ( img, output_type =pytesseract.output.DICT)
  for i,word in enumerate (text_data{"text"})
  if word ==voter_id:
    x,y,w,h= 
    (
      text_data["left"][i],
      text_data["top"][i],
      text_data["width"][i],
      text_data["height"][i],
    )
    img =cv2.rectangle(img , (x,y) , (x+w,y+h) , (0,0,0) , -1)
return img

















  





