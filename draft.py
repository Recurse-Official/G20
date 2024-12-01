import cv2
import pytesseract
import re
import numpy as np
from PIL import Image
import streamlit as st

st.markdown(
  """
  <style>
  .stApp{
    background-color: #1f2b2b;
    color: #E5E5E5;
  }
  
    html, body, [class*="css"] {
    color: #E5E5E5 !important;
    }
    
    header[data-testid="stHeader"] {
      background-color: #1f2b2b;
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0.5rem 1rem;
    }
    
    .nav-buttons {
      display: flex;
      gap: 1rem;
     } 
     
    .nav-buttons a {
        text-decoration: none;
        color: #1f2b2b;
        background-color: #E5E5E5;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weigth: bold;
        transition: background-color 0.3s,color 0.3s;
      }
      
      .nav-buttons a:hover {
          background-color: #1f2b2b;
          color: #e5E5E5;
        }
        
       </style>
       <header data-testid="stHeader">
         <div class="nav-buttons">
           <a href="/?nav=home">Home</a>
           <a href="/?nav=about">About Us</a>
          </div>
        </header>
  """
)
pytesseract.pytesseract.tesseract_cmd= r"C:\Program Files\Tesseract-OCR\tesseract.exe"
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+ "haarcasade_frontalface_default.xml")
def extract_aadhaar_number (text):
  aadhaar_pattern =r"\b\d{4}[- ]?\d{4}[- ]?\d{4}\b"
  match = re.search(aadhaar_pattern,text)
  return match.group (0) if match else None
 
def extract_pan_number(text):
  pan_pattern= r"\b[A-Z]{5}\d{4}[A-Z]\b"
  match= re.search(pan_pattern,text)
  return match.group (0) if match else None 

def extract_voter_id(text):
  voter_id_pattern = r"\b[A-Z]{3}\d{7}\b"
  match = re.search(voter_id_pattern,text) 
  return match.group (0) if match else None

def extract_driving_license(text):
  driving_license_pattern = r"\b[A-Z]{2}[- ]?\d{2}[- ]?\d{7,13}\b"
  match = re.search(driving_license_pattern,text) 
  return match.group (0) if match else None

def extract_dob(text):
  dob_pattern = r"\b\d{2}/\d{2}/\d{4}\b"
  match = re.search(dob_pattern,text) 
  return match.group (0) if match else None

def extract_address(text):
  lines= text.split('\n')
  address_block=[]
  address_started=False
  for line in lines:
    if "Address" in line or "पता" in line:
      address_started=True
      if address_started:
        address_block.append(line.strip()) 
        if len(address_block) >=4:
          break

  address = " ".join(address_block).replace("Address","").replace("पता","").strip()
  return address if address else None
    
def mask_aadhaar_number (img,aadhaar_number):
  text_data = pytesseract.image_to_data(img, output_type =pytesseract.output.DICT)
  for i,word in enumerate (text_data["text"]):
    if word in aadhaar_number and len(word)==4:
      x,y,w,h= (
        text_data["left"][i],
        text_data["top"][i],
        text_data["width"][i],
        text_data["height"][i],
      )
    img =cv2.rectangle(img, (x, y) , (x + w, y + h ), (0, 0, 0), -1)
  return img

def mask_pan_number (img,pan_number):
  text_data = pytesseract.image_to_data(img, output_type =pytesseract.output.DICT)
  for i,word in enumerate (text_data["text"]):
    if word ==pan_number:
      x,y,w,h= (
        text_data["left"][i],
        text_data["top"][i],
        text_data["width"][i],
        text_data["height"][i],
      )
    img =cv2.rectangle(img , (x, y) , (x + w, y + h) , (0, 0, 0) , -1)
  return img

def mask_voter_id (img,voter_id):
  text_data = pytesseract.image_to_data(img, output_type =pytesseract.output.DICT)
  for i,word in enumerate (text_data["text"]):
    if word ==voter_id:
      x,y,w,h= (
        text_data["left"][i],
        text_data["top"][i],
        text_data["width"][i],
        text_data["height"][i],
      )
    img =cv2.rectangle(img , (x,y) , (x+w,y+h) , (0,0,0) , -1)
  return img

def mask_driving_license (img,driving_license):
  text_data = pytesseract.image_to_data(img, output_type =pytesseract.output.DICT)
  for i , word in enumerate (text_data["text"]):
    if word ==driving_license:
      x,y,w,h = (
        text_data["left"][i],
        text_data["top"][i],
        text_data["width"][i],
        text_data["height"][i],
    )
    img =cv2.rectangle(img , (x,y) , (x+w,y+h) , (0,0,0) , -1)
  return img

def mask_text (img,text_to_mask):
  text_data = pytesseract.image_to_data( img, output_type =pytesseract.output.DICT)
  for i, word in enumerate (text_data["text"]):
    if text_to_mask in word:
      x,y,w,h = (
        text_data["left"][i],
        text_data["top"][i],
        text_data["width"][i],
        text_data["height"][i],
      )
      img =cv2.rectangle(img , (x,y) , (x+w,y+h) , (0,0,0) , -1)
  return img
   
def mask_address (img,address):
  text_data = pytesseract.image_to_data(img, output_type =pytesseract.output.DICT)
  coords=[]
  for i , word in enumerate (text_data["text"]):
      if word.strip() in address and len(word.strip()) > 1:
        x, y, w, h = (
          text_data["left"][i],
          text_data["top"][i],
          text_data["width"][i],
          text_data["height"][i],
        )
  coords.append((x , y , x + w , y + h))
  if not coords:
    return img
  coords=np.array(coords)
  img=cv2.rectangle(img, (coords[:, 0].min(), coords[:, 1].min()),
                     (coords[:, 2].max(), coords[:, 3].max()), (0, 0, 0), -1)
  return img

def blur_faces(img):
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray_img, scalefactor=1.1, minNeighbors=5, minSize=(30, 30))
  for (x, y, w, h) in faces:
    face=img[y:y+h, x:x+w]
    blurred_face=cv2.GaussianBlur(face, (51,51), 30)
    img[y:y+h,x:x+w]=blurred_face
  return img
 
st.title("Document Details Extractor & Masker")
st.write("Upload an image to extract Aadhaar, PAN, Voter ID, Driving License numbers, DOB details, and optionally mask them along with face blurring.")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    #img_array = cv2.erode(img_array, kernel=np.ones((2,1)), iterations=1)
    img_rgb = img_array.copy()

    extracted_text = pytesseract.image_to_string(img_array)
  
    aadhaar_number = extract_aadhaar_number(extracted_text)
    pan_number = extract_pan_number(extracted_text)
    voter_id = extract_voter_id(extracted_text)
    driving_license = extract_driving_license(extracted_text)
    dob = extract_dob(extracted_text)
    address=extract_address(extracted_text)
  
    st.write("Extracted Information: ")

    extracted_info = {}
    if aadhaar_number:
        st.success(f"Aadhaar Number: {aadhaar_number}")
        extracted_info["Aadhaar Number"] = aadhaar_number
    if pan_number:
        st.success(f"PAN Number: {pan_number}")
        extracted_info["PAN Number"] = pan_number
    if voter_id:
        st.success(f"Voter ID: {voter_id}")
        extracted_info["Voter ID"] = voter_id
    if driving_license:
        st.success(f"Driving License: {driving_license}")
        extracted_info["Driving License"] = driving_license
    if dob:
        st.success(f"Date of Birth: {dob}")
        extracted_info["Date of Birth"] = dob
    if address:
        st.success(f"Address: {address}")
        extracted_info["Address"] = address
      
    extracted_info["Blur Faces"] = "Yes"

    if extracted_info:
        
        selected_pii = st.multiselect(
            "Select the PII to Mask or Blur",
            options=extracted_info.keys(),
            default=list(extracted_info.keys())
        )
      
         if st.button("Apply Masking"):
           for pii_type in selected_pii:
                    if pii_type == "Aadhaar Number":
                        img_rgb = mask_aadhaar_number(img_rgb, extracted_info[pii_type])
                    elif pii_type == "PAN Number":
                        img_rgb = mask_pan_number(img_rgb, extracted_info[pii_type])
                    elif pii_type == "Voter ID":
                        img_rgb = mask_voter_id(img_rgb, extracted_info[pii_type])
                    elif pii_type == "Driving License":
                        img_rgb = mask_driving_license(img_rgb, extracted_info[pii_type])
                    elif pii_type == "Date of Birth":
                        img_rgb = mask_text(img_rgb, extracted_info[pii_type])
                    elif pii_type == "Address":
                        img_rgb = mask_address(img_rgb, extracted_info[pii_type])
                    elif pii_type=="Blur Faces":
                        img_rgb = blur_faces(img_rgb)

                masked_pil = Image.fromarray(img_rgb)
                st.image(masked_pil, caption="Masked and Blurred Image", use_container_width = True)

                masked_pil.save("masked_image.png")
                with open("masked_image.png", "rb") as file:
                    st.download_button(
                        label="Download Masked and Blurred Image",
                        data=file,
                        file_name="masked_image.png",
                        mime="image/png",
                    )
    

  





