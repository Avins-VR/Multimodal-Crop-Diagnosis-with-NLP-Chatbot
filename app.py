import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import cv2

# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="Leaf Nutrient Deficiency Detection",
    page_icon="🌿",
    layout="wide"
)

# -----------------------------
# TITLE (CENTER)
# -----------------------------
col1, col2, col3 = st.columns([1,3,1])

with col2:
    st.markdown(
        "<h1 style='text-align:center;'>🌿 Leaf Nutrient Deficiency Detection</h1>",
        unsafe_allow_html=True
    )

# -----------------------------
# MODEL DEFINITION
# -----------------------------
class EarlyFusionModel(nn.Module):

    def __init__(self,num_tabular_features):
        super().__init__()

        self.cnn = models.efficientnet_b3(weights=None)
        self.cnn.classifier = nn.Identity()

        self.tabular_net = nn.Sequential(
            nn.Linear(num_tabular_features,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(1536+32,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,3)
        )

    def forward(self,image,tabular):

        img_feat = self.cnn(image)
        tab_feat = self.tabular_net(tabular)

        fused = torch.cat((img_feat,tab_feat),dim=1)

        return self.classifier(fused)

# -----------------------------
# LOAD MODEL
# -----------------------------
device = torch.device("cpu")

model = EarlyFusionModel(9)

model.load_state_dict(torch.load("model.pth",map_location=device))

model.eval()

class_names = ["Healthy","Early Deficiency","Critical Deficiency"]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -----------------------------
# UPLOAD AREA (CENTER)
# -----------------------------
col1, col2, col3 = st.columns([1.5,2,1.5])

with col2:
    uploaded_file = st.file_uploader(
        "Upload Leaf Image",
        type=["jpg","jpeg","png"]
    )

# -----------------------------
# PREDICTION
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    img_tensor = transform(image).unsqueeze(0)

    tabular_tensor = torch.zeros((1,9),dtype=torch.float32)

    with torch.no_grad():

        outputs = model(img_tensor,tabular_tensor)

        probs = torch.softmax(outputs,dim=1)

        confidence,predicted = torch.max(probs,1)

    label = class_names[predicted.item()]
    conf_score = confidence.item()*100

    image_np = np.array(image)
    image_cv = image_np.copy()

    hsv = cv2.cvtColor(image_cv,cv2.COLOR_RGB2HSV)

    lower_yellow = np.array([15,40,40])
    upper_yellow = np.array([40,255,255])

    mask = cv2.inRange(hsv,lower_yellow,upper_yellow)

    contours,_ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > 200:

            x,y,w,h = cv2.boundingRect(cnt)

            cv2.rectangle(
                image_cv,
                (x,y),
                (x+w,y+h),
                (255,255,0),
                2
            )

    result_img = cv2.resize(image_cv,(420,420))

    # -----------------------------
    # CENTER IMAGE
    # -----------------------------
    col1, col2, col3 = st.columns([1.5,2,1.5])

    with col2:
        st.image(result_img)

        st.markdown("### Prediction Result")

        st.success(f"Deficiency Type: {label}")

        st.info(f"Confidence Score: {conf_score:.2f}%")