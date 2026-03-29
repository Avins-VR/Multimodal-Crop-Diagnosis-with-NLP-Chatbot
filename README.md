# 🌿 AgriSense AI  – Multimodal AI Nutrient Deficiency Detection System

AgriSense AI  is an **AI-powered multimodal agricultural intelligence system** designed to detect **nutrient deficiencies in crops** using a combination of **leaf image analysis and soil/environmental data**.

Unlike traditional crop monitoring systems that rely only on image analysis, AgriSense AI  integrates **computer vision and environmental metadata** to make **more accurate and reliable predictions** about crop health.

The system classifies crop condition into:

- ✅ Healthy
- 🟡 Early Deficiency
- 🔴 Critical Deficiency

It also provides:

- 📊 Confidence score
- 💡 Fertilizer recommendation
- 🔄 Self-learning improvement tracking
- 🤖 NLP chatbot assistance for farmers

---

# 🚨 Problem Statement

Farmers often struggle to identify **nutrient deficiencies at the right time**. This leads to reduced crop yield and inefficient fertilizer usage.

Major challenges include:

- Manual observation of crop health is inaccurate.
- Leaf color alone cannot determine the deficiency severity.
- Soil nutrient levels (N, P, K, pH) strongly affect plant growth.
- Weather conditions influence nutrient absorption.
- Incorrect fertilizer application increases cost and reduces productivity.

Most existing AI crop systems rely **only on leaf image analysis**, which may produce incomplete or misleading predictions.

---

# 💡 Proposed Solution

AgriSense AI  introduces a **Multimodal AI-Based Crop Health Detection System**.

Instead of using only images, the system combines:

1️⃣ **Leaf Image Analysis (CNN Model)**  
2️⃣ **Soil & Weather Metadata Analysis (Machine Learning Model)**

Both types of information are fused together to generate the **final crop health prediction**.

This approach is known as **Multimodal Learning**, where multiple data sources improve the accuracy of AI predictions.

---

# 📸 1️⃣ Leaf Image Analysis (CNN Model)

The system uses a **Convolutional Neural Network (CNN)** to analyze leaf images.

The CNN detects visual patterns such as:

- Yellowing caused by nitrogen deficiency
- Leaf discoloration
- Texture changes
- Edge damage
- Stress patterns

These visual features help identify the **presence and severity of nutrient deficiencies**.

---

# 🌍 2️⃣ Soil & Weather Metadata (ML Model)

The system also analyzes environmental parameters that influence plant growth.

Metadata inputs include:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Soil pH
- Soil moisture
- Rainfall
- Temperature
- Sunlight exposure

A machine learning model processes these parameters to understand **nutrient availability and environmental stress factors**.

---

# 🧠 Multimodal Fusion Model

The outputs from the **CNN model (image features)** and the **metadata ML model** are combined together.

This fusion allows the system to make **more accurate predictions** about crop health compared to single-input models.

Final prediction categories:

- ✅ Healthy
- 🟡 Early Deficiency
- 🔴 Critical Deficiency

The system also generates a **confidence score** for the prediction.

Example Output:

```
Prediction: Early Deficiency
Confidence Score: 91%
Recommended Action: Apply Nitrogen Fertilizer
```

---

# 🔄 Self-Learning Feedback Mechanism

One of the key features of AgriSense AI  is its **self-learning feedback system**.

The system continuously improves using **real farm feedback**.

### Example Workflow

1️⃣ Farmer uploads today's leaf image  
→ System predicts: **Early Deficiency**

2️⃣ Farmer applies recommended fertilizer

3️⃣ After **10 days**, farmer uploads a new image

4️⃣ System compares:

- Previous crop condition
- Current crop condition

5️⃣ The system calculates improvement.

Example Result:

```
Crop condition improved by 70% compared to 10 days earlier.
```

6️⃣ This improvement data is stored in the system

7️⃣ The model learns from these outcomes to improve future predictions.

This allows the system to become **more accurate over time**.

---

# 🤖 NLP Chatbot for Farmers

AgriSense AI  includes an **NLP-based chatbot** that allows farmers to interact with the system in natural language.

Farmers can ask questions about crop health, soil conditions, or fertilizer usage.

### Example Interactions

**Farmer Question**

```
Which pH range should I maintain for my crop?
```

**Chatbot Response**

```
Maintain soil pH between 6.0 and 6.5 for better nutrient absorption.
```

---

**Farmer Question**

```
Why is my crop in early deficiency?
```

**Chatbot Response**

```
Nitrogen levels in your soil are below optimal range and soil moisture is low. Apply recommended fertilizer.
```

This makes the system:

- Farmer-friendly
- Easy to use
- Interactive and informative

---

# 📊 System Outputs

The system provides the following outputs:

| Output | Description |
|------|-------------|
| Crop Health Status | Healthy / Early Deficiency / Critical Deficiency |
| Confidence Score | Prediction confidence percentage |
| Fertilizer Recommendation | Suggested fertilizer action |
| Improvement Score | Crop improvement percentage |
| Chatbot Guidance | Answers to farmer queries |

---

# ⚙️ System Workflow

1️⃣ Farmer uploads crop leaf image  

2️⃣ Environmental metadata is provided:

- NPK values
- Soil pH
- Soil moisture
- Weather conditions

3️⃣ CNN model analyzes leaf image

4️⃣ ML model analyzes environmental data

5️⃣ Multimodal fusion model generates final prediction

6️⃣ System outputs:

- Deficiency level
- Confidence score
- Fertilizer recommendation

7️⃣ Farmer can upload future images for **improvement analysis**

---

# 🧪 Technologies Used

### Artificial Intelligence

- Python
- TensorFlow / PyTorch
- Scikit-Learn
- OpenCV

### Computer Vision

- Convolutional Neural Networks (CNN)
- Image preprocessing
- Feature extraction

### Backend

- Flask / FastAPI

### Frontend

- React / Streamlit dashboard

### NLP Chatbot

- Natural Language Processing
- Transformer-based models or rule-based chatbot

---

# 📂 Project Structure

```
AI-Crop-Sense
│
├── dataset
│   ├── leaf_images
│   ├── soil_weather_data
│
├── models
│   ├── cnn_leaf_model
│   ├── metadata_model
│   ├── multimodal_fusion_model
│
├── chatbot
│   ├── nlp_chatbot.py
│
├── backend
│   ├── api
│   ├── server.py
│
├── frontend
│   ├── dashboard
│   ├── components
│
├── utils
│
├── README.md
```

# 🌟 Key Advantages

- Multimodal AI approach (image + metadata)
- Early detection of nutrient deficiencies
- Accurate crop health classification
- Self-learning improvement tracking
- Smart fertilizer recommendation
- NLP chatbot for farmer interaction
- Data-driven precision agriculture support

---

# 👨‍💻 Author

**Avins V R**

Artificial Intelligence & Data Science  
St. Joseph's Institute of Technology, Chennai

Interests:

- Artificial Intelligence
- Smart Agriculture
- Computer Vision
- Full Stack Development
