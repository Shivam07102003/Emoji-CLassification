# 🤖 Emoji Classification Project  

![Python](https://img.shields.io/badge/python-3.9+-blue)  
![License](https://img.shields.io/badge/license-MIT-green)  

Classify emoji images using a **CNN + NLP hybrid model**.  
Upload an emoji image and get the **predicted emoji label/output** instantly through a **Streamlit web app**.  

---

## 📂 Setup Instructions  

Clone the repository and set up a virtual environment:  

```bash
git clone https://github.com/your-username/emoji-classification.git
cd emoji-classification
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```  

If `requirements.txt` is missing, install manually:  

```bash
pip install streamlit tensorflow numpy pandas scikit-learn matplotlib seaborn pillow imageio
```  

---

## 🔄 How to Use  

1. Run the **Streamlit app**:  
   ```bash
   streamlit run app.py
   ```  
   - Upload an emoji image (`.png`).  
   - Get the predicted emoji name.  

2. (Optional) **Retrain the model**:  
   ```bash
   python IMG.py
   ```  

---

## 📊 Dataset  

The project uses a dataset of emojis stored as `.png` images with metadata in CSV format:  

- **emoji_classes.csv** → contains emoji names and labels.  
- **Dataset/image/** → folder with emoji images.  

After preprocessing:  
- Images are resized to `64x64`.  
- Labels are encoded and stored in `label_classes.npy`.  

---

## 📈 Key Results  

- ✅ **High classification accuracy** achieved with CNN architecture.  
- 🖼️ Works on **custom emoji dataset** with multiple classes.  
- 🌐 Provides an **interactive web interface** via Streamlit.  

Example prediction:  
Upload → 🙂  
Output → `"smiling_face_with_smile"`  

---

## 🛠️ Tech Stack  

- **Python** 🐍  
- **TensorFlow / Keras** → Model training & prediction  
- **Pandas / NumPy** → Data handling  
- **Scikit-learn** → Label encoding, preprocessing  
- **Streamlit** → User interface  
- **Matplotlib / Seaborn** → Visualization  

---

## 🤝 Contributing  

Pull requests are welcome!  
- Fork the repo  
- Create a feature branch (`git checkout -b feature-name`)  
- Commit changes and open a Pull Request  

---

## 📜 License  

This project is licensed under the **MIT License**.  

---

📌 **Author:** *Shivam Amitbhai Patel*  
