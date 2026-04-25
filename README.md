# 🧠 Fake News Detection using NLP | NeuroLogic '26 Submission

## 📌 Overview

This project was developed for **NeuroLogic '26 – Global NLP Datathon** under **Challenge 2: Fake News & Misinformation Detection**.

The goal is to build a machine learning model that analyzes news article titles and content, and predicts whether the news is **True** or **False**.

---

## 🎯 Problem Statement

Develop a robust NLP classification model that processes the title and content of news articles and predicts whether the information is reliable (True) or misleading (False).

**Evaluation Metric:** Accuracy

---

## 📂 Dataset Used

### Files Provided:

* `with_label.csv` → Training dataset with labels
* `no_label.csv` → Evaluation dataset without labels

### Input Features:

* `title`
* `text`

### Target:

* `label` (`True`, `False`)

---

## ⚙️ Approach & Methodology

### 1. Data Preprocessing

* Removed null values
* Removed duplicate rows
* Combined title + text
* Converted text to lowercase
* Removed special characters
* Removed extra spaces

---

### 2. Feature Extraction

Used **TF-IDF Vectorization**

* `max_features = 10000`
* `ngram_range = (1,2)`

---

### 3. Model Used

**Linear SVC (Linear Support Vector Classifier)**

Chosen because it performs strongly on high-dimensional sparse text data (like TF-IDF) and provides fast and accurate classification.

---

## 📊 Model Performance

### Validation Method:

Train-Test Split (80% Train / 20% Test)

### Accuracy:

**99.55%**

---

## 🚀 How to Run the Project

1. Clone the repository
2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```
3. Train the model:

   ```
   python train_model.py
   ```
4. Run the application:

   ```
   python app.py
   ```

---

## 🧪 Sample Prediction

### Input:

Title: "Breaking news: Major discovery shocks scientists"
Text: "Detailed article content..."

### Output:

Prediction: ❌ False

---

## 📁 Project Structure

```
fakenews/
│── app.py
│── train_model.py
│── model.pkl
│── vectorizer.pkl
│── with_label.csv
│── no_label.csv
│── README.md
```

---

## ⚠️ Challenges Faced

* Handling noisy and unstructured text data
* Distinguishing subtle differences between real and fake news
* Avoiding overfitting due to high-dimensional features

---

## 🔮 Future Improvements

* Use advanced models like **BERT / Transformers**
* Deploy as a real-time web application
* Improve generalization with larger datasets

---

## 🏁 Conclusion

This project demonstrates how NLP and machine learning can be effectively used to detect fake news. The model shows strong performance and provides a scalable solution for misinformation detection.
