# 👤 Beyond Hello: Predicting Personality Type

This project aims to predict whether an individual is an introvert or an extrovert based on their behavioral and social data. It includes data analysis, preprocessing, model training, and evaluation using machine learning techniques.

---

## 📌 Project Overview

Understanding personality traits like introversion and extroversion has applications in psychology, sociology, marketing, and personalized recommendations. This project utilizes a dataset containing features related to daily habits and social interactions to build models that can classify individuals into these two personality types.

---

## 🧠 Features & Techniques

- ✅ Data Loading and Exploration
- ✅ Data Cleaning and Preprocessing (handling missing values, encoding categorical features, scaling)
- ✅ Exploratory Data Analysis (visualizations of distributions and relationships)
- ✅ Model Training (Logistic Regression, Random Forest, Support Vector Machine)
- ✅ Model Evaluation (accuracy, classification report, confusion matrix)
- ✅ Hyperparameter Tuning (GridSearchCV for SVM)
- ✅ Feature Importance Analysis (Random Forest)

---

## 📂 Dataset

The project uses the "Extrovert vs. Introvert Behavior Data" dataset, which includes features like:

- `Time_spent_Alone`: Hours spent alone daily (0–11).
- `Stage_fear`: Presence of stage fright (Yes/No).
- `Social_event_attendance`: Frequency of social events (0–10).
- `Going_outside`: Frequency of going outside (0–7).
- `Drained_after_socializing`: Feeling drained after socializing (Yes/No).
- `Friends_circle_size`: Number of close friends (0–15).
- `Post_frequency`: Social media post frequency (0–10).
- `Personality`: Target variable (Extrovert/Introvert).

The dataset contains 2,900 rows and 8 columns and includes some missing values.

---

## 🛠️ Tech Stack

- **Language**: Python
- **Libraries**:
    - `pandas`, `numpy`
    - `matplotlib`, `seaborn`
    - `scikit-learn`

---

## 📊 Exploratory Data Analysis

- Visualizations of the distribution of personality types.
- Histograms of numerical features.
- Boxplots showing the relationship between numerical features and personality.
- Count plots showing the relationship between categorical features and personality.

---

## 📈 Models Implemented

| Model                 | Highlights                                  | Test Accuracy |
|-----------------------|---------------------------------------------|---------------|
| Logistic Regression   | Linear model for binary classification      | ~0.9172       |
| Random Forest         | Ensemble of decision trees                  | ~0.9011       |
| Support Vector Machine| Effective in high dimensional spaces          | ~0.9195       |
| Tuned SVM             | SVM with optimized hyperparameters          | ~0.9195       |

---

## 🚀 Getting Started

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Souvik-Rana/Beyond-Hello-Predicting-Personality-Type.git
    cd Beyond-Hello-Predicting-Personality-Type
    ```

2.  **Install dependencies**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```

3.  **Run the analysis**
    Execute the Python script or Jupyter Notebook containing the analysis.
---

## 🧪 Evaluation Metrics

* 📈 Accuracy
* 📊 Classification Report (precision, recall, F1-score)
* 📉 Confusion Matrix

---

## ✅ Key Findings

- The Support Vector Machine (SVM) model achieved the highest test accuracy.
- Features like "Drained\_after\_socializing" and "Stage\_fear" appear to be strong predictors of personality type based on the Random Forest's feature importance.

---

## ✅ Future Enhancements

* Explore other classification models.
* Perform more extensive hyperparameter tuning.
* Investigate potential feature engineering opportunities.

---

## 👤 Author

<p align="center">
  <b> SOUVIK RANA </b><br>
  <br><a href="https://github.com/souvikrana17">
    <img src="https://img.shields.io/badge/GitHub-000?style=for-the-badge&logo=github&logoColor=white" style="margin-right: 10px;" />
  </a>
  <a href="https://www.linkedin.com/in/souvikrana17/">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" style="margin-right: 10px;" />
  </a>
  <a href="https://www.kaggle.com/souvikrana17">
    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" style="margin-right: 10px;" />
  </a>
  <a href="https://souvikrana17.vercel.app">
    <img src="https://img.shields.io/badge/Portfolio-FF5722?style=for-the-badge&logo=Firefox&logoColor=white" />
  </a>
</p>


<p align="center">
  <img src="https://raw.githubusercontent.com/souvikrana17/souvikrana17/main/SOUVIK%20RANA%20BANNER.jpg" alt="Banner" width="100%" />
</p>

---

## 🤝 Contributions

Contributions, issues, and feature requests are welcome. Feel free to fork the repository and submit a pull request!
