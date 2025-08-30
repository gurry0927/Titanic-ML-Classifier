# 🚢 Titanic Survival Prediction with Machine Learning

[![Streamlit App](https://img.shields.io/badge/Streamlit-Demo-brightgreen?logo=streamlit)](https://YOUR-STREAMLIT-APP-LINK)  
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-blue?logo=kaggle)](https://www.kaggle.com/YOUR-KAGGLE-USERNAME)  

---

## 📖 專案簡介 (中文)
本專案使用 Python 與機器學習模型（Random Forest、SVC）對 **鐵達尼號乘客生存率** 進行預測。  
專案內容包含 **資料清理、特徵工程、模型訓練、交叉驗證、模型比較**，並提供互動式網頁介面 (Streamlit)。  

🔹 **功能特色**
- 資料清理與缺失值處理  
- 探索性資料分析 (EDA)  
- 模型訓練與交叉驗證  
- 展示 **特徵重要性分析與可視化** (隨機森林、決策樹)
- 支援 **自訂乘客資料** 即時預測生存機率 
- **下載模型並提交結果至 Kaggle** 

 

👉 [點此體驗 Demo (Streamlit App)](https://titanic-ml-classifier-kvrfcmvzuaqu6dtpxzneb8.streamlit.app/)

---

## 📖 Project Description (English)
This project applies **Machine Learning models (Random Forest, SVC)** to predict the **survival of Titanic passengers**.  
It includes **data preprocessing, feature engineering, model training, cross-validation, and performance comparison**,  
with an interactive **Streamlit web interface** for real-time predictions.  

🔹 **Key Features**
- Real-time survival prediction with **custom passenger input**  
- **Data cleaning pipeline** and **exploratory data analysis** visualizations  
- **Feature importance & interpretability** (Random Forest, Decision Tree)  
- Ability to **download trained models** and **submit predictions to Kaggle**  

👉 [Try the Demo on Streamlit](https://titanic-ml-classifier-kvrfcmvzuaqu6dtpxzneb8.streamlit.app/)

---

## 📊 Demo Screenshots

### 🔹 模型訓練與參數選擇
<div align="center">
  <img src="images/demo01.gif" width="600">
</div>

使用者可調整 **模型類型 (Random Forest, SVC)** 與超參數，並立即看到 **訓練過程與評估結果**。  
輸出包含：準確率、混淆矩陣、ROC 曲線等指標，方便比較不同模型表現。  

---

### 🔹 自訂乘客資料即時預測
<div align="center">
  <img src="images/demo02.gif" width="600">
</div>

使用者可輸入 **自訂乘客屬性（性別、年齡、艙等）**，系統即時回傳預測結果。  
輸出包含：生存機率百分比，並可下載模型或產生 **Kaggle 提交檔**。  


---

## 📊 專案流程圖
資料處理與模型訓練的主要步驟：
<div align="center">
  <img src="images/data_cleaning.png" width="600">
</div>

---

## 🛠️ Tech Stack
- **Python**  
- **Scikit-learn**  
- **Imbalanced-learn** (SMOTE 處理不平衡)  
- **Streamlit**  
- **Pandas / NumPy**  
- **Matplotlib / Seaborn**  
- **Joblib** (模型保存與載入)
  
---

## 🚀 使用方式 (Usage)

```bash
# 1. 下載專案
git clone https://github.com/gurry0927/Titanic-ML-Classifier.git
cd titanic-ml-classifier

# 2. 安裝需求
pip install -r requirements.txt

# 3. 執行 Streamlit 專案
streamlit run app.py
