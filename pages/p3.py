import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from datetime import datetime

st.title('Titanic 模型預測')

train = pd.read_csv('datas/train_processed.csv')
test = pd.read_csv('datas/test_cleaned.csv')
answer = pd.read_csv('datas/gender_submission.csv')

# 載入模型
@st.cache_resource
def load_models():
    RF = joblib.load('models/RF_0.85_0.88_STRATIFY.joblib')
    SVC = joblib.load('models/svc_0.99.joblib')
    return RF, SVC

RF, SVC = load_models()

# 初始化 session state
if 'kaggle_results' not in st.session_state:
    st.session_state.kaggle_results = None
if 'kaggle_clicked' not in st.session_state:
    st.session_state.kaggle_clicked = False
if 'custom_predictions' not in st.session_state:
    st.session_state.custom_predictions = pd.DataFrame(columns=[
        '預測時間', '模型', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', '預測結果'
    ])
if 'prediction_count' not in st.session_state:
    st.session_state.prediction_count = 0

tab1 ,tab2 = st.tabs(["🎭 自訂角色預測", "🏆 Kaggle 測試集結果"])

# 自訂角色預測區域
with tab1:
    st.header('⚙️ 輸入參數自訂角色')
    st.info("請調整下列參數，模擬乘客資訊，並點擊下方按鈕進行預測")
    columns = st.columns(2)
    
    with columns[0]:
        model = st.selectbox('選擇模型', ['Random Forest', 'SVC'])
        se1 = st.selectbox('座艙等級(Pclass)', [1, 2, 3])
        se2 = st.selectbox('性別(Sex)', ['male', 'female'])
        se3 = st.selectbox('登船港口(Embarked)', ['C', 'Q', 'S'])
        embarked_mapping = {"C": 0, "Q": 1, "S": 2}
        embarked_num = embarked_mapping[se3]
        
        # 按鈕區域
        col1, col2 = st.columns([1, 1])
        with col1:
            predict = st.button('進行預測', type='primary', key='predict')
        with col2:
            clear_results = st.button('清除結果', key='clear')

    with columns[1]:
        se4 = st.slider('年齡(Age)', min_value=1, max_value=100, 
                        value=30, step=1, key='age_select')
        se5 = st.slider('兄弟姐妹/配偶數(Sibsp)', min_value=0, max_value=10, 
                        value=0, step=1, key='sibsp_select')
        se6 = st.slider('父母/子女數(Parch)', min_value=0, max_value=10, 
                        value=0, step=1, key='parch_select')
        se7 = st.slider('票價(Fare)', min_value=0.0, max_value=520.0, 
                        value=35.5, step=0.05, key='fare_select')

    # 清除自訂預測結果
    if clear_results:
        st.session_state.custom_predictions = pd.DataFrame(columns=[
            '預測時間', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', '模型', '預測結果'
        ])
        st.session_state.prediction_count = 0
        st.success("✅ 自訂角色預測結果已清除")

    # 自訂角色預測邏輯
    input_df = pd.DataFrame([{
        "Pclass": se1,
        "Sex": 0 if se2 == "male" else 1,
        "Age": se4,
        "SibSp": se5,
        "Parch": se6,
        "Fare": se7,
        "Embarked": embarked_num
    }])

    labels = ['🔴 Dead', '🟢 Alive']

    if predict:
        # 執行預測
        if model == 'Random Forest':
            pred = RF.predict(input_df)
        elif model == 'SVC':
            pred = SVC.predict(input_df)
        
        pred_label = labels[pred[0]]
        
        # 顯示預測結果
        if pred_label == "🟢 Alive":
            st.success("### 🟢 Passenger Survived!")
        else:
            st.error("### 🔴 Passenger Did Not Survive")
        
        # 儲存預測結果到歷史記錄
        st.session_state.prediction_count += 1
        new_prediction = pd.DataFrame([{
        
            '預測時間': datetime.now().strftime("%H:%M:%S"),
            '預測結果': pred_label,
            '模型': model,
            'Pclass': se1,
            'Sex': se2,
            'Age': se4,
            'SibSp': se5,
            'Parch': se6,
            'Fare': se7,
            'Embarked': se3
            
        }])
        
        st.session_state.custom_predictions = pd.concat([
            st.session_state.custom_predictions, 
            new_prediction
        ], ignore_index=True)

    # 顯示自訂角色預測歷史
    if not st.session_state.custom_predictions.empty:
        st.subheader("📋 自訂角色預測歷史")
        st.info(f"總共進行了 {st.session_state.prediction_count} 次預測")
        
        # 使用可編輯的dataframe顯示
        edited_df = st.dataframe(
            st.session_state.custom_predictions,
            use_container_width=True,
        )
        
        # 提供下載功能
        csv = st.session_state.custom_predictions.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 下載預測歷史 CSV",
            data=csv,
            file_name=f'titanic_custom_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv'
        )

with tab2:
    # Kaggle預測按鈕
    cols1, cols2, cols3 = st.columns([1, 2, 1])

    with cols1:
        st.markdown("<div style='display:flex; align-items:center; height:100px; font-size:20px; font-weight:bold; justify-content:center;'> 選擇模型👉 </div>", unsafe_allow_html=True)

    with cols2:
        model = st.selectbox(' ', ['Random Forest', 'SVC'], key='model2')

    with cols3:
        st.markdown("<div style='display:flex; align-items:center; height:28px;'>", unsafe_allow_html=True)
        start = st.button('Kaggle 測試集預測', type='primary')
        st.markdown("</div>", unsafe_allow_html=True)
        
    
    if start:
        st.session_state.kaggle_clicked = True
        
        # 執行Kaggle預測
        if model == 'Random Forest':
            y_pred = RF.predict(test)
        elif model == 'SVC':
            y_pred = SVC.predict(test)
        
        # 儲存結果
        st.session_state.kaggle_results = {
            'model': model,
            'predictions': y_pred,
            'result_df': pd.DataFrame({
                'PassengerId': answer['PassengerId'],
                'Survived_Predict': y_pred,
            })
        }

    # 顯示Kaggle預測成功訊息
    if st.session_state.kaggle_clicked and st.session_state.kaggle_results:
        st.success("📊 Kaggle測試集預測結果已生成，請向下滾動查看詳細資訊")
        # 顯示Kaggle預測結果
        if st.session_state.kaggle_results:
            st.subheader("🏆 Kaggle 測試集預測結果")
            st.info(f"使用模型: {st.session_state.kaggle_results['model']}")
            
            cols = st.columns(2)
            with cols[0]:
                st.dataframe(st.session_state.kaggle_results['result_df'], use_container_width=True)
                

        
        with cols[1]:
            if st.session_state.kaggle_results['model'] == 'Random Forest':
                st.markdown("""
                            <div style='text-align: center; color: #FFF; font-size: 0.8rem;'>
                            <p>
                            本模型預測結果已提交至Kaggle Titanic 競賽<br>
                            點擊下方Logo連結查看詳細Leaderboard分數
                            </p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                            )
                st.markdown(
                    '''
                    <div style="text-align: center;">
                        <a href="https://www.kaggle.com/gurry0927">
                            <img src="https://www.kaggle.com/static/images/site-logo.png" width="150">
                        </a>
                    </div>
                    ''',
                    unsafe_allow_html=True
                    )
                st.image('images/0.77272.png')
            else:
                st.markdown("""
                            <div style='text-align: center; color: #FFF; font-size: 0.8rem;'>
                            <p>
                            本模型預測結果已提交至Kaggle Titanic 競賽<br>
                            點擊下方Logo連結查看詳細Leaderboard分數
                            </p>
                            </div>
                            """, 
                            unsafe_allow_html=True)
                st.markdown(
                    '''
                    <div style="text-align: center;">
                        <a href="https://www.kaggle.com/gurry0927">
                            <img src="https://www.kaggle.com/static/images/site-logo.png" width="150">
                        </a>
                    </div>
                    ''',
                    unsafe_allow_html=True
                    )
                st.image('images/0.77990.png')

            # 提供Kaggle結果下載
            kaggle_csv = st.session_state.kaggle_results['result_df'].to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 下載Kaggle預測結果",
                data=kaggle_csv,
                file_name=f'kaggle_submission_{st.session_state.kaggle_results["model"].lower().replace(" ", "_")}.csv',
                mime='text/csv'
                )

    # 側邊欄狀態顯示
    #st.sidebar.markdown("---")
    st.sidebar.subheader("📊 當前狀態")
    if st.session_state.kaggle_results:
        st.sidebar.success(f"✅ Kaggle預測完成 ({st.session_state.kaggle_results['model']})")
    if st.session_state.prediction_count > 0:
        st.sidebar.info(f"🎭 自訂預測: {st.session_state.prediction_count} 次")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
    <p> Tech-Stack: Python • Scikit-learn • Streamlit • Pandas • Matplotlib • Github • Kaggle </p>
    </div>
    """, 
    unsafe_allow_html=True
)
