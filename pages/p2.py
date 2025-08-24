import io, joblib
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, auc, roc_curve, roc_auc_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 設定頁面配置
st.set_page_config(
    page_title="Titanic ML 模型調參平台",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化 session_state
default_keys = ["smote_toggle", "pca_toggle", "standard_toggle", "model_trained"]
for key in default_keys:
    if key not in st.session_state:
        st.session_state[key] = False

# 快取資料載入
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('datas/train_processed.csv')
        test = pd.read_csv('datas/test_cleaned.csv')
        answer = pd.read_csv('datas/gender_submission.csv')
        return df, test, answer
    except FileNotFoundError:
        st.error("❌ 找不到資料檔案，請確認檔案路徑是否正確")
        st.stop()

# 載入資料
df, test, answer = load_data()
X = df.drop(['Survived'], axis=1)
y = df['Survived']

# 定義模型訓練函數
def fit_model(TEST_SIZE, STRATIFY_mode, RANDOM_STATE, 
              SMOTE_mode, PCA_mode, STANDARD_MODE, 
              select_model, cv, SCORING="roc_auc"):
    
    # 分割資料集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE,
        stratify=STRATIFY_mode, 
        random_state=RANDOM_STATE
    )

    # 設定模型參數
    models_param_grid = {
        "Random Forest": {
            "model": RandomForestClassifier(random_state=RANDOM_STATE),
            "params": {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 3, 5],
                "clf__min_samples_split": [2, 5]
            }
        },
        "SVC": {
            "model": SVC(probability=True),
            "params": [
                {"clf__C": [0.5, 1, 1.5, 2],
                 "clf__gamma": [0.01, 0.1, 0.15], 
                 "clf__kernel": ["rbf"]}
            ]
        }
    }

    # 建立pipeline
    steps = []
    if SMOTE_mode:
        steps.append(("smote", SMOTE(random_state=42)))
    if STANDARD_MODE:
        steps.append(("scaler", StandardScaler()))
    if PCA_mode:
        pca = PCA(n_components=min(2, X.shape[1]))
        steps.append(("pca", pca))
    steps.append(("clf", models_param_grid[select_model]["model"]))

    pipe = Pipeline(steps)

    # GridSearchCV
    param_grid = models_param_grid[select_model]["params"]
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv_splitter,
        scoring=SCORING,
        n_jobs=-1
    )
    gs.fit(X_train, y_train)

    # 訓練決策樹用於可視化
    clf2 = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
    clf2.fit(X_train, y_train)
    
    # 預測
    y_pred = gs.predict(X_test)
    y_proba = gs.predict_proba(X_test)[:, 1]
    
    # 評估
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc_ = roc_auc_score(y_test, y_proba)
    
    # Kaggle預測
    y_test_pred = gs.predict(test)
    submission = pd.DataFrame({
        'PassengerId': answer['PassengerId'],
        'Survived': y_test_pred,
    })
    
    return gs, clf2, X_train, y_train, X_test, y_test, y_pred, y_proba, accuracy, roc_auc_, submission

# ============ 側邊欄 - 參數設定區 ============
with st.sidebar:
    st.header("⚙️ 模型參數設定")
    
    # 模型選擇
    select_model = st.selectbox(
        "🤖 選擇模型:", 
        ('Random Forest', 'SVC'), 
        index=0,
        help="Random Forest: 隨機森林 | SVC: 支援向量機"
    )
    
    st.divider()
    
    # 基本設定
    st.subheader("🎯 基本設定")
    RANDOM_STATE = st.slider(
        "🎲 隨機種子", 
        min_value=350, max_value=450, value=421, step=1,
        help="控制模型訓練的隨機性，確保結果重現"
    )
    TEST_SIZE = st.slider(
        "📊 測試集佔比", 
        min_value=0.1, max_value=0.5, value=0.25, step=0.05,
        help="用於測試的資料比例"
    )
    cv_splits = st.slider(
        "🔄 交叉驗證折數", 
        min_value=2, max_value=10, value=3, step=1,
        help="K-fold交叉驗證的折數"
    )
    
    st.divider()
    
    # 資料處理設定
    st.subheader("🛠️ 資料處理")
    STRATIFY = st.checkbox(
        "⚖️ 依比例切割資料集", 
        value=True,
        help="保持訓練/測試集中各類別的比例一致"
    )
    SMOTE_mode = st.checkbox(
        "🔄 使用SMOTE平衡資料集", 
        value=False, key='smote_toggle',
        help="合成少數類別樣本，解決樣本不平衡問題"
    )
    STANDARD_MODE = st.checkbox(
        "📏 使用標準化", 
        value=False, key='standard_toggle',
        help="將特徵縮放至標準正態分佈"
    )
    PCA_mode = st.checkbox(
        "📉 使用PCA降維", 
        value=False, key='pca_toggle',
        help="降至2維，用於視覺化分析"
    )
    
    st.divider()
    
    # 參數說明
    with st.expander("❓ 參數說明"):
        st.markdown("""
        **📚 技術說明**:
        - **SMOTE**: 合成少數類過採樣技術
        - **標準化**: StandardScaler正規化
        - **PCA**: 主成分分析降維
        - **Stratify**: 分層抽樣保持比例
        """)
    
    # 訓練按鈕
    start = st.button(
        "🚀 開始訓練模型", 
        type="primary", 
        use_container_width=True
    )

# 設定 STRATIFY_mode
STRATIFY_mode = y if STRATIFY else None

# ============ 主頁面 ============
# 標題區域
st.title("🚢 Titanic 生存預測模型調參")

# 主要內容分頁
tab1, tab2, tab3 = st.tabs(["🛠️ 資料預處理流程", "📈 模型結果", "💾 模型下載"])
# ============ Tab 2: 資料處理流程 ============
with tab1:
    st.subheader("🛠️ 資料清理與預處理流程")
    
    # 流程圖片
    try:
        st.image('images/data_cleaned.png', caption="資料清理流程圖",  width=600)
    except:
        st.warning("⚠️ 找不到流程圖片 (images/data_cleaned.png)")
    
    # 詳細步驟
    with st.expander("🔍 查看詳細資料清理步驟", expanded=True):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.code('''
                    # === 資料清理完整流程 ===

                    # 1. 刪除不相關欄位
                    df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
                    # 原因：
                    # - PassengerId: 不影響生存率
                    # - Name, Ticket: 類別過多，編碼時間成本高
                    # - Cabin: 缺失值過多 (近80%)

                    # 2. 填補缺失值
                    df['Age'].fillna(df['Age'].mean(), inplace=True)
                    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
                    # 策略：數值型用平均值，類別型用眾數

                    # 3. 類別特徵編碼
                    sex_mapping = {'male': 0, 'female': 1}
                    embarked_mapping = {'C': 0, 'Q': 1, 'S': 2}

                    df['Sex'] = df['Sex'].map(sex_mapping).astype(int)
                    df['Embarked'] = df['Embarked'].map(embarked_mapping).astype(int)
            ''', language='python')
        
        with col2:
            st.info("**✅ 處理效果**")
            st.success("🗑️ 移除無關特徵")
            st.success("🔧 填補缺失值")  
            st.success("🔢 特徵編碼完成")
            st.success("📊 資料清理完畢")

        cols1, cols2, cols3, cols4 = st.columns([1,1,1,1])    
        with cols1:
            st.info("**清理後統計：**")
        with cols2:
            st.metric("特徵數量", len(df.columns)-1)
        with cols3:
            st.metric("樣本數量", len(df))
        with cols4:
            st.metric("缺失值", df.isnull().sum().sum())

# ============ Tab 3: 模型結果 ============
with tab2:
    if start:
        with st.spinner("🚀 模型訓練中，請稍候..."):
            try:
                gs, clf2, X_train, y_train, X_test, y_test, y_pred, y_proba, accuracy, roc_auc_, submission = fit_model(
                    TEST_SIZE, STRATIFY_mode, RANDOM_STATE, SMOTE_mode, 
                    PCA_mode, STANDARD_MODE, select_model, cv_splits, SCORING="roc_auc"
                )
                st.session_state.model_trained = True
                st.session_state.gs = gs
                st.session_state.submission = submission
                st.session_state.clf2 = clf2
                st.session_state.X_train = X_train
                st.session_state.y_train = y_train
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.y_pred = y_pred
                st.session_state.y_proba = y_proba
                st.session_state.accuracy = accuracy
                st.session_state.roc_auc_ = roc_auc_
                
            except Exception as e:
                st.error(f"❌ 模型訓練失敗: {str(e)}")
                st.stop()
        
        st.success("✅ 模型訓練完成！")
    
    if st.session_state.model_trained:
        st.subheader("📈 模型訓練結果")
        
        # 最佳參數展示
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**🏆 模型選擇**: {select_model}")
        with col2:
            best_params = pd.DataFrame(st.session_state.gs.best_params_, index=[0]).T
            best_params.columns = ['最佳參數值']
            st.dataframe(best_params, use_container_width=True)
        
        # 效能總覽
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 準確率", f"{st.session_state.accuracy:.3f}")
        with col2:
            st.metric("📊 ROC AUC", f"{st.session_state.roc_auc_:.3f}")
        with col3:
            st.metric("🔄 CV折數", cv_splits)
        
        # 詳細評估結果
        eval_tab1, eval_tab2, eval_tab3, eval_tab4, eval_tab5, eval_tab6 = st.tabs([
            "📊 分類報告", "📈 ROC曲線", "🎯 混淆矩陣", "🔍 PCA散佈圖", "📋 Kaggle預測", "🌳 特徵重要性"
        ])
        
        with eval_tab1:  # 分類報告
            st.subheader("📊 詳細分類報告")
            class_report = classification_report(
                st.session_state.y_test, st.session_state.y_pred, 
                output_dict=True, target_names=["Dead(0)", "Alive(1)"]
            )
            class_report_df = pd.DataFrame(class_report).transpose().round(2)
            st.dataframe(class_report_df, use_container_width=True)
            
        with eval_tab2:  # ROC曲線
            st.subheader("📈 ROC曲線分析")
            fpr, tpr, thresholds = roc_curve(st.session_state.y_test, st.session_state.y_proba)
            roc_auc = auc(fpr, tpr)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
            ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Receiver Operating Characteristic (ROC)")
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig, use_container_width=True)
            
        with eval_tab3:  # 混淆矩陣
            st.subheader("🎯 混淆矩陣")
            conf_matrix = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_xticklabels(["Dead", "Alive"])
            ax.set_yticklabels(["Dead", "Alive"])
            ax.set_title("Confusion Matrix")
            st.pyplot(fig, use_container_width=True)
            
            # 混淆矩陣解讀
            tn, fp, fn, tp = conf_matrix.ravel()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("✅ True Negative", tn)
            with col2:
                st.metric("❌ False Positive", fp)
            with col3:
                st.metric("❌ False Negative", fn)
            with col4:
                st.metric("✅ True Positive", tp)
        
        with eval_tab4:  # PCA散佈圖
            st.subheader("🔍 PCA二維散佈圖")
            if PCA_mode:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)
                pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
                pca_df['Target'] = y.values
                
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.scatterplot(
                    data=pca_df,
                    x="PC1", y="PC2",
                    hue="Target",
                    palette="Set1",
                    alpha=0.7,
                    ax=ax
                )
                ax.set_title("PCA 2D Scatter Plot")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig, use_container_width=True)
                
                # PCA解釋
                explained_variance = pca.explained_variance_ratio_
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("PC1 解釋變異", f"{explained_variance[0]:.3f}")
                with col2:
                    st.metric("PC2 解釋變異", f"{explained_variance[1]:.3f}")
                with col3:
                    st.metric("總解釋變異", f"{explained_variance.sum():.3f}")
            else:
                st.warning("⚠️ 請在左側啟用PCA選項以顯示散佈圖")
        
        with eval_tab5:  # Kaggle預測
            st.subheader("📋 Kaggle測試集預測結果")
            st.info("**前5筆預測結果預覽**")
            st.dataframe(st.session_state.submission.head(), use_container_width=True)
            
            # 預測統計
            pred_stats = st.session_state.submission['Survived'].value_counts()
            survival_pred_rate = st.session_state.submission['Survived'].mean()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("💀 預測死亡", pred_stats[0])
            with col2:
                st.metric("❤️ 預測生存", pred_stats[1])
            with col3:
                st.metric("🎯 預測生存率", f"{survival_pred_rate:.2%}")
        
        with eval_tab6:  # 特徵重要性
            st.subheader("🌳 模型可解釋性分析")
            
            if select_model == 'Random Forest':
                best_est = st.session_state.gs.best_estimator_
                rf = best_est.named_steps['clf']
                
                # 確定特徵名稱
                if 'pca' in best_est.named_steps:
                    n_comp = best_est.named_steps['pca'].n_components_
                    feature_names = [f'PC{i+1}' for i in range(n_comp)]
                else:
                    feature_names = list(st.session_state.X_train.columns)
                
                # 特徵重要性圖
                imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=True)
                
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                bars = ax1.barh(imp.index, imp.values, color='skyblue')
                ax1.set_xlabel("Feature Importance")
                ax1.set_title("Random Forest Feature Importance")
                ax1.grid(True, alpha=0.3)
                
                # 在柱狀圖上顯示數值
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                            f'{width:.3f}', ha='left', va='center')
                
                st.pyplot(fig1, use_container_width=True)
                
                # 決策樹可視化
                with st.expander("🌲 決策樹視覺化 (補充說明)", expanded=False):
                    st.info("**📖 說明**: 以下為單一決策樹(深度=3)，用於解釋決策邏輯")
                    
                    fig2, ax2 = plt.subplots(figsize=(20, 12))
                    plot_tree(
                        st.session_state.clf2,
                        feature_names=list(st.session_state.X_train.columns),
                        class_names=['Dead', 'Alive'],
                        filled=True,
                        impurity=False,
                        rounded=True,
                        ax=ax2,
                        fontsize=10
                    )
                    ax2.set_title("Decision Tree Visualization (max_depth=3)", fontsize=16)
                    st.pyplot(fig2, use_container_width=True)
                    
            else:
                st.warning("⚠️ 特徵重要性分析僅支援Random Forest模型")
    else:
        # 未訓練狀態的提示
        st.info("""
        ### 🎯 開始模型訓練
        
        **已配置推薦參數，點擊左側「開始訓練模型」即可查看完整分析結果**
        

        """)

# ============ Tab 4: 模型下載 ============
with tab3:
    if st.session_state.model_trained:
        st.subheader("💾 模型與結果下載")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**🤖 訓練好的模型**")
            st.markdown("""
            **包含內容**:
            - ✅ 最佳參數配置
            - ✅ 完整預處理pipeline
            - ✅ 訓練好的分類器
            """)
            
            # 下載模型
            buf = io.BytesIO()
            joblib.dump(st.session_state.gs.best_estimator_, buf)
            buf.seek(0)
            
            st.download_button(
                label="📥 下載最佳模型 (.joblib)",
                data=buf,
                file_name=f"best_{select_model.lower().replace(' ', '_')}_model.joblib",
                mime="application/octet-stream",
                use_container_width=True
            )
            
        with col2:
            st.info("**📊 Kaggle提交檔案**")
            st.markdown("""
            **檔案格式**:
            - ✅ 符合Kaggle提交要求
            - ✅ PassengerId + Survived
            - ✅ 可直接上傳競賽
            """)
            
            # 下載CSV
            csv_buf = io.BytesIO()
            st.session_state.submission.to_csv(csv_buf, index=False)
            csv_buf.seek(0)
            
            st.download_button(
                label="📥 下載預測結果 (.csv)",
                data=csv_buf,
                file_name=f"titanic_submission_{select_model.lower().replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # 模型摘要資訊
        st.markdown("---")
        with st.expander("📋 模型訓練摘要", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🔧 模型配置**")
                st.write(f"• 模型: {select_model}")
                st.write(f"• 隨機種子: {RANDOM_STATE}")
                st.write(f"• 測試集比例: {TEST_SIZE}")
                st.write(f"• CV折數: {cv_splits}")
                
            with col2:
                st.markdown("**⚙️ 預處理步驟**")
                st.write(f"• 分層抽樣: {'✅' if STRATIFY else '❌'}")
                st.write(f"• SMOTE平衡: {'✅' if SMOTE_mode else '❌'}")
                st.write(f"• 標準化: {'✅' if STANDARD_MODE else '❌'}")
                st.write(f"• PCA降維: {'✅' if PCA_mode else '❌'}")
                
            with col3:
                st.markdown("**📊 性能指標**")
                st.write(f"• 準確率: {st.session_state.accuracy:.3f}")
                st.write(f"• ROC AUC: {st.session_state.roc_auc_:.3f}")
                st.write(f"• 訓練樣本: {len(st.session_state.X_train)}")
                st.write(f"• 測試樣本: {len(st.session_state.X_test)}")
                
    else:
        st.warning("⚠️ 請先完成模型訓練後再進行下載")
        st.info("""
        ### 📋 下載功能說明
        
        完成模型訓練後，您可以下載：
        
        1. **🤖 訓練好的模型檔案** (.joblib格式)
           - 包含完整的預處理pipeline參數
           - 可在其他環境中直接使用
           - 支援joblib.load()載入
        
        2. **📊 Kaggle競賽提交檔案** (.csv格式)
           - 符合官方提交格式要求
           - 包含PassengerId和Survived欄位
           - 可直接上傳至Kaggle競賽平台
        """)

# ============ 底部資訊區 ============
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
    <p> Tech-Stack: Python • Scikit-learn • Streamlit • Pandas • Matplotlib • Github • Kaggle </p>
    </div>
    """, 
    unsafe_allow_html=True
)

# 側邊欄底部資訊
with st.sidebar:
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        <p>🎯 Machine Learning<br> 
        Model Tuning Platform<br>
        Built with Streamlit</p>
        </div>
        """, 
        unsafe_allow_html=True
    )