import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 設定頁面配置
st.set_page_config(
    page_title="Titanic 資料分析專案",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 載入資料
@st.cache_data
def load_data():
    return pd.read_csv('datas/train.csv')

df = load_data()

# ============ 專案背景介紹 ============
st.title('🚢 Titanic 生存預測專案')
tab1, tab2, tab3 = st.tabs(['📖 專案背景', '📋 資料集基本介紹', '📊 探索性資料分析 (EDA)'])
with tab1:
    # 背景介紹區域
    st.header("📖 專案背景")
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        ### 🌊 歷史背景故事
        
        **鐵達尼號**是歷史上最著名的海難之一。1912年4月15日，這艘被譽為「永不沉沒」的豪華郵輪在處女航中撞上冰山，成為20世紀最震撼人心的災難。
        
        在超過2,200名乘客與船員中，有1,500餘人不幸喪生。然而，**並非所有人的生存機會都相等**。除了運氣因素外，生存率與多個社會因素密切相關：
        
        - 🚺 **性別**：「婦孺優先」的救援原則
        - 💰 **社會階層**：頭等艙乘客享有優先權
        - 👶 **年齡**：兒童獲得特別照顧
        - 👨‍👩‍👧‍👦 **家庭狀況**：家庭關係影響逃生策略
        
        這場災難反映了當時社會的階級結構，也成為研究**社會不平等**如何影響生存機會的經典案例。
        """)

    with col2:
        try:
            st.image('images/Titanic.jpg', caption="鐵達尼號", use_container_width=True)
        except:
            st.info("🖼️ 鐵達尼號歷史照片\n（圖片檔案：images/Titanic.jpg）")
        
        # 災難統計
        st.markdown("### 📊 災難統計(訓練集)")
        
        total_people = len(df)
        survivors = df['Survived'].sum()
        deaths = total_people - survivors
        survival_rate = survivors / total_people
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("總人數", f"{total_people:,}")
            st.metric("💀 遇難", f"{deaths:,}")
        with col_b:
            st.metric("❤️ 生還", f"{survivors:,}")
            st.metric("🎯 生存率", f"{survival_rate:.1%}")

    # 問題定義
    st.markdown("---")
    st.header("🎯 問題定義與目標")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🔍 研究問題
        **「什麼因素決定了鐵達尼號乘客的生存機會？」**
        
        - 社會階級是否影響生存率？
        - 性別和年齡如何影響逃生機會？
        - 家庭關係是幫助還是阻礙？
        - 能否從歷史資料中找出生存模式？
        """)

    with col2:
        st.markdown("""
        ### 🚀 專案目標
        **建立機器學習模型預測生存機率**
        
        - 📊 探索影響生存的關鍵因素
        - 🤖 訓練分類模型預測生存
        - 📈 評估模型性能與可解釋性
        - 🏆 參與Kaggle競賽驗證效果
        """)

    with col3:
        st.markdown("""
        ### 🛠️ 技術方法
        **端到端機器學習解決方案**
        
        - 🔍 探索性資料分析(EDA)
        - 🧹 資料清理與特徵工程
        - ⚙️ 模型選擇與超參數調優  
        - 📊 模型評估與結果解釋
        """)
with tab2:  # 資料集介紹
    st.header("📋 資料集基本介紹")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🗂️ 原始資料預覽")
        st.dataframe(df.head(10), use_container_width=True)
        
    with col2:
        st.markdown("### 📏 資料集概況")
        st.info(f"**資料形狀**: {df.shape[0]} 筆資料 × {df.shape[1]} 個特徵")
        
        # 特徵說明
        st.markdown("""
        **🏷️ 特徵說明**:
        - **PassengerId**: 乘客ID
        - **Survived**: 生存狀況 (0=死亡, 1=生存)
        - **Pclass**: 船艙等級 (1=頭等, 2=二等, 3=三等)
        - **Name**: 乘客姓名
        - **Sex**: 性別
        - **Age**: 年齡
        - **SibSp**: 兄弟姊妹/配偶數量
        - **Parch**: 父母/子女數量
        - **Ticket**: 船票編號
        - **Fare**: 船票價格
        - **Cabin**: 客艙編號
        - **Embarked**: 登船港口
        """)
    # ============ 資料品質檢查 ============
    st.markdown("---")
    st.subheader("🔍 資料品質檢查")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📉 缺失值統計")
        missing_values = df.isnull().sum()
        missing_table = pd.DataFrame({
            "缺失數量": missing_values,
            "缺失比例": (missing_values / len(df) * 100).round(2)
        })
        missing_table = missing_table[missing_table["缺失數量"] > 0]
        missing_table["缺失比例"] = missing_table["缺失比例"].astype(str) + "%"
        
        st.dataframe(missing_table, use_container_width=True)
        
        # 缺失值視覺化
        if not missing_table.empty:
            fig, ax = plt.subplots(figsize=(5,5))
            missing_counts = missing_table["缺失數量"]
            colors = ['#ff6b6b' if x > len(df)*0.5 else '#ffa500' if x > len(df)*0.2 else '#4ecdc4' 
                    for x in missing_counts]
            
            bars = ax.bar(missing_counts.index, missing_counts.values, color=colors)
            ax.set_ylabel("Missing Value")
            ax.set_title("Missing Values per Feature")
            ax.tick_params(axis='x', rotation=0)
            
            # 添加數值標籤
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)

    with col2:
        st.markdown("### 📊 樣本分布統計")
        
        # 生存分布
        survival_counts = df["Survived"].value_counts()
        total = survival_counts.sum()
        
        fig, ax = plt.subplots(figsize=(6, 6))
        survival_counts.plot(kind="bar", color=["salmon", "skyblue"], ax=ax)
        ax.set_ylim(0, 650)
        ax.set_ylabel("Number of Passengers")
        ax.set_title("Overall Survival Distribution")
        ax.set_xticklabels(["Dead", "Alive"], rotation=0)
        
        # 添加標籤
        for i, val in enumerate(survival_counts):
            rate = val / total
            ax.text(i, val/2, str(val), ha="center", va="center", 
                color="white", fontweight="bold", fontsize=12)
            ax.text(i, val+20, f"{rate:.1%}", ha="center", va="bottom", 
                color="black", fontweight="bold", fontsize=12)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # 基本統計
        st.markdown("#### 📋 基本統計信息")
        st.info(f"""
        **總乘客數**: {len(df):,} 人  
        **生還者**: {survival_counts[1]:,} 人 ({survival_counts[1]/total:.1%})  
        **遇難者**: {survival_counts[0]:,} 人 ({survival_counts[0]/total:.1%})  
        **平均年齡**: {df['Age'].mean():.1f} 歲  
        **年齡範圍**: {df['Age'].min():.0f} - {df['Age'].max():.0f} 歲
        """)



with tab3:
    # ============ 探索性資料分析 (EDA) ============
    st.header("📊 探索性資料分析 (EDA)")

    # 特徵分布視覺化
    st.subheader("📈 關鍵特徵分布分析")

    tab1, tab2, tab3, tab4 = st.tabs(['🚺🚹 性別 vs 生存率', '🏛️ 艙等 vs 生存率', '👶👴 年齡 vs 生存率', '🎻 年齡分布詳析'])

    with tab1:
        st.markdown("### 性別對生存率的影響")
        
        # 計算統計數據
        gender_survival = pd.crosstab(df["Sex"], df["Survived"])
        gender_survival = gender_survival.loc[["male", "female"]]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 繪製圖表
            fig, ax = plt.subplots(figsize=(8, 6))
            gender_survival.plot(kind="bar", stacked=False, color=["salmon", "skyblue"], ax=ax)
            ax.set_ylim(0, 600)
            ax.set_ylabel("Number of Passengers", fontsize=12)
            ax.set_xlabel("Sex", fontsize=12)
            ax.set_title("Gender vs Survival Rate Analysis", fontsize=14, fontweight='bold')
            ax.set_xticklabels(['male', 'female'], rotation=0)
            ax.legend(["Dead", "Alive"], title="Survival Status", loc='upper right')
            
            # 添加數據標籤
            for i, sex in enumerate(gender_survival.index):
                survived = gender_survival.loc[sex, 1]
                dead = gender_survival.loc[sex, 0]
                total = survived + dead
                rate = survived / total
                high = max(survived, dead) + 20
                
                # 標註人數
                ax.text(i-0.12, dead/2, str(dead), ha="center", va="center", 
                    color="white", fontweight="bold", fontsize=11)
                ax.text(i+0.12, survived/2, str(survived), ha="center", va="center", 
                    color="white", fontweight="bold", fontsize=11)
                # 標註生存率
                ax.text(i, high, f"{rate:.1%}", ha="center", va="bottom", 
                    color="black", fontweight="bold", fontsize=12)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### 🔍 關鍵發現")
            
            male_survival_rate = gender_survival.loc['male', 1] / gender_survival.loc['male'].sum()
            female_survival_rate = gender_survival.loc['female', 1] / gender_survival.loc['female'].sum()
            
            st.metric("👩 女性生存率", f"{female_survival_rate:.1%}", 
                    delta=f"{female_survival_rate-male_survival_rate:.1%} vs 男性")
            st.metric("👨 男性生存率", f"{male_survival_rate:.1%}")
            
            st.markdown(f"""
            **💡 洞察分析**:
            - 女性生存率是男性的 **{female_survival_rate/male_survival_rate:.1f} 倍**
            - 體現了「**婦女優先**」的救援原則
            - 反映當時社會的性別保護觀念
            - 這是影響生存的**最重要因素**
            """)

    with tab2:
        st.markdown("### 社會階層對生存機會的影響")
        
        # 計算統計數據
        class_survival = pd.crosstab(df["Pclass"], df["Survived"])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 繪製圖表
            fig, ax = plt.subplots(figsize=(8, 6))
            class_survival.plot(kind="bar", stacked=False, color=["salmon", "skyblue"], ax=ax)
            ax.set_ylim(0, 450)
            ax.set_ylabel("Number of Passengers", fontsize=12)
            ax.set_xlabel("Pclass", fontsize=12)
            ax.set_title("Passenger Class vs Survival Rate Analysis", fontsize=14, fontweight='bold')
            ax.set_xticklabels(['Class1', 'Class2', 'Class3'], rotation=0)
            ax.legend(["Dead", "Alive"], title="Survival Status", loc='upper left')
            
            # 添加數據標籤
            for i, pclass in enumerate(class_survival.index):
                survived = class_survival.loc[pclass, 1]
                dead = class_survival.loc[pclass, 0]
                total = survived + dead
                rate = survived / total
                high = max(survived, dead) + 15
                
                # 標註人數
                ax.text(i-0.12, dead/2, str(dead), ha="center", va="center", 
                    color="white", fontweight="bold", fontsize=10)
                ax.text(i+0.12, survived/2, str(survived), ha="center", va="center", 
                    color="white", fontweight="bold", fontsize=10)
                # 標註生存率
                ax.text(i, high, f"{rate:.1%}", ha="center", va="bottom", 
                    color="black", fontweight="bold", fontsize=12)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### 🔍 階級分析")
            
            for pclass in [1, 2, 3]:
                rate = class_survival.loc[pclass, 1] / class_survival.loc[pclass].sum()
                class_name = ['頭等艙', '二等艙', '三等艙'][pclass-1]
                st.metric(f"🏛️ {class_name}", f"{rate:.1%}")
            
            st.markdown("""
            **💡 社會階層效應**:
            - **頭等艙**：最高生存率，享有救生艇優先權
            - **二等艙**：中等生存率，部分特權保護
            - **三等艙**：最低生存率，位置不利逃生
            
            **🏗️ 船體結構影響**:
            - 三等艙位於船底，逃生路線最長
            - 頭等艙靠近甲板，最快接近救生艇
            """)

    with tab3:
        st.markdown("### 年齡對生存機會的影響")
        
        # 年齡分組
        bins = [0, 10, 20, 30, 40, 50, 60, 70, 100]
        labels = ['0-9y', '10-19y', '20-29y', '30-39y', '40-49y', '50-59y', '60-69y', '70+y']
        df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
        
        # 計算生存率
        age_survival = pd.crosstab(df['AgeGroup'], df['Survived'])
        totals = age_survival.sum(axis=1)
        rates = age_survival[1] / totals
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 繪製生存率趨勢圖
            fig, ax = plt.subplots(figsize=(8, 6))
            
            x = np.arange(len(rates))
            line = ax.plot(x, rates, marker='o', linewidth=3, markersize=8, 
                        color='skyblue', markerfacecolor='salmon', markeredgewidth=2)
            
            # 標註每個點的生存率
            for i, rate in enumerate(rates):
                ax.text(x[i], rate + 0.03, f"{rate:.1%}", ha='center', va='bottom', 
                    fontweight='bold', fontsize=11)
            
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45)
            ax.set_ylim(0, 0.8)
            ax.set_ylabel('Survival Rate', fontsize=12)
            ax.set_xlabel('Age Group', fontsize=12)
            ax.set_title('Survival Rate by Age Group', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### 📊 年齡洞察")
            
            # 找出最高和最低生存率的年齡組
            max_age_group = rates.idxmax()
            min_age_group = rates.idxmin()
            
            st.metric("👶 兒童組生存率", f"{rates[0]:.1%}", "最受保護")
            st.metric("👴 老年組生存率", f"{rates.iloc[-1]:.1%}", "生存困難")
            
            st.markdown(f"""
            **🎯 年齡趨勢分析**:
            - **{max_age_group}**: 生存率最高 ({rates[max_age_group]:.1%})
            - **{min_age_group}**: 生存率最低 ({rates[min_age_group]:.1%})
            
            **👶 兒童優勢**:
            - 「婦孺優先」政策受益
            - 體型小，救援容易
            
            **👨‍💼 青壯年劣勢**:
            - 承擔救援他人責任
            - 最後登上救生艇
            """)

    with tab4:
        st.markdown("### 年齡與性別交互分析")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 小提琴圖
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.violinplot(
                data=df,
                x="Sex",
                y="Age", 
                hue="Survived",
                split=True,
                palette=["salmon", "skyblue"],
                order = ['male', 'female'],
                ax=ax
            )
            
            ax.set_title("Age Distribution and Survival Status (by Gender)", fontsize=14, fontweight='bold')
            ax.set_xlabel("Sex", fontsize=12)
            ax.set_ylabel("Age", fontsize=12)
            ax.set_xticklabels(['Male', 'Female'])
            
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, ["Dead", "Alive"], title="Survival Status", loc='upper right')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### 🔍 交互效應")
            
            # 按性別和年齡分組分析
            young_female = df[(df['Sex']=='female') & (df['Age']<=16)]['Survived'].mean()
            young_male = df[(df['Sex']=='male') & (df['Age']<=16)]['Survived'].mean() 
            adult_female = df[(df['Sex']=='female') & (df['Age']>16)]['Survived'].mean()
            adult_male = df[(df['Sex']=='male') & (df['Age']>16)]['Survived'].mean()
            
            st.metric("👧 女童生存率", f"{young_female:.1%}")
            st.metric("👦 男童生存率", f"{young_male:.1%}")
            st.metric("👩 成年女性生存率", f"{adult_female:.1%}")
            st.metric("👨 成年男性生存率", f"{adult_male:.1%}")
            
            st.markdown("""
            **💡 關鍵發現**:
            - **女童**享有雙重保護 (性別+年齡)
            - **成年男性**生存機會最低
            - 年齡分布顯示青壯年乘客最多
            - 老年乘客相對較少但生存困難
            """)


    # ============ 總結洞察 ============
    st.markdown("---")
    st.header("🎯 探索性分析總結")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🔑 關鍵影響因素
        
        **📊 影響力排序**:
        1. **性別** - 女性生存率74% vs 男性19%
        2. **社會階層** - 頭等艙63% vs 三等艙24%  
        3. **年齡** - 兒童61% vs 老年人17%
        
        **💡 社會現象**:
        - 「婦孺優先」救援原則
        - 階級特權影響生存機會
        - 船體結構決定逃生難易
        """)

    with col2:
        st.markdown("""
        ### 📈 資料特徵總覽
        
        **✅ 資料優勢**:
        - 樣本數量適中 (891筆)
        - 目標變數平衡性尚可
        - 包含多維度特徵
        
        **⚠️ 資料挑戰**:
        - Age欄位缺失20%
        - Cabin欄位缺失77%
        - 部分特徵需要工程化處理
        """)

    with col3:
        st.markdown("""
        ### 🚀 建模方向建議
        
        **🎯 特徵工程重點**:
        - 移除不使用的特徵
        - 處理年齡缺失值
        - 性別、港口轉為數值

        **🤖 模型選擇考量**:
        - 樹狀模型：捕捉非線性關係
        - 整合模型：提升預測準確性
        - 可解釋性：理解決策邏輯
        """)

    # 底部資訊
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; padding: 1rem; background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px;'>
            <p style='color: #495057; margin: 0;'>
            📊 <strong>探索性資料分析完成</strong> | 下一步：資料清理與機器學習建模 ➡️
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )


st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
    <p> Tech-Stack: Python • Scikit-learn • Streamlit • Pandas • Matplotlib • Github • Kaggle </p>
    </div>
    """, 
    unsafe_allow_html=True
)
