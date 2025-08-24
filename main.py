import streamlit as st

pg = st.navigation([
    st.Page('pages/p1.py', title='Titanic 相關資訊', icon='🚢'),
    st.Page('pages/p2.py', title='Titanic 模型調參示範', icon='📊'),
    st.Page('pages/p3.py', title='Titanic 模型預測', icon='🔮'),
])
pg.run()