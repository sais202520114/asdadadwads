import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import io

# --- 1. 환경 설정 ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="Titanic Dashboard", layout="wide")

FILE_PATH = "titanic.xls"

# --- 2. 데이터 처리 ---
@st.cache_data
def load_and_process(file_path):
    try:
        df = pd.read_excel(file_path, engine='xlrd')
    except:
        df = pd.read_excel(file_path)
    
    df = df[['pclass', 'survived', 'sex', 'age', 'fare']].copy()
    df['pclass'] = df['pclass'].fillna(df['pclass'].mode()[0]).astype(int)
    df['survived'] = df['survived'].fillna(0).astype(int)
    df['age'] = df['age'].fillna(df['age'].median())
    df['fare'] = df['fare'].fillna(df['fare'].median())
    df['Death'] = 1 - df['survived']
    df['Survival'] = df['survived']
    
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 100]
    labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)
    return df

# --- 3. 크기 강제 조절 함수 (핵심) ---
def render_small_plot(fig, width=300):
    """그래프를 이미지로 변환하여 너비를 강제 고정함"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    st.image(buf, width=width) # 여기서 너비(픽셀)를 직접 꽂아버림

# --- 4. 메인 로직 ---
def main():
    df = load_and_process(FILE_PATH)
    if df is None: return

    df_norm = df.copy()
    scaler = MinMaxScaler()
    df_norm[['age', 'fare']] = scaler.fit_transform(df_norm[['age', 'fare']])

    st.sidebar.title("🔍 메뉴")
    menu = st.sidebar.radio("이동", ['종합 요약', '분석 그래프', '상관관계/박스플롯'])

    if menu == '종합 요약':
        st.title("🚢 Titanic Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", f"{len(df)}")
        c2.metric("Death", f"{df['Death'].sum()}")
        c3.metric("Surv", f"{df['Survival'].sum()}")
        st.dataframe(df.head(10), use_container_width=True)

    elif menu == '분석 그래프':
        target = st.sidebar.selectbox("대상", ['Death', 'Survival'])
        cat = st.sidebar.selectbox("기준", ['age_group', 'pclass'])
        plot_data = df.groupby(cat, observed=True)[target].sum().reset_index()
        
        st.write(f"**{target} by {cat}**")
        fig, ax = plt.subplots(figsize=(4, 2.5))
        sns.barplot(data=plot_data, x=cat, y=target, ax=ax, palette='viridis')
        ax.set_title(f"{target} by {cat}", fontsize=9)
        ax.tick_params(labelsize=8)
        
        # st.pyplot 대신 render_small_plot 사용 (너비 400픽셀 제한)
        render_small_plot(fig, width=400)

    elif menu == '상관관계/박스플롯':
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Correlation Heatmap**")
            fig1, ax1 = plt.subplots(figsize=(3, 2.5))
            sns.heatmap(df[['survived', 'age', 'fare']].corr(), annot=True, cmap='coolwarm', ax=ax1, annot_kws={"size": 7})
            ax1.tick_params(labelsize=7)
            render_small_plot(fig1, width=300)
            
        with col2:
            st.write("**Box Plot**")
            fig2, ax2 = plt.subplots(figsize=(3, 2.5))
            sns.boxplot(data=df_norm[['age', 'fare']], ax=ax2)
            ax2.tick_params(labelsize=7)
            render_small_plot(fig2, width=300)

if __name__ == "__main__":
    main()
