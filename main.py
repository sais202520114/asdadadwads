import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# --- 1. 환경 설정 ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="Titanic Dashboard", layout="wide")

FILE_PATH = "titanic.xls"

# --- 2. 데이터 로드 및 전처리 ---
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

# --- 3. 메인 로직 ---
def main():
    df = load_and_process(FILE_PATH)
    if df is None: return

    # 정규화 데이터
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
        st.markdown("---")
        st.dataframe(df.head(10), use_container_width=True)

    elif menu == '분석 그래프':
        target = st.sidebar.selectbox("대상", ['Death', 'Survival'])
        cat = st.sidebar.selectbox("기준", ['age_group', 'pclass'])
        
        plot_data = df.groupby(cat, observed=True)[target].sum().reset_index()
        
        # 컬럼을 나누어 그래프가 왼쪽 작은 공간만 차지하게 함
        col_left, col_right = st.columns([1, 2]) 
        
        with col_left:
            # figsize를 아주 작게 설정
            fig, ax = plt.subplots(figsize=(3, 1.8))
            sns.barplot(data=plot_data, x=cat, y=target, ax=ax, palette='viridis')
            ax.set_title(f"{target} by {cat}", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_xlabel(cat, fontsize=6)
            ax.set_ylabel("Count", fontsize=6)
            
            # use_container_width=False를 설정해야 figsize가 먹힘
            st.pyplot(fig, use_container_width=False)

    elif menu == '상관관계/박스플롯':
        # 세 컬럼으로 나누어 그래프를 더 작게 배치
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.write("**Correlation**")
            fig1, ax1 = plt.subplots(figsize=(2.5, 2))
            sns.heatmap(df[['survived', 'age', 'fare']].corr(), annot=True, cmap='coolwarm', ax=ax1, annot_kws={"size": 5})
            ax1.tick_params(labelsize=5)
            st.pyplot(fig1, use_container_width=False)
            
        with col2:
            st.write("**Box Plot**")
            fig2, ax2 = plt.subplots(figsize=(2.5, 2))
            sns.boxplot(data=df_norm[['age', 'fare']], ax=ax2)
            ax2.tick_params(labelsize=5)
            st.pyplot(fig2, use_container_width=False)

if __name__ == "__main__":
    main()
