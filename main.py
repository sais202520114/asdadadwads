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

# --- 3. 메인 앱 실행 ---
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
        st.markdown("---")
        st.dataframe(df.head(10), use_container_width=True)

    elif menu == '분석 그래프':
        target = st.sidebar.selectbox("대상", ['Death', 'Survival'])
        cat = st.sidebar.selectbox("기준", ['age_group', 'pclass'])
        plot_data = df.groupby(cat, observed=True)[target].sum().reset_index()
        
        # [2, 1] 비율로 나누어 그래프가 화면의 약 60%만 차지하게 함 (적당한 크기)
        col_plot, col_empty = st.columns([2, 1]) 
        
        with col_plot:
            st.subheader(f"Analysis: {target} by {cat}")
            fig, ax = plt.subplots(figsize=(8, 4)) # figsize를 다시 적당히 키움
            sns.barplot(data=plot_data, x=cat, y=target, ax=ax, palette='viridis')
            ax.set_title(f"{target} by {cat} (English Chart)", fontsize=12)
            st.pyplot(fig) # 컨테이너 너비에 맞춤 (하지만 컬럼이 이미 제한됨)

    elif menu == '상관관계/박스플롯':
        # 2개 컬럼으로 나누어 너무 크지도 작지도 않게 배치
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Correlation Matrix**")
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            sns.heatmap(df[['survived', 'age', 'fare']].corr(), annot=True, cmap='coolwarm', ax=ax1)
            st.pyplot(fig1)
            
        with col2:
            st.write("**Normalized Box Plot**")
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            sns.boxplot(data=df_norm[['age', 'fare']], ax=ax2)
            st.pyplot(fig2)

if __name__ == "__main__":
    main()
