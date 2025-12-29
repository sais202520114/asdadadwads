import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

# --- 1. 환경 설정 ---
# 한글 깨짐 방지: 시스템 폰트가 없어도 오류 안 나게 설정
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="타이타닉 대시보드", layout="wide")

# --- 2. 데이터 로드 (오류 방지 로직 강화) ---
@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return None
    
    try:
        # xls는 xlrd 엔진이 필요함. 설치 안 되어 있을 경우를 위해 엔진 시도
        try:
            df = pd.read_excel(file_path, engine='xlrd')
        except:
            df = pd.read_excel(file_path) # 기본 엔진 시도
            
        # 필수 컬럼 존재 확인
        cols = ['pclass', 'survived', 'sex', 'age', 'fare']
        df = df[cols].copy()
        
        # 기본 전처리 (결측치)
        df['pclass'] = df['pclass'].fillna(df['pclass'].mode()[0]).astype(int)
        df['survived'] = df['survived'].fillna(0).astype(int)
        df['age'] = df['age'].fillna(df['age'].median())
        df['fare'] = df['fare'].fillna(df['fare'].median())
        
        # 분석용 열
        df['Death'] = 1 - df['survived']
        df['Survival'] = df['survived']
        
        # 연령대 생성
        bins = [0, 10, 20, 30, 40, 50, 60, 70, 100]
        labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71+']
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)
        
        return df
    except Exception as e:
        st.error(f"❌ 데이터 처리 중 오류 발생: {e}")
        return None

# --- 3. 메인 로직 ---
def main():
    df = load_data("titanic.xls")
    
    if df is not None:
        st.sidebar.title("🔍 분석 메뉴")
        menu = st.sidebar.radio("선택", ["데이터 요약", "사망/생존 분석", "통계"])

        if menu == "데이터 요약":
            st.title("🚢 타이타닉 요약")
            c1, c2, c3 = st.columns(3)
            c1.metric("총 승객", f"{len(df)}명")
            c2.metric("총 사망", f"{df['Death'].sum()}명")
            c3.metric("총 생존", f"{df['Survival'].sum()}명")
            st.dataframe(df.head())

        elif menu == "사망/생존 분석":
            target = st.sidebar.selectbox("대상", ["Death", "Survival"])
            cat = st.sidebar.selectbox("기준", ["age_group", "pclass"])
            
            # 여기서 observed=True를 써야 카테고리 에러가 안 남
            plot_data = df.groupby(cat, observed=True)[target].sum().reset_index()
            
            fig, ax = plt.subplots()
            sns.barplot(data=plot_data, x=cat, y=target, ax=ax)
            st.pyplot(fig)

        elif menu == "통계":
            st.subheader("📊 상관관계 (Age, Fare, Survived)")
            corr = df[['age', 'fare', 'survived']].corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
