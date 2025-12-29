import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 파일 경로
FILE_PATH = "titanic.xls"

# Matplotlib 폰트 설정
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# Streamlit 페이지 설정
st.set_page_config(
    page_title="타이타닉 데이터 분석 대시보드",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- 데이터 로드 -------------------
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        st.error(f"오류: 파일 경로('{FILE_PATH}') 확인 또는 'xlrd' 설치 필요 ({e})")
        return None
    return df[['pclass', 'survived', 'sex', 'age', 'fare']].copy()

# ------------------- 결측치 처리 -------------------
def handle_missing_data(df):
    df['pclass'] = df['pclass'].fillna(df['pclass'].mode()[0]).astype(int)
    df['survived'] = df['survived'].fillna(0).astype(int)
    df['age'] = df['age'].fillna(df['age'].median())
    df['fare'] = df['fare'].fillna(df['fare'].median())
    return df

# ------------------- 이상치 처리 -------------------
def handle_outliers(df):
    df['age'] = np.where((df['age'] < 0) | (df['age'] > 100), np.nan, df['age'])
    Q1 = df['fare'].quantile(0.25)
    Q3 = df['fare'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df['fare'] = np.where((df['fare'] < lower) | (df['fare'] > upper), np.nan, df['fare'])
    return df

# ------------------- 분석 컬럼 -------------------
def create_analysis_columns(df):
    df['Death'] = 1 - df['survived']
    df['Survival'] = df['survived']
    bins = [0,10,20,30,40,50,60,70,100]
    labels = ['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)
    return df

# ------------------- 정규화 -------------------
def normalize_data(df):
    scaler = MinMaxScaler()
    df[['age_norm', 'fare_norm']] = scaler.fit_transform(df[['age','fare']])
    return df

# ------------------- 박스 플롯 -------------------
def plot_boxplot(df):
    st.subheader("📊 박스 플롯: Age & Fare (Normalized)")
    fig, ax = plt.subplots(figsize=(4,3))
    sns.boxplot(data=df[['age_norm','fare_norm']], palette="Set2", ax=ax)
    ax.set_ylabel("Normalized Value")
    st.pyplot(fig, use_container_width=False)

# ------------------- 산점도 -------------------
def plot_scatter(df):
    st.subheader("📊 산점도: Age vs Fare (Normalized)")
    fig, ax = plt.subplots(figsize=(5,4))
    sns.scatterplot(
        x='age_norm',
        y='fare_norm',
        hue='pclass',
        palette='Set1',
        data=df,
        ax=ax
    )
    ax.set_xlabel("Age (Normalized)")
    ax.set_ylabel("Fare (Normalized)")
    ax.set_title("Scatter Plot by Passenger Class")
    st.pyplot(fig, use_container_width=False)

# ------------------- 메인 -------------------
def main():
    df = load_data(FILE_PATH)
    if df is None:
        return

    # 원본 요약용
    df_raw = handle_missing_data(df.copy())
    df_raw = create_analysis_columns(df_raw)

    # 분석용 데이터
    df = handle_missing_data(df)
    df = handle_outliers(df)
    df = handle_missing_data(df)
    df = create_analysis_columns(df)
    df = normalize_data(df)

    st.sidebar.title("메뉴 선택")
    graph_type = st.sidebar.radio(
        "📊 분석 유형 선택",
        ('데이터 확인', '산점도', '박스 플롯')
    )

    if graph_type == '데이터 확인':
        st.subheader("📄 정규화 포함 데이터")
        st.dataframe(df[['pclass','survived','age','fare','age_norm','fare_norm']].head())

    elif graph_type == '산점도':
        plot_scatter(df)

    elif graph_type == '박스 플롯':
        plot_boxplot(df)

if __name__ == "__main__":
    main()
