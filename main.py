import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 파일 경로
FILE_PATH = "titanic.xls"

# Matplotlib 설정
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# Streamlit 페이지 설정
st.set_page_config(
    page_title="타이타닉 데이터 분석 대시보드",
    layout="wide"
)

# ---------------- 데이터 로드 ----------------
@st.cache_data
def load_data(path):
    return pd.read_excel(path)[['pclass', 'survived', 'sex', 'age', 'fare']]

# ---------------- 결측치 처리 ----------------
def handle_missing_data(df):
    df['pclass'] = df['pclass'].fillna(df['pclass'].mode()[0]).astype(int)
    df['survived'] = df['survived'].fillna(0).astype(int)
    df['age'] = df['age'].fillna(df['age'].median())
    df['fare'] = df['fare'].fillna(df['fare'].median())
    return df

# ---------------- 이상치 처리 ----------------
def handle_outliers(df):
    df['age'] = np.where((df['age'] < 0) | (df['age'] > 100), np.nan, df['age'])

    Q1 = df['fare'].quantile(0.25)
    Q3 = df['fare'].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df['fare'] = np.where((df['fare'] < lower) | (df['fare'] > upper), np.nan, df['fare'])

    return df

# ---------------- 분석 컬럼 ----------------
def create_analysis_columns(df):
    df['Death'] = 1 - df['survived']
    df['Survival'] = df['survived']

    bins = [0,10,20,30,40,50,60,70,100]
    labels = ['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)
    return df

# ---------------- 🔥 정규화 파트 ----------------
def normalize_data(df):
    """
    Min-Max Scaling
    age, fare 값을 0~1 범위로 정규화
    """
    scaler = MinMaxScaler()
    df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])
    return df

# ---------------- 박스플롯 ----------------
def plot_boxplot(df):
    st.subheader("📦 Age & Fare Boxplot (Normalized)")
    fig, ax = plt.subplots(figsize=(4,3))
    sns.boxplot(data=df[['age','fare']], palette="Set2", ax=ax)
    ax.set_ylabel("Normalized Value")
    st.pyplot(fig)

# ---------------- 메인 ----------------
def main():
    data = load_data(FILE_PATH)

    # 원본 요약용
    raw = handle_missing_data(data.copy())
    raw = create_analysis_columns(raw)

    # 분석용 데이터
    data = handle_missing_data(data)
    data = handle_outliers(data)
    data = handle_missing_data(data)
    data = create_analysis_columns(data)
    data = normalize_data(data)

    st.sidebar.title("메뉴")
    menu = st.sidebar.radio("선택", ["데이터 미리보기", "박스 플롯"])

    if menu == "데이터 미리보기":
        st.subheader("📄 정규화된 데이터")
        st.dataframe(data.head())

    elif menu == "박스 플롯":
        plot_boxplot(data)

if __name__ == "__main__":
    main()
