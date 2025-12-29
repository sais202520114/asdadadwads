import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ===============================
# 기본 설정
# ===============================
FILE_PATH = "titanic.xls"

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(
    page_title="타이타닉 데이터 분석",
    layout="wide"
)

# ===============================
# 데이터 로드
# ===============================
@st.cache_data
def load_data(path):
    df = pd.read_excel(path)
    return df[['pclass', 'survived', 'sex', 'age', 'fare']]

# ===============================
# 결측치 처리
# ===============================
def handle_missing_data(df):
    df['pclass'] = df['pclass'].fillna(df['pclass'].mode()[0]).astype(int)
    df['survived'] = df['survived'].fillna(0).astype(int)
    df['age'] = df['age'].fillna(df['age'].median())
    df['fare'] = df['fare'].fillna(df['fare'].median())
    return df

# ===============================
# 이상치 처리
# ===============================
def handle_outliers(df):
    # age: 0~100
    df.loc[(df['age'] < 0) | (df['age'] > 100), 'age'] = np.nan

    # fare: IQR
    Q1 = df['fare'].quantile(0.25)
    Q3 = df['fare'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df.loc[(df['fare'] < lower) | (df['fare'] > upper), 'fare'] = np.nan
    return df

# ===============================
# 분석용 컬럼
# ===============================
def create_analysis_columns(df):
    df['Death'] = 1 - df['survived']
    df['Survival'] = df['survived']

    bins = [0,10,20,30,40,50,60,70,100]
    labels = ['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)
    return df

# ===============================
# 🔥 정규화 (중요)
# ===============================
def normalize_data(df):
    scaler = MinMaxScaler()

    # 정규화는 숫자 컬럼만
    df[['age_norm', 'fare_norm']] = scaler.fit_transform(
        df[['age', 'fare']]
    )
    return df

# ===============================
# 산점도 (정규화 기준)
# ===============================
def plot_scatter(df):
    st.subheader("📊 산점도: Age vs Fare (Normalized)")

    fig, ax = plt.subplots(figsize=(5,4))

    sns.scatterplot(
        data=df,
        x='age_norm',
        y='fare_norm',
        hue='pclass',
        palette='Set1',
        ax=ax
    )

    ax.set_xlabel("Age (Normalized)")
    ax.set_ylabel("Fare (Normalized)")
    ax.set_title("Normalized Scatter Plot by Passenger Class")

    st.pyplot(fig)

# ===============================
# 박스플롯
# ===============================
def plot_boxplot(df):
    st.subheader("📦 박스 플롯 (Normalized)")

    fig, ax = plt.subplots(figsize=(4,3))
    sns.boxplot(
        data=df[['age_norm', 'fare_norm']],
        palette="Set2",
        ax=ax
    )
    ax.set_ylabel("Normalized Value")
    st.pyplot(fig)

# ===============================
# 메인
# ===============================
def main():
    df = load_data(FILE_PATH)

    # 전처리
    df = handle_missing_data(df)
    df = handle_outliers(df)
    df = handle_missing_data(df)
    df = create_analysis_columns(df)

    # 🔥 정규화
    df = normalize_data(df)

    st.sidebar.title("메뉴")
    menu = st.sidebar.radio(
        "선택",
        ["데이터 확인", "산점도", "박스 플롯"]
    )

    if menu == "데이터 확인":
        st.subheader("📄 정규화 포함 데이터")
        st.dataframe(
            df[['pclass','survived','age','fare','age_norm','fare_norm']].head()
        )

    elif menu == "산점도":
        plot_scatter(df)

    elif menu == "박스 플롯":
        plot_boxplot(df)

if __name__ == "__main__":
    main()
