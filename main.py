import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import platform

# =========================================================
# 한글 폰트 설정 (OS별 안정화)
# =========================================================
plt.rcParams['axes.unicode_minus'] = False

os_name = platform.system()
if os_name == "Windows":
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif os_name == "Darwin":  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
else:  # Linux (Streamlit Cloud 포함)
    plt.rcParams['font.family'] = 'NanumGothic'

# =========================================================
# Streamlit 페이지 설정
# =========================================================
st.set_page_config(
    page_title="타이타닉 데이터 분석 대시보드",
    layout="wide"
)

FILE_PATH = "titanic.xls"

# =========================================================
# 데이터 로드
# =========================================================
@st.cache_data
def load_data(path):
    df = pd.read_excel(path)

    df = df[['pclass', 'survived', 'sex', 'age', 'fare']].copy()

    df['pclass'] = df['pclass'].fillna(df['pclass'].mode()[0]).astype(int)
    df['survived'] = df['survived'].fillna(0).astype(int)
    df['age'] = df['age'].fillna(df['age'].median())
    df['fare'] = df['fare'].fillna(df['fare'].median())

    bins = [0, 10, 20, 30, 40, 50, 60, 100]
    labels = ['0-10대', '10-20대', '20-30대', '30-40대',
              '40-50대', '50-60대', '60대 이상']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

    df['Death'] = 1 - df['survived']
    df['Survival'] = df['survived']

    return df

# =========================================================
# 요약 테이블
# =========================================================
def generate_summary_tables(df):
    st.title("타이타닉 데이터 분석 종합 요약")

    st.header(f"💔 총 사망자 수: {df['Death'].sum()}명")
    st.header(f"✅ 총 생존자 수: {df['Survival'].sum()}명")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("연령대별 사망자 수")
        st.dataframe(df.groupby('age_group')['Death'].sum())

    with col2:
        st.subheader("선실 등급별 사망자 수")
        st.dataframe(df.groupby('pclass')['Death'].sum())

# =========================================================
# 사망 / 생존 그래프
# =========================================================
def plot_counts(df, category, target, title, plot_type, extreme):
    if category == 'age':
        data = df.groupby('age_group')[target].sum().reset_index()
        x = 'age_group'
        xlabel = '연령대'
    else:
        data = df.groupby('pclass')[target].sum().reset_index()
        x = 'pclass'
        xlabel = '선실 등급'
        data[x] = data[x].astype(str) + "등급"

    fig, ax = plt.subplots(figsize=(6, 4))

    if plot_type == '막대 그래프':
        sns.barplot(x=x, y=target, data=data, ax=ax)
    else:
        sns.lineplot(x=x, y=target, data=data, ax=ax, marker='o')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(title)

    st.pyplot(fig)

    max_row = data.loc[data[target].idxmax()]
    min_row = data.loc[data[target].idxmin()]

    if extreme == '가장 높은 지점':
        st.success(f"최대: {max_row[x]} ({max_row[target]}명)")
    else:
        st.error(f"최소: {min_row[x]} ({min_row[target]}명)")

# =========================================================
# 상관관계 계산
# =========================================================
def calculate_correlation(df):
    corr_df = df[['age', 'fare', 'pclass']]
    corr = corr_df.corr()

    np.fill_diagonal(corr.values, np.nan)

    pairs = corr.unstack().dropna().sort_values(ascending=False)

    return corr, pairs

# =========================================================
# 상관관계 시각화
# =========================================================
def plot_correlation(df, corr_type, plot_type):
    corr, pairs = calculate_correlation(df)

    if corr_type == '양의 상관관계':
        pair = pairs.head(1)
    else:
        pair = pairs.tail(1)

    x_var, y_var = pair.index[0]

    if plot_type == '히트맵':
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(corr, annot=True, cmap='YlGnBu', ax=ax)
        ax.set_title("변수 간 상관관계 히트맵")
        st.pyplot(fig)

    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=df,
            x=x_var,
            y=y_var,
            hue='survived',
            palette='Set1',
            ax=ax
        )
        ax.set_title(f"{x_var} vs {y_var} 산점도 (생존 여부 색상)")
        st.pyplot(fig)

# =========================================================
# 메인 앱
# =========================================================
def main():
    df = load_data(FILE_PATH)

    st.sidebar.title("메뉴")

    menu = st.sidebar.radio(
        "분석 선택",
        ['종합 요약', '사망/생존 분석', '상관관계 분석']
    )

    if menu == '종합 요약':
        generate_summary_tables(df)

    elif menu == '사망/생존 분석':
        target_name = st.sidebar.radio('대상', ['사망자 수', '생존자 수'])

        if target_name == '사망자 수':
            target = 'Death'
        else:
            target = 'Survival'

        category = st.sidebar.radio('분류', ['age', 'pclass'])
        plot_type = st.sidebar.radio('그래프', ['막대 그래프', '꺾은선 그래프'])
        extreme = st.sidebar.radio('강조', ['가장 높은 지점', '가장 낮은 지점'])

        plot_counts(df, category, target, target_name, plot_type, extreme)

    else:
        corr_type = st.sidebar.radio('상관 방향', ['양의 상관관계', '음의 상관관계'])
        plot_type = st.sidebar.radio('표현', ['산점도', '히트맵'])
        plot_correlation(df, corr_type, plot_type)

if __name__ == "__main__":
    main()
