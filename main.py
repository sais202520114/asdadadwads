import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 파일 경로 설정
FILE_PATH = "titanic.xls"

# --- Matplotlib 폰트 설정 ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# Streamlit 페이지 설정
st.set_page_config(
    page_title="타이타닉 데이터 분석 대시보드",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 데이터 로드 및 전처리 함수 ---
@st.cache_data
def load_data(file_path):
    try:
        # engine='xlrd' 또는 'openpyxl' 시도
        df = pd.read_excel(file_path)
    except Exception as e:
        st.error(f"오류: 파일을 찾을 수 없거나 라이브러리(xlrd, openpyxl)가 필요합니다. ({e})")
        return None
    df_clean = df[['pclass', 'survived', 'sex', 'age', 'fare']].copy()
    return df_clean

def handle_missing_data(df):
    df = df.copy()
    df['pclass'] = df['pclass'].fillna(df['pclass'].mode()[0]).astype(int)
    df['survived'] = df['survived'].fillna(0).astype(int)
    df['age'] = df['age'].fillna(df['age'].median())
    df['fare'] = df['fare'].fillna(df['fare'].median())
    return df

def handle_outliers(df):
    df = df.copy()
    # 나이 이상치 처리
    df['age'] = np.where((df['age'] < 0) | (df['age'] > 100), np.nan, df['age'])
    # 요금 IQR 기준 이상치 처리
    Q1_fare = df['fare'].quantile(0.25)
    Q3_fare = df['fare'].quantile(0.75)
    IQR_fare = Q3_fare - Q1_fare
    lower_bound_fare = Q1_fare - 1.5 * IQR_fare
    upper_bound_fare = Q3_fare + 1.5 * IQR_fare
    df['fare'] = np.where((df['fare'] < lower_bound_fare) | (df['fare'] > upper_bound_fare), np.nan, df['fare'])
    return df

def create_analysis_columns(df):
    df = df.copy()
    df['Death'] = 1 - df['survived']
    df['Survival'] = df['survived']
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 100]
    labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=True, include_lowest=True)
    return df

def normalize_data(df):
    df = df.copy()
    scaler = MinMaxScaler()
    df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])
    return df

# --- 시각화 함수들 ---
def plot_boxplot(df):
    st.subheader("📊 박스 플롯: 나이 (Age)와 요금 (Fare)")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df[['age', 'fare']], ax=ax, palette="Set2")
    ax.set_title("Box Plot of Age and Fare (Normalized)")
    st.pyplot(fig)

def generate_summary_tables(df_raw):
    st.title("🚢 타이타닉 데이터 분석 종합 요약")
    st.info(f"분석 데이터 파일: {FILE_PATH}")
    
    total_people = len(df_raw)
    total_deaths = df_raw['Death'].sum()
    total_survival = df_raw['Survival'].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("총 인원", f"{total_people}명")
    col2.metric("총 사망자", f"{total_deaths}명", delta="-사망", delta_color="inverse")
    col3.metric("총 구조자", f"{total_survival}명", delta="+구조")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("💔 사망자 상세 요약")
        st.write("**연령대별**")
        st.dataframe(df_raw.groupby('age_group')['Death'].sum(), use_container_width=True)
        st.write("**선실 등급별**")
        st.dataframe(df_raw.groupby('pclass')['Death'].sum(), use_container_width=True)
    with c2:
        st.subheader("✅ 구조자 상세 요약")
        st.write("**연령대별**")
        st.dataframe(df_raw.groupby('age_group')['Survival'].sum(), use_container_width=True)
        st.write("**선실 등급별**")
        st.dataframe(df_raw.groupby('pclass')['Survival'].sum(), use_container_width=True)

def plot_counts(df_raw, category, target, target_name, plot_type, extreme_select):
    if category == 'age':
        plot_data = df_raw.groupby('age_group', observed=False)[target].sum().reset_index()
        x_col = 'age_group'
    else:
        plot_data = df_raw.groupby(category)[target].sum().reset_index()
        x_col = category
        plot_data[x_col] = plot_data[x_col].astype(str) + " Class"

    fig, ax = plt.subplots(figsize=(7, 4))
    if plot_type == 'Bar Chart':
        sns.barplot(x=x_col, y=target, data=plot_data, ax=ax, palette='viridis')
    else:
        sns.lineplot(x=x_col, y=target, data=plot_data, ax=ax, marker='o', size=10)
    
    ax.set_title(f"{target_name} by {category.capitalize()}")
    st.pyplot(fig)

    # 지점 강조
    if extreme_select == '가장 높은 지점':
        top = plot_data.loc[plot_data[target].idxmax()]
        st.success(f"🥇 최고치: {top[x_col]} ({top[target]}명)")
    else:
        bottom = plot_data.loc[plot_data[target].idxmin()]
        st.error(f"🥉 최저치: {bottom[x_col]} ({bottom[target]}명)")

def plot_correlation(df, corr_plot_type):
    numeric_df = df[['survived', 'age', 'fare']].dropna()
    corr_matrix = numeric_df.corr()
    
    st.subheader(f"🔗 상관관계 분석: {corr_plot_type}")
    fig, ax = plt.subplots(figsize=(6, 5))
    if corr_plot_type == 'Heatmap':
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    else:
        sns.scatterplot(data=df, x='age', y='fare', hue='survived', alpha=0.6, ax=ax)
    st.pyplot(fig)

def analyze_quantiles_and_outliers(df_raw):
    st.markdown("---")
    st.header("📈 통계 상세 분석 (분위수)")
    for col in ['age', 'fare']:
        q1 = df_raw[col].quantile(0.25)
        median = df_raw[col].median()
        q3 = df_raw[col].quantile(0.75)
        st.write(f"**{col.capitalize()}** - Q1: {q1:.1f}, Median: {median:.1f}, Q3: {q3:.1f}")

# --- 메인 앱 ---
def main():
    data = load_data(FILE_PATH)
    if data is None: return

    # 전처리 데이터 생성
    data_raw = handle_missing_data(data)
    data_raw = create_analysis_columns(data_raw)
    
    # 정규화 데이터 생성 (박스플롯/상관관계용)
    data_norm = handle_outliers(data_raw)
    data_norm = normalize_data(data_norm)

    # 사이드바 메뉴
    st.sidebar.title("🔍 Dashboard Menu")
    menu = st.sidebar.radio("메뉴를 선택하세요", 
        ['종합 요약 (표)', '사망/구조자 분석 (그래프)', '상관관계 분석', '박스 플롯'])

    if menu == '종합 요약 (표)':
        generate_summary_tables(data_raw)
    
    elif menu == '사망/구조자 분석 (그래프)':
        theme = st.sidebar.selectbox("분석 대상", ['사망자 수', '구조자 수'])
        target = 'Death' if theme == '사망자 수' else 'Survival'
        cat = st.sidebar.selectbox("분류 기준", ['age', 'pclass'])
        style = st.sidebar.radio("그래프 형태", ['Bar Chart', 'Line Chart'])
        extreme = st.sidebar.radio("강조 지점", ['가장 높은 지점', '가장 낮은 지점'])
        plot_counts(data_raw, cat, target, theme, style, extreme)

    elif menu == '상관관계 분석':
        style = st.sidebar.radio("시각화 방식", ['Heatmap', 'Scatter Plot'])
        plot_correlation(data_norm, style)

    elif menu == '박스 플롯':
        plot_boxplot(data_norm)
        analyze_quantiles_and_outliers(data_raw)

if __name__ == "__main__":
    main()
