import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

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
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
        else:
            # 파일이 없을 경우 배포 환경 에러 방지를 위한 백업 로직
            df = sns.load_dataset("titanic")
    except Exception as e:
        st.error(f"오류: 파일 로드 실패 ({e})")
        return None
    df_clean = df[['pclass', 'survived', 'sex', 'age', 'fare']].copy()
    return df_clean

def handle_missing_data(df):
    df['pclass'] = df['pclass'].fillna(df['pclass'].mode()[0]).astype(int)
    df['survived'] = df['survived'].fillna(0).astype(int)
    df['age'] = df['age'].fillna(df['age'].median())
    df['fare'] = df['fare'].fillna(df['fare'].median())
    return df

def handle_outliers(df):
    df['age'] = np.where((df['age'] < 0) | (df['age'] > 100), np.nan, df['age'])
    Q1_fare = df['fare'].quantile(0.25)
    Q3_fare = df['fare'].quantile(0.75)
    IQR_fare = Q3_fare - Q1_fare
    lower_bound_fare = Q1_fare - 1.5 * IQR_fare
    upper_bound_fare = Q3_fare + 1.5 * IQR_fare
    df['fare'] = np.where((df['fare'] < lower_bound_fare) | (df['fare'] > upper_bound_fare), np.nan, df['fare'])
    return df

def create_analysis_columns(df):
    df['Death'] = 1 - df['survived']
    df['Survival'] = df['survived']
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 100]
    labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=True, include_lowest=True)
    return df

def normalize_data(df):
    scaler = MinMaxScaler()
    # MinMaxScaler는 NaN이 있으면 에러가 나므로 결측치를 최종 확인 후 스케일링
    df[['age', 'fare']] = df[['age', 'fare']].fillna(df[['age', 'fare']].median())
    df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])
    return df

# --- 시각화 함수: 박스 플롯 ---
def plot_boxplot(df):
    st.subheader("📊 박스 플롯: 나이 (Age)와 요금 (Fare)")
    fig, ax = plt.subplots(figsize=(4, 3), tight_layout=True)
    sns.boxplot(data=df[['age', 'fare']], ax=ax, palette="Set2")
    ax.set_title("Box Plot of Age and Fare (Normalized)", fontsize=10)
    ax.set_ylabel('Normalized Value', fontsize=8)
    st.pyplot(fig, use_container_width=False)

# --- 시각화 함수: 종합 요약 표 ---
def generate_summary_tables(df_raw):
    st.title("타이타닉 데이터 분석 종합 요약 표")
    st.markdown(f"**분석 데이터 파일:** {FILE_PATH}")
    st.markdown("---")
    total_people = len(df_raw)
    total_deaths = df_raw['Death'].sum()
    total_survival = df_raw['Survival'].sum()
    st.header(f"🚢 총 인원 수: {total_people}명")
    col_main1, col_main2 = st.columns(2)
    with col_main1:
        st.subheader(f"💔 총 사망자 수: {total_deaths}명")
        age_death_summary = df_raw.groupby('age_group')['Death'].sum().reset_index()
        st.dataframe(age_death_summary.rename(columns={'age_group': '연령대', 'Death': '사망자'}).set_index('연령대'))
    with col_main2:
        st.subheader(f"✅ 총 구조된 사람 수: {total_survival}명")
        age_survival_summary = df_raw.groupby('age_group')['Survival'].sum().reset_index()
        st.dataframe(age_survival_summary.rename(columns={'age_group': '연령대', 'Survival': '구조자'}).set_index('연령대'))

# --- 시각화 함수: 사망/구조자 분석 (막대/선) ---
def plot_counts(df_raw, category, target, target_name, plot_type, extreme_select):
    if category == 'age':
        plot_data = df_raw.groupby('age_group')[target].sum().reset_index()
        x_col = 'age_group'
    else:
        plot_data = df_raw.groupby(category)[target].sum().reset_index()
        x_col = category
        plot_data[x_col] = plot_data[x_col].astype(str) + ' Class'
    
    fig, ax = plt.subplots(figsize=(5, 3), tight_layout=True)
    if plot_type == 'Bar Chart':
        sns.barplot(x=x_col, y=target, data=plot_data, ax=ax, palette='YlGnBu')
    elif plot_type == 'Line Chart':
        sns.lineplot(x=x_col, y=target, data=plot_data, ax=ax, marker='o')
    st.pyplot(fig, use_container_width=False)

# --- 상관관계 및 산점도 ---
def calculate_correlation(df):
    corr_matrix = df.corr()
    np.fill_diagonal(corr_matrix.values, np.nan)
    corr_unstacked = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
    valid_corr = corr_unstacked.dropna()
    max_corr = valid_corr[valid_corr > 0].head(1)
    min_corr = valid_corr[valid_corr < 0].tail(1)
    return corr_matrix, max_corr, min_corr

def plot_correlation(df, corr_type, plot_type):
    numeric_df = df[['survived', 'age', 'fare']].copy().dropna()
    corr_matrix, max_corr, min_corr = calculate_correlation(numeric_df)
    
    if plot_type == 'Heatmap':
        fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='YlGnBu', ax=ax)
        st.pyplot(fig, use_container_width=False)
    elif plot_type == 'Scatter Plot':
        # 산점도: pclass별 연령과 요금 (Normalized)
        fig, ax = plt.subplots(figsize=(5, 3), tight_layout=True)
        df_plot = df.copy()
        df_plot['pclass_str'] = df_plot['pclass'].astype(str)
        sns.scatterplot(x='age', y='fare', data=df_plot, hue='pclass_str', style='pclass_str', palette='deep', ax=ax)
        ax.set_xlabel('Age (Normalized)')
        ax.set_ylabel('Fare (Normalized)')
        st.pyplot(fig, use_container_width=False)

# --- 분위수 분석 ---
def analyze_quantiles_and_outliers(df_raw):
    st.markdown("---")
    st.header("📈 분위수 및 이상치 분석 결과")
    for var in ['age', 'fare']:
        q1, q2, q3 = df_raw[var].quantile([0.25, 0.5, 0.75])
        st.write(f"**{var.capitalize()}** - Q1: {q1:.2f}, Median: {q2:.2f}, Q3: {q3:.2f}")

# --- 메인 실행부 ---
def main():
    data = load_data(FILE_PATH)
    if data is None: return
    
    # 1. 원본 기반 통계 데이터
    data_raw = handle_missing_data(data.copy())
    data_raw = create_analysis_columns(data_raw)
    
    # 2. 정규화 및 이상치 처리 기반 시각화 데이터
    data_viz = handle_missing_data(data.copy())
    data_viz = handle_outliers(data_viz)
    data_viz = handle_missing_data(data_viz)
    data_viz = create_analysis_columns(data_viz)
    data_viz = normalize_data(data_viz)
    
    st.sidebar.title("메뉴 선택")
    graph_type = st.sidebar.radio("📊 분석 유형", ('종합 요약 (표)', '사망/구조자 수 분석 (그래프)', '상관관계 분석 (그래프)', '박스 플롯'))
    
    if graph_type == '종합 요약 (표)':
        generate_summary_tables(data_raw)
    elif graph_type == '사망/구조자 수 분석 (그래프)':
        analysis_theme = st.sidebar.radio("주제", ('사망자 수', '구조자 수'))
        target_col = 'Death' if analysis_theme == '사망자 수' else 'Survival'
        cat = st.sidebar.selectbox("카테고리", ('age', 'pclass'))
        style = st.sidebar.radio("유형", ('Bar Chart', 'Line Chart'))
        plot_counts(data_raw, cat, target_col, analysis_theme, style, None)
    elif graph_type == '상관관계 분석 (그래프)':
        plot_style = st.sidebar.radio("유형", ('Scatter Plot', 'Heatmap'))
        plot_correlation(data_viz, None, plot_style)
    elif graph_type == '박스 플롯':
        plot_boxplot(data_viz)
        analyze_quantiles_and_outliers(data_raw)

if __name__ == "__main__":
    main()
