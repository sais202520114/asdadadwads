import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 파일 경로 설정
FILE_PATH = "titanic.xls"

# --- Matplotlib 폰트 설정: 모든 그래프 관련 폰트는 영어/sans-serif 유지 ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False 

# Streamlit 페이지 설정 (UI는 한국어)
st.set_page_config(
    page_title="타이타닉 데이터 분석 대시보드",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 데이터 로드 및 전처리 함수 (변동 없음) ---
@st.cache_data
def load_data(file_path):
    """엑셀 파일을 로드하고 전처리를 수행합니다."""
    try:
        df = pd.read_excel(file_path)
    except Exception:
        st.error(f"오류: 파일 경로('{FILE_PATH}')를 확인하거나 'xlrd' 라이브러리를 설치해 주세요.")
        return None
    
    df_clean = df[['pclass', 'survived', 'sex', 'age', 'fare']].copy()

    # 결측치 처리 및 타입 변환
    df_clean['pclass'] = df_clean['pclass'].fillna(df_clean['pclass'].mode()[0]).astype(int)
    df_clean['survived'] = df_clean['survived'].fillna(0).astype(int)
    
    df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())
    df_clean['fare'] = df_clean['fare'].fillna(df_clean['fare'].median())
    
    # 연령 그룹 생성 (라벨은 영어로 유지)
    bins = [0, 10, 20, 30, 40, 50, 60, 100]
    labels = ['0-10s', '10-20s', '20-30s', '30-40s', '40-50s', '50-60s', '60s+']
    df_clean['age_group'] = pd.cut(df_clean['age'], bins=bins, labels=labels, right=False)

    df_clean['Death'] = 1 - df_clean['survived']
    df_clean['Survival'] = df_clean['survived']
    
    return df_clean

# --- 요약 표 출력 함수 (UI는 한국어, 변동 없음) ---
def generate_summary_tables(df):
    st.title("타이타닉 데이터 분석 종합 요약 표")
    st.markdown(f"**분석 데이터 파일:** `{FILE_PATH}`")
    st.markdown("---")
    
    total_deaths = df['Death'].sum()
    st.header(f"💔 총 사망자 수: {total_deaths}명")
    st.subheader("사망자 세부 분석")
    
    col_d1, col_d2 = st.columns(2)
    
    age_death_summary = df.groupby('age_group')['Death'].sum().reset_index()
    age_death_summary = age_death_summary.rename(columns={'age_group': '연령대 (Age Group)', 'Death': '사망자 수'})
    with col_d1:
        st.caption("연령별 사망자 수")
        st.dataframe(age_death_summary.set_index('연령대 (Age Group)'))
        
    class_death_summary = df.groupby('pclass')['Death'].sum().reset_index()
    class_death_summary = class_death_summary.rename(columns={'pclass': '선실 등급', 'Death': '사망자 수'})
    class_death_summary['선실 등급'] = class_death_summary['선실 등급'].astype(str) + '등급'
    with col_d2:
        st.caption("선실 등급별 사망자 수")
        st.dataframe(class_death_summary.set_index('선실 등급'))

    st.markdown("---")

    total_survival = df['Survival'].sum()
    st.header(f"✅ 총 구조된 사람 수: {total_survival}명")
    st.subheader("구조자 세부 분석")
    
    col_s1, col_s2 = st.columns(2)

    age_survival_summary = df.groupby('age_group')['Survival'].sum().reset_index()
    age_survival_summary = age_survival_summary.rename(columns={'age_group': '연령대 (Age Group)', 'Survival': '구조자 수'})
    with col_s1:
        st.caption("연령별 구조자 수")
        st.dataframe(age_survival_summary.set_index('연령대 (Age Group)'))
        
    class_survival_summary = df.groupby('pclass')['Survival'].sum().reset_index()
    class_survival_summary = class_survival_summary.rename(columns={'pclass': '선실 등급', 'Survival': '구조자 수'})
    class_survival_summary['선실 등급'] = class_survival_summary['선실 등급'].astype(str) + '등급'
    with col_s2:
        st.caption("선실 등급별 구조자 수")
        st.dataframe(class_survival_summary.set_index('선실 등급'))
    
    st.markdown("---")

# --- 시각화 함수 (그래프 제목/라벨은 영어) ---
def plot_counts(df, category, target, target_name, plot_type, extreme_select):
    """사망/구조자 수를 막대 또는 꺾은선 그래프로 그립니다. (내부 라벨은 영어)"""
    
    if category == 'age':
        plot_data = df.groupby('age_group')[target].sum().reset_index()
        x_col = 'age_group'
        x_label = 'Age Group'
    else: # pclass
        plot_data = df.groupby(category)[target].sum().reset_index()
        x_col = category
        x_label = 'Passenger Class'
        plot_data[x_col] = plot_data[x_col].astype(str) + ' Class'

    total_sum = plot_data[target].sum()
    st.info(f"**Total {target_name} Count by {x_label}:** `{total_sum}`")
    
    st.subheader(f"📊 {target_name} by {x_label}")

    plt.figure(figsize=(6, 4))
    fig, ax = plt.subplots(figsize=(6, 4))
    
    if plot_type == 'Bar Chart':
        sns.barplot(x=x_col, y=target, data=plot_data, ax=ax, palette='YlGnBu', errorbar=None)
        
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 5), 
                        textcoords='offset points', 
                        fontsize=8)
            
    elif plot_type == 'Line Chart':
        sns.lineplot(x=x_col, y=target, data=plot_data, ax=ax, marker='o', color='blue')
        
        for x, y in zip(plot_data[x_col], plot_data[target]):
            ax.annotate(f'{int(y)}', (x, y), 
                        textcoords="offset points", 
                        xytext=(0, 8), 
                        ha='center', 
                        fontsize=8)
        
    ax.set_title(f"{target_name} by {x_label} ({plot_type})", fontsize=12)
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(target_name, fontsize=10)
    st.pyplot(fig, use_container_width=False) 
    

    max_val = plot_data[target].max()
    min_val = plot_data[target].min()
    
    if extreme_select == '가장 높은 지점':
        extreme_data = plot_data[plot_data[target] == max_val]
        extreme_label = '가장 높은 지점'
        st.success(f"🥇 **{extreme_label}:** {extreme_data[x_col].iloc[0]} ({max_val})")
    else:
        extreme_data = plot_data[plot_data[target] == min_val]
        extreme_label = '가장 낮은 지점'
        st.error(f"🥉 **{extreme_label}:** {extreme_data[x_col].iloc[0]} ({min_val})")


def plot_correlation(df, corr_type, plot_type):
    """상관관계를 산점도 또는 히트맵으로 그립니다. (내부 라벨은 영어)"""
    
    # pclass 제외한 연속형 변수만 상관관계 행렬에 포함
    numeric_df = df[['survived', 'age', 'fare']].copy() 
    
    corr_matrix, max_corr, min_corr = calculate_correlation(numeric_df)
    
    st.header(f"🔗 상관관계 분석 결과 ({plot_type})")
    
    if plot_type == 'Heatmap':
        # 1
