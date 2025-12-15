import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 사용자님이 요청하신 파일명으로 정확히 설정
FILE_PATH = "titanic.xls"

# Streamlit 페이지 설정
st.set_page_config(
    page_title="타이타닉 데이터 분석 대시보드",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 데이터 로드 및 전처리 함수 ---
@st.cache_data
def load_data(file_path):
    """엑셀(.xls) 파일을 로드하고 필요한 전처리를 수행합니다."""
    try:
        df = pd.read_excel(file_path)
    except Exception:
        st.error(f"오류: 파일을 찾을 수 없거나 'xlrd' 라이브러리가 설치되지 않았습니다. 파일 경로('{file_path}')와 requirements.txt를 확인해 주세요.")
        return None
    
    # 분석에 필요한 열 선택
    df_clean = df[['pclass', 'survived', 'sex', 'age', 'fare']].copy()

    # 'pclass', 'survived'는 수치형이지만 범주형으로 사용하기 위해 정수형 변환
    df_clean['pclass'] = df_clean['pclass'].fillna(df_clean['pclass'].mode()[0]).astype(int)
    df_clean['survived'] = df_clean['survived'].fillna(0).astype(int)
    
    # 'age'와 'fare' 결측값은 중앙값으로 대체
    df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())
    df_clean['fare'] = df_clean['fare'].fillna(df_clean['fare'].median())
    
    return df_clean

# --- 시각화 함수 ---

def plot_counts(df, category, target, plot_type):
    """사망/구조자 수를 막대 또는 꺾은선 그래프로 그립니다."""
    st.subheader(f"📊 {target} (타겟) vs. {category} (분류)")
    
    # 연령을 그룹화 (Age Group)
    if category == 'age':
        bins = [0, 10, 20, 30, 40, 50, 60, 100]
        labels = ['0-10대', '10-20대', '20-30대', '30-40대', '40-50대', '50-60대', '60대 이상']
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
        plot_data = df.groupby('age_group')[target].sum().reset_index()
        x_col = 'age_group'
    else:
        plot_data = df.groupby(category)[target].sum().reset_index()
        x_col = category
        if category == 'pclass':
             plot_data[x_col] = plot_data[x_col].astype(str).replace({'1': '1등급', '2': '2등급', '3': '3등급'})


    fig, ax = plt.subplots(figsize=(10, 5))
    
    if plot_type == '막대 그래프':
        sns.barplot(x=x_col, y=target, data=plot_data, ax=ax, palette='viridis')
    elif plot_type == '꺾은선 그래프':
        sns.lineplot(x=x_col, y=target, data=plot_data, ax=ax, marker='o', color='red')
    
    ax.set_title(f"{category}별 {target}", fontsize=15)
    ax.set_xlabel(category.replace('pclass', '클래스').replace('age_group', '연령대'))
    ax.set_ylabel(target)
    st.pyplot(fig)


def plot_correlation(df, corr_type, plot_type):
    """상관관계를 산점도 또는 히트맵으로 그립니다."""
    numeric_df = df.select_dtypes(include=['number']).copy()
    
    corr_matrix, max_corr, min_corr = calculate_correlation(numeric_df)
    
    st.header(f"🔗 상관관계 분석 결과 ({plot_type})")
    
    if plot_type == '히트맵':
        # 1. 히트맵 시각화
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap='coolwarm', 
            cbar=True,
            linewidths=0.5,
            linecolor='black',
            ax=ax
        )
        ax.set_title("속성 간 상관관계 히트맵")
        st.pyplot(fig)
        
        # 2. 강한 상관관계 출력
        if corr_type == '양의 상관관계':
            if not max_corr.empty:
                pair = max_corr.index[0]
                value = max_corr.values[0]
                st.success(f"📈 **가장 강한 양의 상관관계:** **{pair[0]}**와 **{pair[1]}** (계수: {value:.4f})")
        else: # 음의 상관관계
            if not min_corr.empty:
                pair = min_corr.index[0]
                value = min_corr.values[0]
                st.error(f"📉 **가장 강한 음의 상관관계:** **{pair[0]}**와 **{pair[1]}** (계수: {value:.4f})")

    elif plot_type == '산점도':
        # 산점도는 가장 강한 상관관계를 가진 변수 쌍에 대해서만 시각화
        if corr_type == '양의 상관관계':
            if max_corr.empty:
                st.warning("분석할 수 있는 양의 상관관계 쌍이 없습니다.")
                return
            pair = max_corr.index[0]
        else:
            if min_corr.empty:
                st.warning("분석할 수 있는 음의 상관관계 쌍이 없습니다.")
                return
            pair = min_corr.index[0]

        st.subheader(f"산점도: {pair[0]} vs {pair[1]}")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=pair[0], y=pair[1], data=df, ax=ax, hue='survived', palette='deep')
        ax.set_title(f"{pair[0]}와 {pair[1]}의 관계 (생존 여부 기준)")
        st.pyplot(fig)

def calculate_correlation(df):
    """상관 행렬을 계산하고 가장 강한 양/음의 상관관계를 추출합니다."""
    corr_matrix = df.corr()
    np.fill_diagonal(corr_matrix.values, float('nan'))
    corr_unstacked = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
    valid_corr = corr_unstacked.dropna()
    max_corr = valid_corr.head(1)
    min_corr = valid_corr.tail(1)
    return corr_matrix, max_corr, min_corr

# --- 메인 앱 로직 ---
def main():
    
    data = load_data(FILE_PATH)
    if data is None:
        return

    # ------------------
    # 1. 사이드바 메뉴 구성
    # ------------------

    st.sidebar.title("메뉴 선택")
    
    # 1단계: 메인 그래프 선택
    graph_type = st.sidebar.radio(
        "📊 그래프 유형 선택",
        ('사망자/구조자 수 분석', '상관관계 분석')
    )
    
    st.sidebar.markdown("---")
    
    if graph_type == '사망자/구조자 수 분석':
        
        # 2단계: 분석 주제 (사망자 수 또는 구조자 수)
        analysis_theme = st.sidebar.radio(
            "🔎 분석 주제 선택",
            {
                "사망자 수": "사망자 수 (막대 그래프)",
                "구조된 사람 수": "구조된 사람 수 (막대 그래프)"
            },
            format_func=lambda x: x 
        )

        # 3단계: 세부 카테고리 선택
        if "사망자 수" in analysis_theme:
            category_options = {
                '사망자 수': 'survived_0', # 전체 사망자 수 (일단 미사용)
                '연령별 사망자 수': 'age',
                '클래스별 사망자 수': 'pclass'
            }
            target_col = 1 - data['survived'] # 0: 사망, 1: 생존. 타겟을 0으로 바꿔서 사망자 수로 계산
            data['Death'] = target_col
            default_key = '연령별 사망자 수'
            target_name = 'Death'
        else: # 구조된 사람 수
            category_options = {
                '구조된 사람 수': 'survived_1', # 전체 구조된 사람 수 (일단 미사용)
                '연령별 구조된 사람 수': 'age',
                '클래스별 구조된 사람 수': 'pclass'
            }
            data['Survival'] = data['survived']
            default_key = '연령별 구조된 사람 수'
            target_name = 'Survival'
            
        selected_category_name = st.sidebar.selectbox(
            f"세부 {analysis_theme} 카테고리",
            options=list(category_options.keys())[1:], # 전체 수 제외
            index=0
        )
        selected_category_col = category_options[selected_category_name]
        
        st.sidebar.markdown("---")
        
        # 4단계: 시각화 유형 선택 (맨 오른쪽 요구사항)
        plot_style = st.sidebar.radio(
            "📈 시각화 유형 선택",
            ('막대 그래프', '꺾은선 그래프')
        )
        
        # 메인 화면 출력
        if selected_category_name in ['연령별 사망자 수', '클래스별 사망자 수', '연령별 구조된 사람 수', '클래스별 구조된 사람 수']:
            plot_counts(data, selected_category_col, target_name, plot_style)


    elif graph_type == '상관관계 분석':
        
        # 2단계: 양/음의 상관관계 선택 (맨 아래 요구사항)
        corr_type = st.sidebar.radio(
            "🔗 상관관계 방향 선택",
            ('양의 상관관계', '음의 상관관계')
        )
        
        st.sidebar.markdown("---")
        
        # 3단계: 시각화 유형 선택 (맨 오른쪽 요구사항)
        corr_plot_type = st.sidebar.radio(
            "📊 시각화 유형 선택",
            ('히트맵', '산점도')
        )
        
        # 메인 화면 출력
        plot_correlation(data, corr_type, corr_plot_type)
        
        
if __name__ == "__main__":
    main()
