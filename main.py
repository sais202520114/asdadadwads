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

# --- 데이터 로드 및 전처리 함수 ---
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

# --- 상관관계 분석 함수 수정 ---
def plot_correlation(df, corr_type, plot_type):
    """상관관계를 산점도 또는 히트맵으로 그립니다. (내부 라벨은 영어)"""
    
    # 상관 분석에서 연속형 변수만 사용
    numeric_df = df[['survived', 'age', 'fare']].copy() 
    
    corr_matrix, max_corr, min_corr = calculate_correlation(numeric_df)
    
    st.header(f"🔗 상관관계 분석 결과 ({plot_type})")
    
    if plot_type == 'Heatmap':
        # 히트맵 시각화
        plt.figure(figsize=(6, 6))
        fig, ax = plt.subplots(figsize=(6, 6))
        
        col_names = ['Survived', 'Age', 'Fare']
        corr_matrix.columns = col_names
        corr_matrix.index = col_names
        
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap='YlGnBu', 
            cbar=True,
            linewidths=0.5,
            linecolor='black',
            annot_kws={"size": 9},
            ax=ax
        )
        ax.set_title("Correlation Heatmap of Titanic Attributes", fontsize=12)
        st.pyplot(fig, use_container_width=False) 
        
        # 강한 상관관계 출력
        if corr_type == '양의 상관관계':
            if not max_corr.empty:
                pair = max_corr.index[0]
                value = max_corr.values[0]
                st.success(f"📈 **가장 강한 양의 상관관계:** **{pair[0].capitalize()}**와 **{pair[1].capitalize()}** (계수: {value:.4f})")
            else:
                st.warning("분석할 수 있는 유효한 양의 상관관계 쌍이 없습니다.")
        else: # 음의 상관관계
            if not min_corr.empty:
                pair = min_corr.index[0]
                value = min_corr.values[0]
                st.error(f"📉 **가장 강한 음의 상관관계:** **{pair[0].capitalize()}**와 **{pair[1].capitalize()}** (계수: {value:.4f})")
            else:
                st.warning("분석할 수 있는 유효한 음의 상관관계 쌍이 없습니다.")

    elif plot_type == 'Scatter Plot':
        # pclass에 따라 산점도 그리기
        st.subheader(f"산점도: pclass별 연령과 요금")
        
        plt.figure(figsize=(6, 4))
        fig, ax = plt.subplots(figsize=(6, 4))
        
        sns.scatterplot(x='age', y='fare', data=df, hue='pclass', palette='deep', style='pclass', ax=ax, legend='full')
        
        ax.set_title(f"Scatter Plot: Age vs Fare (Grouped by Passenger Class)", fontsize=12)
        ax.set_xlabel('Age', fontsize=10)
        ax.set_ylabel('Fare', fontsize=10)
        
        st.pyplot(fig, use_container_width=False) 

def calculate_correlation(df):
    """상관 행렬을 계산하고 가장 강한 비자명 상관관계 쌍을 추출합니다."""
    corr_matrix = df.corr()
    
    np.fill_diagonal(corr_matrix.values, np.nan) 
    
    corr_unstacked = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
    
    valid_corr = corr_unstacked.dropna()
    
    valid_corr = valid_corr[abs(valid_corr) < 0.999999] 

    max_corr = valid_corr.head(1)
    min_corr = valid_corr.tail(1)
    
    return corr_matrix, max_corr, min_corr

# --- 메인 앱 로직 ---
def main():
    
    data = load_data(FILE_PATH)
    if data is None:
        return

    st.sidebar.title("메뉴 선택")
    
    graph_type = st.sidebar.radio(
        "📊 분석 유형 선택",
        ('종합 요약 (표)', '사망/구조자 수 분석 (그래프)', '상관관계 분석 (그래프)')
    )
    
    st.sidebar.markdown("---")
    
    if graph_type == '종합 요약 (표)':
        generate_summary_tables(data)

    elif graph_type == '사망/구조자 수 분석 (그래프)':
        
        analysis_theme_kor = st.sidebar.radio(
            "🔎 분석 주제 선택",
            ('사망자 수', '구조자 수')
        )

        if analysis_theme_kor == '사망자 수':
            target_col = 'Death'
            target_name = 'Death Count'
        else: 
            target_col = 'Survival'
            target_name = 'Survival Count'
            
        category_options = {
            f'연령별': 'age',
            f'선실 등급별': 'pclass'
        }
            
        selected_category_name = st.sidebar.selectbox(
            f"세부 분류 카테고리",
            options=list(category_options.keys()),
            index=0
        )
        selected_category_col = category_options[selected_category_name]
        
        st.sidebar.markdown("---")
        
        plot_style = st.sidebar.radio(
            "📈 시각화 유형 선택",
            ('Bar Chart', 'Line Chart')
        )
        
        st.sidebar.markdown("---")

        extreme_select_kor = st.sidebar.radio(
            "⬆️ 지점 강조 선택",
            ('가장 높은 지점', '가장 낮은 지점'),
            index=0 
        )
        
        plot_counts(data, selected_category_col, target_col, target_name, plot_style, extreme_select_kor)

    elif graph_type == '상관관계 분석 (그래프)':
        
        corr_type_kor = st.sidebar.radio(
            "🔗 상관관계 방향 선택",
            ('양의 상관관계', '음의 상관관계')
        )
        
        st.sidebar.markdown("---")
        
        corr_plot_type = st.sidebar.radio(
            "📊 시각화 유형 선택",
            ('Scatter Plot', 'Heatmap')
        )
        
        plot_correlation(data, corr_type_kor, corr_plot_type)

if __name__ == "__main__":
    main()
