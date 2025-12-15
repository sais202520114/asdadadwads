import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm 

# 사용자님이 요청하신 파일명으로 정확히 설정
FILE_PATH = "titanic.xls"

# --- Matplotlib 한글 폰트 설정 (최종, 보수적 방식 유지) ---
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

# 시스템에 설치된 나눔고딕 폰트 검색 및 설정
font_name = None
for font_path in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
    font_prop = fm.FontProperties(fname=font_path)
    # 나눔고딕이 있다면 최우선으로 사용
    if 'NanumGothic' in font_prop.get_name():
        font_name = font_prop.get_name()
        break

# 만약 나눔고딕을 찾지 못했다면, 다른 흔한 폰트 시도 (Mac/Windows)
if not font_name:
    preferred_fonts = ['Malgun Gothic', 'AppleGothic', 'sans-serif']
    for p_font in preferred_fonts:
        if p_font == 'Malgun Gothic' and 'C:/Windows/Fonts/malgun.ttf' in fm.findSystemFonts(fontext='ttf'):
             font_name = 'Malgun Gothic'
             break
        if p_font == 'AppleGothic':
             font_name = 'AppleGothic'
             break
        if p_font == 'sans-serif':
             font_name = 'sans-serif'

if font_name:
    plt.rcParams['font.family'] = font_name
else:
    # 모든 시도가 실패하면 경고 메시지 출력
    plt.rcParams['font.family'] = 'sans-serif'
    st.warning("경고: 시스템에서 '나눔고딕', '맑은 고딕' 등 적절한 한글 폰트를 찾을 수 없습니다. 그래프의 한글이 깨질 수 있습니다. 나눔 폰트를 설치해 보세요.")


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
        # xlrd 관련 오류 및 파일 오류 메시지
        st.error(f"오류: 파일을 찾을 수 없거나 엑셀 로드 라이브러리('xlrd')가 설치되지 않았습니다. 파일 경로('{file_path}')와 requirements.txt를 확인해 주세요.")
        return None
    
    # 분석에 필요한 열 선택
    df_clean = df[['pclass', 'survived', 'sex', 'age', 'fare']].copy()

    # 'pclass', 'survived'는 수치형이지만 범주형으로 사용하기 위해 정수형 변환
    df_clean['pclass'] = df_clean['pclass'].fillna(df_clean['pclass'].mode()[0]).astype(int)
    df_clean['survived'] = df_clean['survived'].fillna(0).astype(int)
    
    # 'age'와 'fare' 결측값은 중앙값으로 대체
    df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())
    df_clean['fare'] = df_clean['fare'].fillna(df_clean['fare'].median())
    
    # Age Group 생성
    bins = [0, 10, 20, 30, 40, 50, 60, 100]
    labels = ['0-10대', '10-20대', '20-30대', '30-40대', '40-50대', '50-60대', '60대 이상']
    df_clean['age_group'] = pd.cut(df_clean['age'], bins=bins, labels=labels, right=False)

    # 분석에 필요한 타겟 열 생성
    df_clean['Death'] = 1 - df_clean['survived'] # 사망자 (0:생존, 1:사망)
    df_clean['Survival'] = df_clean['survived'] # 구조자 (0:사망, 1:생존)
    
    return df_clean

# --- 요약 표 출력 함수 ---
def generate_summary_tables(df):
    st.title("타이타닉 데이터 분석 종합 요약 표")
    st.markdown(f"**분석 데이터 파일명:** `{FILE_PATH}`")
    st.markdown("---")
    
    # 1. 사망자 요약
    total_deaths = df['Death'].sum()
    st.header(f"💔 총 사망자 수: {total_deaths}명")
    st.subheader("사망자 세부 분석 표")
    
    col_d1, col_d2 = st.columns(2)
    
    # 연령별 사망자 표
    age_death_summary = df.groupby('age_group')['Death'].sum().reset_index()
    age_death_summary = age_death_summary.rename(columns={'age_group': '연령대', 'Death': '사망자 수'})
    with col_d1:
        st.caption("연령별 사망자 수")
        st.dataframe(age_death_summary.set_index('연령대'))
        
    # 클래스별 사망자 표
    class_death_summary = df.groupby('pclass')['Death'].sum().reset_index()
    class_death_summary = class_death_summary.rename(columns={'pclass': '선실 등급', 'Death': '사망자 수'})
    class_death_summary['선실 등급'] = class_death_summary['선실 등급'].astype(str) + '등급'
    with col_d2:
        st.caption("선실 등급별 사망자 수")
        st.dataframe(class_death_summary.set_index('선실 등급'))

    st.markdown("---")

    # 2. 구조자 요약
    total_survival = df['Survival'].sum()
    st.header(f"✅ 총 구조된 사람 수: {total_survival}명")
    st.subheader("구조자 세부 분석 표")
    
    col_s1, col_s2 = st.columns(2)

    # 연령별 구조자 표
    age_survival_summary = df.groupby('age_group')['Survival'].sum().reset_index()
    age_survival_summary = age_survival_summary.rename(columns={'age_group': '연령대', 'Survival': '구조자 수'})
    with col_s1:
        st.caption("연령별 구조자 수")
        st.dataframe(age_survival_summary.set_index('연령대'))
        
    # 클래스별 구조자 표
    class_survival_summary = df.groupby('pclass')['Survival'].sum().reset_index()
    class_survival_summary = class_survival_summary.rename(columns={'pclass': '선실 등급', 'Survival': '구조자 수'})
    class_survival_summary['선실 등급'] = class_survival_summary['선실 등급'].astype(str) + '등급'
    with col_s2:
        st.caption("선실 등급별 구조자 수")
        st.dataframe(class_survival_summary.set_index('선실 등급'))
    
    st.markdown("---")

# --- 시각화 함수 ---

def plot_counts(df, category, target, target_name_kor, plot_type, extreme_select):
    """사망/구조자 수를 막대 또는 꺾은선 그래프로 그립니다."""
    
    # 데이터 준비
    if category == 'age':
        plot_data = df.groupby('age_group')[target].sum().reset_index()
        x_col = 'age_group'
        x_label_kor = '연령대'
    else: # pclass
        plot_data = df.groupby(category)[target].sum().reset_index()
        x_col = category
        x_label_kor = '선실 등급'
        # pclass를 한글 레이블로 변환 (그래프용)
        plot_data[x_col] = plot_data[x_col].astype(str).replace({'1': '1', '2': '2', '3': '3'}) + '등급'

    # 총합계 출력
    total_sum = plot_data[target].sum()
    st.info(f"**{x_label_kor}별 {target_name_kor} 총 합계:** `{total_sum}`명")
    
    st.subheader(f"📊 {target_name_kor} ({x_label_kor}별)")

    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 1. 그래프 그리기
    if plot_type == '막대 그래프':
        # 요청하신 대로 청량하고 예쁜 파란색 그라데이션 ('YlGnBu') 적용
        sns.barplot(x=x_col, y=target, data=plot_data, ax=ax, palette='YlGnBu', errorbar=None)
        
        # 막대 위에 숫자 출력
        for p in ax.patches:
            # 막대 그래프 높이에 숫자 표시
            ax.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 5), 
                        textcoords='offset points', 
                        fontsize=10)
            
    elif plot_type == '꺾은선 그래프':
        # 꺾은선 그래프는 선명한 파란색 단일 색상으로 지정 (가독성 고려)
        sns.lineplot(x=x_col, y=target, data=plot_data, ax=ax, marker='o', color='blue')
        
        # 점 위에 숫자 출력
        for x, y in zip(plot_data[x_col], plot_data[target]):
            ax.annotate(f'{int(y)}', (x, y), 
                        textcoords="offset points", 
                        xytext=(0, 10), 
                        ha='center', 
                        fontsize=10)
        
    # 2. 그래프 제목 및 라벨 (한글 설정)
    ax.set_title(f"{x_label_kor}별 {target_name_kor} ({plot_type})", fontsize=15)
    ax.set_xlabel(x_label_kor)
    ax.set_ylabel(target_name_kor)
    st.pyplot(fig) 
    
    # 3. 최대/최소 지점 출력
    max_val = plot_data[target].max()
    min_val = plot_data[target].min()
    
    # 지점 선택에 따라 결과 출력
    if extreme_select == '가장 높은 지점':
        extreme_data = plot_data[plot_data[target] == max_val]
        extreme_label = '가장 높은 지점'
        st.success(f"🥇 **{extreme_label}:** {extreme_data[x_col].iloc[0]} ({max_val}명)")
    else:
        extreme_data = plot_data[plot_data[target] == min_val]
        extreme_label = '가장 낮은 지점'
        st.error(f"🥉 **{extreme_label}:** {extreme_data[x_col].iloc[0]} ({min_val}명)")


def plot_correlation(df, corr_type, plot_type):
    """상관관계를 산점도 또는 히트맵으로 그립니다."""
    numeric_df = df.select_dtypes(include=['number']).copy()
    
    corr_matrix, max_corr, min_corr = calculate_correlation(numeric_df)
    
    st.header(f"🔗 상관관계 분석 결과 ({plot_type})")
    
    if plot_type == '히트맵':
        # 1. 히트맵 시각화 (청량한 파란색 그라데이션 'YlGnBu' 적용)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap='YlGnBu', # 색상 변경
            cbar=True,
            linewidths=0.5,
            linecolor='black',
            ax=ax
        )
        ax.set_title("타이타닉 속성 간 상관관계 히트맵", fontsize=15)
        st.pyplot(fig) 
        
        # 2. 강한 상관관계 출력
        if corr_type == '양의 상관관계':
            if not max_corr.empty:
                pair = max_corr.index[0]
                value = max_corr.values[0]
                st.success(f"📈 **가장 강한 양의 상관관계:** **{pair[0]}**와 **{pair[1]}** (계수: {value:.4f})")
            else:
                st.warning("분석할 수 있는 유효한 양의 상관관계 쌍이 없습니다.")
        else: # 음의 상관관계
            if not min_corr.empty:
                pair = min_corr.index[0]
                value = min_corr.values[0]
                st.error(f"📉 **가장 강한 음의 상관관계:** **{pair[0]}**와 **{pair[1]}** (계수: {value:.4f})")
            else:
                st.warning("분석할 수 있는 유효한 음의 상관관계 쌍이 없습니다.")

    elif plot_type == '산점도':
        # 산점도는 가장 강한 상관관계를 가진 변수 쌍에 대해서만 시각화
        if corr_type == '양의 상관관계':
            if max_corr.empty:
                st.warning("분석할 수 있는 양의 상관관계 쌍이 없습니다.")
                return
            pair = max_corr.index[0]
            title_prefix = "양의 상관관계"
        else:
            if min_corr.empty:
                st.warning("분석할 수 있는 음의 상관관계 쌍이 없습니다.")
                return
            pair = min_corr.index[0]
            title_prefix = "음의 상관관계"

        st.subheader(f"산점도: {title_prefix} - {pair[0]} vs {pair[1]}")
        fig, ax = plt.subplots(figsize=(8, 6))
        # 산점도는 hue를 기준으로 색을 나누기 때문에, 기본 'deep' 팔레트 유지
        sns.scatterplot(x=pair[0], y=pair[1], data=df, ax=ax, hue='survived', palette='deep') 
        ax.set_title(f"{pair[0]}와 {pair[1]}의 {title_prefix} 관계 (생존 여부 기준)", fontsize=15)
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
    
    # 1단계: 메인 그래프 선택 (표 메뉴 추가)
    graph_type = st.sidebar.radio(
        "📊 분석 유형 선택",
        ('종합 요약 (표)', '사망자/구조자 수 분석 (그래프)', '상관관계 분석 (그래프)')
    )
    
    st.sidebar.markdown("---")
    
    # ------------------
    # 2. 메인 화면 구성
    # ------------------
    
    if graph_type == '종합 요약 (표)':
        generate_summary_tables(data)

    elif graph_type == '사망자/구조자 수 분석 (그래프)':
        
        # 2단계: 분석 주제 (사망자 수 또는 구조자 수)
        analysis_theme_kor = st.sidebar.radio(
            "🔎 분석 주제 선택",
            ('사망자 수', '구조된 사람 수')
        )

        # 타겟 설정
        if analysis_theme_kor == '사망자 수':
            target_col = 'Death'
            target_name_kor = '사망자 수'
        else: # 구조된 사람 수
            target_col = 'Survival'
            target_name_kor = '구조된 사람 수'
            
        # 3단계: 세부 카테고리 선택
        category_options = {
            f'연령별 {target_name_kor}': 'age',
            f'클래스별 {target_name_kor}': 'pclass'
        }
            
        selected_category_name = st.sidebar.selectbox(
            f"세부 분류 카테고리",
            options=list(category_options.keys()),
            index=0
        )
        selected_category_col = category_options[selected_category_name]
        
        st.sidebar.markdown("---")
        
        # 4단계: 시각화 유형 선택 (막대/꺾은선)
        plot_style = st.sidebar.radio(
            "📈 시각화 유형 선택",
            ('막대 그래프', '꺾은선 그래프')
        )
        
        st.sidebar.markdown("---")

        # 5단계: 최대/최소 지점 선택 (기본: 가장 높은 지점)
        extreme_select = st.sidebar.radio(
            "⬆️ 지점 강조 선택",
            ('가장 높은 지점', '가장 낮은 지점'),
            index=0 # 기본적으로 가장 높은 지점을 출력
        )
        
        # 메인 화면 출력
        plot_counts(data, selected_category_col, target_col, target_name_kor, plot_style, extreme_select)


    elif graph_type == '상관관계 분석 (그래프)':
        
        # 2단계: 양/음의 상관관계 선택 (맨 아래 요구사항)
        corr_type = st.sidebar.radio(
            "🔗 상관관계 방향 선택",
            ('양의 상관관계', '음의 상관관계')
        )
        
        st.sidebar.markdown("---")
        
        # 3단계: 시각화 유형 선택 (산점도/히트맵)
        corr_plot_type = st.sidebar.radio(
            "📊 시각화 유형 선택",
            ('산점도', '히트맵')
        )
        
        # 메인 화면 출력
        plot_correlation(data, corr_type, corr_plot_type)
        
        
if __name__ == "__main__":
    main()
