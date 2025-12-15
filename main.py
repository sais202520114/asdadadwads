import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm 

# 사용자님이 요청하신 파일명으로 정확히 설정
FILE_PATH = "titanic.xls"

# =========================================================
# --- Matplotlib 한글 폰트 설정 (가장 확실한 안정화 코드) ---
# =========================================================
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

# 1. 시스템 폰트 목록에서 가장 안정적인 한글 폰트를 찾습니다.
font_name = None
preferred_fonts = ['AppleGothic', 'Malgun Gothic', 'NanumGothic']

for font_prop in [fm.FontProperties(fname=font_path) for font_path in fm.findSystemFonts(fontext='ttf')]:
    name = font_prop.get_name()
    if name in preferred_fonts:
        font_name = name
        break
    if 'Malgun' in name:
        font_name = 'Malgun Gothic'
        break

# 2. 폰트를 설정합니다.
if font_name:
    plt.rcParams['font.family'] = font_name
    st.info(f"사용된 한글 폰트: {font_name} (깨짐 방지 설정)")
else:
    # 3. 폰트를 찾지 못했을 경우, 스트림릿 환경에서 비교적 안전한 폰트 지정 및 경고
    plt.rcParams['font.family'] = 'sans-serif'
    st.warning("경고: 적절한 한글 폰트를 찾지 못했습니다. NanumGothic을 설치하거나, Streamlit 환경의 기본 'sans-serif'를 사용합니다.")
    # 폰트 캐시를 지워서 재시도하는 코드는 Streamlit 환경에서 보안 문제로 작동하지 않을 수 있으므로 제거함


# Streamlit 페이지 설정
st.set_page_config(
    page_title="타이타닉 데이터 분석 대시보드",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 데이터 로드 및 전처리 함수 (변경 없음) ---
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_excel(file_path)
    except Exception:
        st.error(f"오류: 파일을 찾을 수 없거나 엑셀 로드 라이브러리('xlrd')가 설치되지 않았습니다. 파일 경로('{file_path}')와 requirements.txt를 확인해 주세요.")
        return None
    
    df_clean = df[['pclass', 'survived', 'sex', 'age', 'fare']].copy()
    df_clean['pclass'] = df_clean['pclass'].fillna(df_clean['pclass'].mode()[0]).astype(int)
    df_clean['survived'] = df_clean['survived'].fillna(0).astype(int)
    df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())
    df_clean['fare'] = df_clean['fare'].fillna(df_clean['fare'].median())
    
    bins = [0, 10, 20, 30, 40, 50, 60, 100]
    labels = ['0-10대', '10-20대', '20-30대', '30-40대', '40-50대', '50-60대', '60대 이상']
    df_clean['age_group'] = pd.cut(df_clean['age'], bins=bins, labels=labels, right=False)

    df_clean['Death'] = 1 - df_clean['survived']
    df_clean['Survival'] = df_clean['survived']
    
    return df_clean

# --- 요약 표 출력 함수 (변경 없음) ---
def generate_summary_tables(df):
    st.title("타이타닉 데이터 분석 종합 요약 표")
    st.markdown(f"**분석 데이터 파일명:** `{FILE_PATH}`")
    st.markdown("---")
    
    total_deaths = df['Death'].sum()
    st.header(f"💔 총 사망자 수: {total_deaths}명")
    st.subheader("사망자 세부 분석 표")
    
    col_d1, col_d2 = st.columns(2)
    
    age_death_summary = df.groupby('age_group')['Death'].sum().reset_index()
    age_death_summary = age_death_summary.rename(columns={'age_group': '연령대', 'Death': '사망자 수'})
    with col_d1:
        st.caption("연령별 사망자 수")
        st.dataframe(age_death_summary.set_index('연령대'))
        
    class_death_summary = df.groupby('pclass')['Death'].sum().reset_index()
    class_death_summary = class_death_summary.rename(columns={'pclass': '선실 등급', 'Death': '사망자 수'})
    class_death_summary['선실 등급'] = class_death_summary['선실 등급'].astype(str) + '등급'
    with col_d2:
        st.caption("선실 등급별 사망자 수")
        st.dataframe(class_death_summary.set_index('선실 등급'))

    st.markdown("---")

    total_survival = df['Survival'].sum()
    st.header(f"✅ 총 구조된 사람 수: {total_survival}명")
    st.subheader("구조자 세부 분석 표")
    
    col_s1, col_s2 = st.columns(2)

    age_survival_summary = df.groupby('age_group')['Survival'].sum().reset_index()
    age_survival_summary = age_survival_summary.rename(columns={'age_group': '연령대', 'Survival': '구조자 수'})
    with col_s1:
        st.caption("연령별 구조자 수")
        st.dataframe(age_survival_summary.set_index('연령대'))
        
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
    
    if category == 'age':
        plot_data = df.groupby('age_group')[target].sum().reset_index()
        x_col = 'age_group'
        x_label_kor = '연령대'
    else: # pclass
        plot_data = df.groupby(category)[target].sum().reset_index()
        x_col = category
        x_label_kor = '선실 등급'
        plot_data[x_col] = plot_data[x_col].astype(str) + '등급'

    total_sum = plot_data[target].sum()
    st.info(f"**{x_label_kor}별 {target_name_kor} 총 합계:** `{total_sum}`명")
    
    st.subheader(f"📊 {target_name_kor} ({x_label_kor}별)")

    # === 그래프 크기 수정: (6, 4)로 최소화 ===
    fig, ax = plt.subplots(figsize=(6, 4))
    
    if plot_type == '막대 그래프':
        sns.barplot(x=x_col, y=target, data=plot_data, ax=ax, palette='YlGnBu', errorbar=None)
        
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 5), 
                        textcoords='offset points', 
                        fontsize=8) # 폰트 크기 조정
            
    elif plot_type == '꺾은선 그래프':
        sns.lineplot(x=x_col, y=target, data=plot_data, ax=ax, marker='o', color='blue')
        
        for x, y in zip(plot_data[x_col], plot_data[target]):
            ax.annotate(f'{int(y)}', (x, y), 
                        textcoords="offset points", 
                        xytext=(0, 8), 
                        ha='center', 
                        fontsize=8) # 폰트 크기 조정
        
    ax.set_title(f"{x_label_kor}별 {target_name_kor} ({plot_type})", fontsize=12)
    ax.set_xlabel(x_label_kor, fontsize=10)
    ax.set_ylabel(target_name_kor, fontsize=10)
    st.pyplot(fig) 
    
    # 3. 최대/최소 지점 출력
    max_val = plot_data[target].max()
    min_val = plot_data[target].min()
    
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
    
    numeric_df = df[['survived', 'pclass', 'age', 'fare']].copy()
    
    corr_matrix, max_corr, min_corr = calculate_correlation(numeric_df)
    
    st.header(f"🔗 상관관계 분석 결과 ({plot_type})")
    
    if plot_type == '히트맵':
        # 1. 히트맵 시각화
        # === 그래프 크기 수정: (6, 6)으로 최소화 ===
        fig, ax = plt.subplots(figsize=(6, 6))
        
        col_names = ['생존 여부', '선실 등급', '나이', '운임']
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
            annot_kws={"size": 9}, # 주석 폰트 크기 조정
            ax=ax
        )
        ax.set_title("타이타닉 속성 간 상관관계 히트맵", fontsize=12)
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
        # 산점도 시각화
        
        if corr_type == '양의 상관관계':
            if not max_corr.empty:
                pair = max_corr.index[0]
                x_var, y_var = pair[0], pair[1]
                title_prefix = "가장 강한 양의 상관관계"
            else:
                # Fallback: 운임 (Fare)과 나이 (Age)는 보통 양의 상관관계
                x_var, y_var = 'fare', 'age'
                title_prefix = "양의 상관관계 (대체: 운임 vs 나이)"

        else: # 음의 상관관계
            if not min_corr.empty:
                pair = min_corr.index[0]
                x_var, y_var = pair[0], pair[1]
                title_prefix = "가장 강한 음의 상관관계"
            else:
                # Fallback: 선실 등급 (Pclass)과 운임 (Fare)은 음의 상관관계
                x_var, y_var = 'pclass', 'fare'
                title_prefix = "음의 상관관계 (대체: 선실 등급 vs 운임)"

        st.subheader(f"산점도: {title_prefix} - {x_var} vs {y_var}")
        # === 그래프 크기 수정: (6, 4)로 최소화 ===
        fig, ax = plt.subplots(figsize=(6, 4))
        
        sns.scatterplot(x=x_var, y=y_var, data=df, ax=ax, hue='survived', palette='deep') 
        
        ax.set_title(f"{x_var}와 {y_var}의 {title_prefix} 관계 (생존 여부 기준)", fontsize=12)
        ax.set_xlabel(x_var, fontsize=10)
        ax.set_ylabel(y_var, fontsize=10)
        st.pyplot(fig) 

def calculate_correlation(df):
    """상관 행렬을 계산하고 가장 강한 양/음의 상관관계를 추출합니다. (1, -1만 나오는 문제 해결)"""
    corr_matrix = df.corr()
    
    # 대각선 값 (자기 자신과의 상관관계)을 NaN으로 명시적으로 채우기
    np.fill_diagonal(corr_matrix.values, np.nan) 
    
    corr_unstacked = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
    
    valid_corr = corr_unstacked.dropna()
    
    # === 1, -1에 가까운 값 필터링 완화 (0.999999 미만) ===
    valid_corr = valid_corr[abs(valid_corr) < 0.999999] 

    max_corr = valid_corr.head(1)
    min_corr = valid_corr.tail(1)
    
    # 필터링 후에도 값이 없으면 원본에서 추출 (매우 희박한 경우 대비)
    if max_corr.empty and not corr_unstacked.empty:
         max_corr = corr_unstacked.dropna().head(1)
         min_corr = corr_unstacked.dropna().tail(1)
    
    return corr_matrix, max_corr, min_corr

# --- 메인 앱 로직 (변경 없음) ---
def main():
    
    data = load_data(FILE_PATH)
    if data is None:
        return

    st.sidebar.title("메뉴 선택")
    
    graph_type = st.sidebar.radio(
        "📊 분석 유형 선택",
        ('종합 요약 (표)', '사망자/구조자 수 분석 (그래프)', '상관관계 분석 (그래프)')
    )
    
    st.sidebar.markdown("---")
    
    if graph_type == '종합 요약 (표)':
        generate_summary_tables(data)

    elif graph_type == '사망자/구조자 수 분석 (그래프)':
        
        analysis_theme_kor = st.sidebar.radio(
            "🔎 분석 주제 선택",
            ('사망자 수', '구조된 사람 수')
        )

        if analysis_theme_kor == '사망자 수':
            target_col = 'Death'
            target_name_kor = '사망자 수'
        else: 
            target_col = 'Survival'
            target_name_kor = '구조된 사람 수'
            
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
        
        plot_style = st.sidebar.radio(
            "📈 시각화 유형 선택",
            ('막대 그래프', '꺾은선 그래프')
        )
        
        st.sidebar.markdown("---")

        extreme_select = st.sidebar.radio(
            "⬆️ 지점 강조 선택",
            ('가장 높은 지점', '가장 낮은 지점'),
            index=0 
        )
        
        plot_counts(data, selected_category_col, target_col, target_name_kor, plot_style, extreme_select)


    elif graph_type == '상관관계 분석 (그래프)':
        
        corr_type = st.sidebar.radio(
            "🔗 상관관계 방향 선택",
            ('양의 상관관계', '음의 상관관계')
        )
        
        st.sidebar.markdown("---")
        
        corr_plot_type = st.sidebar.radio(
            "📊 시각화 유형 선택",
            ('산점도', '히트맵')
        )
        
        plot_correlation(data, corr_type, corr_plot_type)
        
        
if __name__ == "__main__":
    main()
