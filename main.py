import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

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
    
    return df_clean

# --- 결측치 처리 ---
def handle_missing_data(df):
    """결측치 처리 함수"""
    df['pclass'] = df['pclass'].fillna(df['pclass'].mode()[0]).astype(int)
    df['survived'] = df['survived'].fillna(0).astype(int)
    df['age'] = df['age'].fillna(df['age'].median())
    df['fare'] = df['fare'].fillna(df['fare'].median())
    return df

# --- 이상치 처리 (IQR 방법) ---
def handle_outliers(df):
    """이상치 처리 함수 (IQR 방법)"""
    # 'age' 변수에 대한 이상치 처리
    Q1_age = df['age'].quantile(0.25)
    Q3_age = df['age'].quantile(0.75)
    IQR_age = Q3_age - Q1_age
    lower_bound_age = Q1_age - 1.5 * IQR_age
    upper_bound_age = Q3_age + 1.5 * IQR_age

    # 'fare' 변수에 대한 이상치 처리
    Q1_fare = df['fare'].quantile(0.25)
    Q3_fare = df['fare'].quantile(0.75)
    IQR_fare = Q3_fare - Q1_fare
    lower_bound_fare = Q1_fare - 1.5 * IQR_fare
    upper_bound_fare = Q3_fare + 1.5 * IQR_fare

    # 이상치 범위 밖의 데이터를 NaN 처리
    df['age'] = np.where((df['age'] < lower_bound_age) | (df['age'] > upper_bound_age), np.nan, df['age'])
    df['fare'] = np.where((df['fare'] < lower_bound_fare) | (df['fare'] > upper_bound_fare), np.nan, df['fare'])

    return df

# --- 정규화 ---
def normalize_data(df):
    """정규화 함수 (Min-Max Scaling)"""
    scaler = MinMaxScaler()
    df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])
    return df

# --- 박스 플롯 함수 ---
def plot_boxplot(df):
    """박스 플롯 시각화"""
    st.subheader("📊 박스 플롯: 나이 (Age)와 요금 (Fare)")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.boxplot(data=df[['age', 'fare']], ax=ax, palette="Set2")
    ax.set_title("Box Plot of Age and Fare", fontsize=14)
    ax.set_ylabel('Value', fontsize=12)
    
    st.pyplot(fig, use_container_width=True)

# --- 종합 요약에 총 인원 추가 ---
def generate_summary_tables(df):
    st.title("타이타닉 데이터 분석 종합 요약 표")
    st.markdown(f"**분석 데이터 파일:** `{FILE_PATH}`")
    st.markdown("---")
    
    total_people = len(df)
    total_deaths = df['Death'].sum()
    st.header(f"👥 총 인원 수: {total_people}명")
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

# --- 시각화 함수 ---
def plot_counts(df, category, target, target_name, plot_type, extreme_select):
    """사망/구조자 수를 막대 또는 꺾은선 그래프로 그립니다."""
    
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
        
        # pclass를 문자형으로 변환하여 스타일링
        df['pclass'] = df['pclass'].astype(str)  # pclass를 문자형으로 변환
        
        sns.scatterplot(x='age', y='fare', data=df, hue='pclass', style='pclass', palette='deep', ax=ax, legend='full')
        
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

    # 데이터 전처리 과정
    data = handle_missing_data(data)
    data = handle_outliers(data)
    data = normalize_data(data)

    st.sidebar.title("메뉴 선택")
    
    graph_type = st.sidebar.radio(
        "📊 분석 유형 선택",
        ('종합 요약 (표)', '사망/구조자 수 분석 (그래프)', '상관관계 분석 (그래프)', '박스 플롯')
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
    
    elif graph_type == '박스 플롯':
        plot_boxplot(data)

if __name__ == "__main__":
    main()
