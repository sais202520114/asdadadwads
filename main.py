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
    """엑셀 파일을 로드하고 초기 전처리를 수행합니다."""
    try:
        # xlrd 라이브러리가 설치되어 있어야 엑셀 파일을 로드할 수 있습니다.
        df = pd.read_excel(file_path)
    except Exception as e:
        # 오류 메시지를 사용자에게 표시하고 None을 반환
        st.error(f"오류: 파일 경로('{FILE_PATH}')를 확인하거나 'xlrd' 라이브러리를 설치해 주세요. ({e})")
        return None
    
    # 분석에 필요한 컬럼만 선택 (원본 load_data 함수와 일치)
    df_clean = df[['pclass', 'survived', 'sex', 'age', 'fare']].copy()

    return df_clean

# --- 결측치 처리 (중복 제거 및 통합) ---
def handle_missing_data(df):
    """결측치 처리 함수: mode/median으로 채우기"""
    # pclass (선실 등급) 결측치는 최빈값으로 채운 후 정수형으로 변환
    df['pclass'] = df['pclass'].fillna(df['pclass'].mode()[0]).astype(int)
    # survived (생존 여부) 결측치는 0 (사망)으로 채운 후 정수형으로 변환
    df['survived'] = df['survived'].fillna(0).astype(int)
    # age (나이) 결측치는 중앙값으로 채우기
    df['age'] = df['age'].fillna(df['age'].median())
    # fare (요금) 결측치는 중앙값으로 채우기
    df['fare'] = df['fare'].fillna(df['fare'].median())
    return df

# --- 이상치 처리 (IQR 방법) ---
def handle_outliers(df):
    """이상치 처리 함수 (IQR 방법): 이상치를 NaN으로 처리"""
    
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

# --- 보조 분석 컬럼 생성 (Death, Survival, age_group) ---
def create_analysis_columns(df):
    """분석에 필요한 추가 컬럼 (Death, Survival, age_group)을 생성합니다."""
    # Death (사망): survived가 0이면 1, 아니면 0
    df['Death'] = 1 - df['survived']
    # Survival (구조): survived와 동일
    df['Survival'] = df['survived']
    
    # age_group 생성: 0-10, 11-20, ..., 61-70, 71+
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 100]
    labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71+']
    # 'age'에 NaN이 있으면 'age_group'도 NaN이 되므로, 이상치/결측치 처리 후 수행해야 함
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=True, include_lowest=True)
    
    return df

# --- 정규화 ---
def normalize_data(df):
    """정규화 함수 (Min-Max Scaling)"""
    # 주의: 정규화는 이상치/결측치 처리 후에 수행해야 하며, age와 fare 컬럼에만 적용
    scaler = MinMaxScaler()
    # age와 fare 컬럼은 이상치 처리 후 NaN이 있을 수 있으므로, 이 시점에서는 NaN이 없어야 함
    # (main 함수에서 handle_missing_data를 다시 호출하여 NaN을 채워야 합니다.)
    df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])
    return df

# --- 박스 플롯 함수 ---
def plot_boxplot(df):
    """박스 플롯 시각화"""
    st.subheader("📊 박스 플롯: 나이 (Age)와 요금 (Fare)")
    
    # 이미 정규화된 데이터이므로, 스케일이 유사함.
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.boxplot(data=df[['age', 'fare']], ax=ax, palette="Set2")
    ax.set_title("Box Plot of Age and Fare (Normalized)", fontsize=14)
    ax.set_ylabel('Normalized Value', fontsize=12)
    
    st.pyplot(fig, use_container_width=True)

# --- 종합 요약에 총 인원 추가 ---
def generate_summary_tables(df):
    st.title("타이타닉 데이터 분석 종합 요약 표")
    st.markdown(f"**분석 데이터 파일:** `{FILE_PATH}`")
    st.markdown("---")
    
    total_people = len(df)
    
    # 'Death'와 'Survival' 컬럼이 이미 생성되어 있어야 합니다.
    total_deaths = df['Death'].sum()
    total_survival = df['Survival'].sum()
    
    # 데이터 유효성 검사
    if 'age_group' not in df.columns:
        st.error("오류: 'age_group' 컬럼이 데이터에 없습니다. 전처리 단계를 확인하세요.")
        return

    st.header(f"🚢 총 인원 수: {total_people}명")
    
    col_main1, col_main2 = st.columns(2)
    
    with col_main1:
        st.subheader(f"💔 총 사망자 수: {total_deaths}명")
        st.caption("사망자 세부 분석")
        
        # 연령별 사망자 수
        age_death_summary = df.groupby('age_group')['Death'].sum().reset_index()
        age_death_summary = age_death_summary.rename(columns={'age_group': '연령대 (Age Group)', 'Death': '사망자 수'})
        st.dataframe(age_death_summary.set_index('연령대 (Age Group)'))
            
        # 선실 등급별 사망자 수
        class_death_summary = df.groupby('pclass')['Death'].sum().reset_index()
        class_death_summary = class_death_summary.rename(columns={'pclass': '선실 등급', 'Death': '사망자 수'})
        class_death_summary['선실 등급'] = class_death_summary['선실 등급'].astype(str) + '등급'
        st.caption("선실 등급별 사망자 수")
        st.dataframe(class_death_summary.set_index('선실 등급'))

    with col_main2:
        st.subheader(f"✅ 총 구조된 사람 수: {total_survival}명")
        st.caption("구조자 세부 분석")

        # 연령별 구조자 수
        age_survival_summary = df.groupby('age_group')['Survival'].sum().reset_index()
        age_survival_summary = age_survival_summary.rename(columns={'age_group': '연령대 (Age Group)', 'Survival': '구조자 수'})
        st.dataframe(age_survival_summary.set_index('연령대 (Age Group)'))
            
        # 선실 등급별 구조자 수
        class_survival_summary = df.groupby('pclass')['Survival'].sum().reset_index()
        class_survival_summary = class_survival_summary.rename(columns={'pclass': '선실 등급', 'Survival': '구조자 수'})
        class_survival_summary['선실 등급'] = class_survival_summary['선실 등급'].astype(str) + '등급'
        st.caption("선실 등급별 구조자 수")
        st.dataframe(class_survival_summary.set_index('선실 등급'))
        
    st.markdown("---")

# --- 시각화 함수 ---
def plot_counts(df, category, target, target_name, plot_type, extreme_select):
    """사망/구조자 수를 막대 또는 꺾은선 그래프로 그립니다."""
    
    if 'age_group' not in df.columns:
        st.error("오류: 'age_group' 컬럼이 데이터에 없습니다. 전처리 단계를 확인하세요.")
        return

    # 데이터 집계
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

    # plt.figure() 중복 호출 제거, fig, ax만 사용
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
    
    # 가장 높은/낮은 지점 찾기
    max_val = plot_data[target].max()
    min_val = plot_data[target].min()
    
    if extreme_select == '가장 높은 지점':
        extreme_data = plot_data[plot_data[target] == max_val]
        extreme_label = '가장 높은 지점'
        # .iloc[0] 전에 .reset_index(drop=True)를 사용하면 인덱스 오류 방지
        st.success(f"🥇 **{extreme_label}:** {extreme_data.reset_index(drop=True)[x_col].iloc[0]} ({max_val})")
    else:
        extreme_data = plot_data[plot_data[target] == min_val]
        extreme_label = '가장 낮은 지점'
        st.error(f"🥉 **{extreme_label}:** {extreme_data.reset_index(drop=True)[x_col].iloc[0]} ({min_val})")

# --- 상관관계 분석 함수 수정 ---
def plot_correlation(df, corr_type, plot_type):
    """상관관계를 산점도 또는 히트맵으로 그립니다. (내부 라벨은 영어)"""
    
    # 상관 분석에서 연속형 변수만 사용
    # 주의: 'pclass'는 범주형이지만, 순서가 있는 등급이므로 분석에 포함 가능.
    # 하지만 여기서는 'survived', 'age', 'fare'만 사용하도록 유지합니다.
    numeric_df = df[['survived', 'age', 'fare']].copy() 
    
    # 데이터에 NaN이 있을 경우 상관계수 계산에 문제가 생길 수 있으므로 제거 (이상치 처리 후 재결측치 처리가 중요)
    numeric_df.dropna(inplace=True) 
    
    corr_matrix, max_corr, min_corr = calculate_correlation(numeric_df)
    
    st.header(f"🔗 상관관계 분석 결과 ({plot_type})")
    
    if plot_type == 'Heatmap':
        # 히트맵 시각화
        # plt.figure() 중복 호출 제거, fig, ax만 사용
        fig, ax = plt.subplots(figsize=(6, 6))
        
        col_names = ['Survived', 'Age', 'Fare']
        # 계산된 상관 행렬의 컬럼과 인덱스 이름 설정
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
                # 변수 이름이 튜플로 되어 있으므로 접근 방식 수정
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
        st.subheader(f"산점도: pclass별 연령과 요금 (Normalized)")
        
        # plt.figure() 중복 호출 제거, fig, ax만 사용
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # pclass를 문자형으로 변환하여 스타일링
        # df는 main에서 이미 수정된 버전이므로, 다시 변환하지 않고 사용합니다.
        # 단, 산점도를 위해 'pclass'를 문자열로 변환하는 코드가 필요합니다.
        df_plot = df.copy()
        df_plot['pclass_str'] = df_plot['pclass'].astype(str) 
        
        sns.scatterplot(x='age', y='fare', data=df_plot, hue='pclass_str', style='pclass_str', palette='deep', ax=ax, legend='full')
        
        ax.set_title(f"Scatter Plot: Age vs Fare (Grouped by Passenger Class)", fontsize=12)
        ax.set_xlabel('Age (Normalized)', fontsize=10)
        ax.set_ylabel('Fare (Normalized)', fontsize=10)
        
        st.pyplot(fig, use_container_width=False) 

def calculate_correlation(df):
    """상관 행렬을 계산하고 가장 강한 비자명 상관관계 쌍을 추출합니다."""
    # df에 NaN이 없다고 가정 (plot_correlation에서 dropna 처리)
    corr_matrix = df.corr()
    
    # 대각선 (자기 자신과의 상관관계)을 NaN으로 채워 분석에서 제외
    np.fill_diagonal(corr_matrix.values, np.nan) 
    
    # 딕셔너리처럼 풀어서 정렬 (중복 제거)
    corr_unstacked = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
    
    valid_corr = corr_unstacked.dropna()
    
    # 자기 자신과의 상관관계 (1.0) 또는 부동 소수점 오차로 인한 값 제거
    valid_corr = valid_corr[abs(valid_corr) < 0.999999] 

    # 가장 강한 양의 상관관계 (가장 큰 양수)
    max_corr = valid_corr[valid_corr > 0].head(1)
    # 가장 강한 음의 상관관계 (가장 작은 음수)
    min_corr = valid_corr[valid_corr < 0].tail(1)
    
    return corr_matrix, max_corr, min_corr

# --- 메인 앱 로직 ---
def main():
    # 1. 데이터 로드
    data = load_data(FILE_PATH)
    if data is None:
        return

    # 2. 전처리 1단계: 초기 결측치 처리 (load_data에서 제거됨)
    data = handle_missing_data(data)
    
    # 3. 이상치 처리 (NaN 생성)
    data = handle_outliers(data)
    
    # 4. 전처리 2단계: 이상치 처리로 인해 생긴 NaN 값 재처리
    # 중앙값으로 다시 채웁니다.
    data = handle_missing_data(data)
    
    # 5. 보조 분석 컬럼 생성 (Death, Survival, age_group)
    data = create_analysis_columns(data)
    
    # 6. 정규화
    data = normalize_data(data)

    st.sidebar.title("메뉴 선택")
    
    graph_type = st.sidebar.radio(
        "📊 분석 유형 선택",
        ('종합 요약 (표)', '사망/구조자 수 분석 (그래프)', '상관관계 분석 (그래프)', '박스 플롯')
    )
    
    st.sidebar.markdown("---")
    
    if graph_type == '종합 요약 (표)':
        # 'Death', 'Survival', 'age_group' 컬럼이 생성된 후 호출
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
        
        # 'age'와 'fare'에 NaN이 있으면 상관계수가 NaN이 되므로, 전처리가 완벽해야 함
        plot_correlation(data, corr_type_kor, corr_plot_type)
    
    elif graph_type == '박스 플롯':
        plot_boxplot(data)

if __name__ == "__main__":
    main()
