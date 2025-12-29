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
        df = pd.read_excel(file_path)
    except Exception as e:
        st.error(f"오류: 파일 경로('{FILE_PATH}')를 확인하거나 'xlrd' 라이브러리를 설치해 주세요. ({e})")
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
    df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])
    return df

# --- 박스 플롯 ---
def plot_boxplot(df):
    st.subheader("📊 박스 플롯: 나이 (Age)와 요금 (Fare)")
    fig, ax = plt.subplots(figsize=(4, 3), tight_layout=True)
    sns.boxplot(data=df[['age', 'fare']], ax=ax, palette="Set2")
    ax.set_title("Box Plot of Age and Fare (Normalized)", fontsize=10)
    ax.set_ylabel('Normalized Value', fontsize=8)
    st.pyplot(fig, use_container_width=False)

# --- 종합 요약 ---
def generate_summary_tables(df_raw):
    st.title("타이타닉 데이터 분석 종합 요약 표")
    st.markdown(f"**분석 데이터 파일:** {FILE_PATH}")
    st.markdown("---")
    total_people = len(df_raw)
    total_deaths = df_raw['Death'].sum()
    total_survival = df_raw['Survival'].sum()
    if 'age_group' not in df_raw.columns:
        st.error("오류: 'age_group' 컬럼이 데이터에 없습니다. 전처리 단계를 확인하세요.")
        return
    st.header(f"🚢 총 인원 수: {total_people}명")
    col_main1, col_main2 = st.columns(2)
    with col_main1:
        st.subheader(f"💔 총 사망자 수: {total_deaths}명")
        age_death_summary = df_raw.groupby('age_group')['Death'].sum().reset_index()
        age_death_summary = age_death_summary.rename(columns={'age_group': '연령대 (Age Group)', 'Death': '사망자 수'})
        st.dataframe(age_death_summary.set_index('연령대 (Age Group)'))
        class_death_summary = df_raw.groupby('pclass')['Death'].sum().reset_index()
        class_death_summary = class_death_summary.rename(columns={'pclass': '선실 등급', 'Death': '사망자 수'})
        class_death_summary['선실 등급'] = class_death_summary['선실 등급'].astype(str) + '등급'
        st.dataframe(class_death_summary.set_index('선실 등급'))
    with col_main2:
        st.subheader(f"✅ 총 구조된 사람 수: {total_survival}명")
        age_survival_summary = df_raw.groupby('age_group')['Survival'].sum().reset_index()
        age_survival_summary = age_survival_summary.rename(columns={'age_group': '연령대 (Age Group)', 'Survival': '구조자 수'})
        st.dataframe(age_survival_summary.set_index('연령대 (Age Group)'))
        class_survival_summary = df_raw.groupby('pclass')['Survival'].sum().reset_index()
        class_survival_summary = class_survival_summary.rename(columns={'pclass': '선실 등급', 'Survival': '구조자 수'})
        class_survival_summary['선실 등급'] = class_survival_summary['선실 등급'].astype(str) + '등급'
        st.dataframe(class_survival_summary.set_index('선실 등급'))
    st.markdown("---")

# --- 사망/구조자 수 분석 ---
def plot_counts(df_raw, category, target, target_name, plot_type, extreme_select):
    if 'age_group' not in df_raw.columns:
        st.error("오류: 'age_group' 컬럼이 데이터에 없습니다. 전처리 단계를 확인하세요.")
        return
    if category == 'age':
        plot_data = df_raw.groupby('age_group')[target].sum().reset_index()
        x_col = 'age_group'
        x_label = 'Age Group'
    else:
        plot_data = df_raw.groupby(category)[target].sum().reset_index()
        x_col = category
        x_label = 'Passenger Class'
        plot_data[x_col] = plot_data[x_col].astype(str) + ' Class'
    total_sum = plot_data[target].sum()
    st.info(f"**Total {target_name} Count by {x_label}:** {total_sum}")
    st.subheader(f"📊 {target_name} by {x_label}")
    y_max = plot_data[target].max()
    y_upper_limit = y_max * 1.10 if y_max > 0 else 10
    fig, ax = plt.subplots(figsize=(5, 3), tight_layout=True)
    if plot_type == 'Bar Chart':
        sns.barplot(x=x_col, y=target, data=plot_data, ax=ax, palette='YlGnBu', errorbar=None)
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width()/2., p.get_height()),
                        ha='center', va='center', xytext=(0,5), textcoords='offset points', fontsize=8)
    elif plot_type == 'Line Chart':
        sns.lineplot(x=x_col, y=target, data=plot_data, ax=ax, marker='o', color='blue')
        for x, y in zip(plot_data[x_col], plot_data[target]):
            ax.annotate(f'{int(y)}', (x, y), textcoords="offset points", xytext=(0,8), ha='center', fontsize=8)
    ax.set_title(f"{target_name} by {x_label} ({plot_type})", fontsize=10)
    ax.set_xlabel(x_label, fontsize=8)
    ax.set_ylabel(target_name, fontsize=8)
    ax.set_ylim(0, y_upper_limit)
    st.pyplot(fig, use_container_width=False)
    max_val = plot_data[target].max()
    min_val = plot_data[target].min()
    if extreme_select == '가장 높은 지점':
        extreme_data = plot_data[plot_data[target] == max_val]
        extreme_label = '가장 높은 지점'
        st.success(f"🥇 **{extreme_label}:** {extreme_data.reset_index(drop=True)[x_col].iloc[0]} ({max_val})")
    else:
        extreme_data = plot_data[plot_data[target] == min_val]
        extreme_label = '가장 낮은 지점'
        st.error(f"🥉 **{extreme_label}:** {extreme_data.reset_index(drop=True)[x_col].iloc[0]} ({min_val})")

# --- 상관관계 분석 ---
def plot_correlation(df, corr_type, plot_type):
    numeric_df = df[['survived', 'age', 'fare']].copy()
    numeric_df.dropna(inplace=True)
    corr_matrix, max_corr, min_corr = calculate_correlation(numeric_df)
    st.header(f"🔗 상관관계 분석 결과 ({plot_type})")
    if plot_type == 'Heatmap':
        fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
        col_names = ['Survived', 'Age', 'Fare']
        corr_matrix.columns = col_names
        corr_matrix.index = col_names
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='YlGnBu',
                    cbar=True, linewidths=0.5, linecolor='black', annot_kws={"size":9}, ax=ax)
        ax.set_title("Correlation Heatmap of Titanic Attributes", fontsize=10)
        st.pyplot(fig, use_container_width=False)
        if corr_type == '양의 상관관계':
            if not max_corr.empty:
                pair = max_corr.index[0]
                value = max_corr.values[0]
                st.success(f"📈 **가장 강한 양의 상관관계:** **{pair[0].capitalize()}**와 **{pair[1].capitalize()}** (계수: {value:.4f})")
            else:
                st.warning("분석할 수 있는 유효한 양의 상관관계 쌍이 없습니다.")
        else:
            if not min_corr.empty:
                pair = min_corr.index[0]
                value = min_corr.values[0]
                st.error(f"📉 **가장 강한 음의 상관관계:** **{pair[0].capitalize()}**와 **{pair[1].capitalize()}** (계수: {value:.4f})")
            else:
                st.warning("분석할 수 있는 유효한 음의 상관관계 쌍이 없습니다.")
    elif plot_type == 'Scatter Plot':
        st.subheader(f"산점도: pclass별 연령과 요금 (Normalized)")
        fig, ax = plt.subplots(figsize=(5, 3), tight_layout=True)
        df_plot = df.copy()
        df_plot['pclass_str'] = df_plot['pclass'].astype(str)
        sns.scatterplot(x='age', y='fare', data=df_plot, hue='pclass_str', style='pclass_str',
                        palette='deep', ax=ax, legend='full')
        ax.set_title(f"Scatter Plot: Age vs Fare (Grouped by Passenger Class)", fontsize=10)
        ax.set_xlabel('Age (Normalized)', fontsize=8)
        ax.set_ylabel('Fare (Normalized)', fontsize=8)
        st.pyplot(fig, use_container_width=False)

def calculate_correlation(df):
    corr_matrix = df.corr()
    np.fill_diagonal(corr_matrix.values, np.nan)
    corr_unstacked = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
    valid_corr = corr_unstacked.dropna()
    valid_corr = valid_corr[abs(valid_corr) < 0.999999]
    max_corr = valid_corr[valid_corr > 0].head(1)
    min_corr = valid_corr[valid_corr < 0].tail(1)
    return corr_matrix, max_corr, min_corr

# --- 분위수/이상치 ---
def analyze_quantiles_and_outliers(df_raw):
    st.markdown("---")
    st.header("📈 분위수 및 이상치 분석 결과")
    analysis_vars = ['age', 'fare']
    results = {}
    for var in analysis_vars:
        Q1 = df_raw[var].quantile(0.25)
        Q2 = df_raw[var].quantile(0.5)
        Q3 = df_raw[var].quantile(0.75)
        IQR = Q3 - Q1
        if var == 'age':
            outliers_count = len(df_raw[(df_raw[var].notna()) & ((df_raw[var]<0)|(df_raw[var]>100))])
            vip_count, general_count, upper_bound = None, None, None
        elif var == 'fare':
            lower_bound = Q1 - 1.5*IQR
            upper_bound = Q3 + 1.5*IQR
            vip_count = len(df_raw[(df_raw[var].notna()) & (df_raw[var] > upper_bound)])
            general_count = len(df_raw[df_raw[var].notna()]) - vip_count
            outliers_count = vip_count
        results[var] = {
            'Q1': Q1, 'Q2_Median': Q2, 'Q3': Q3, 'Outliers_Count': outliers_count,
            'Upper_Bound': upper_bound if var=='fare' else None,
            'VIP_Count': vip_count, 'General_Count': general_count
        }
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.subheader("나이 (Age) 분석")
        st.markdown(f"**1분위수 (Q1):** {results['age']['Q1']:.2f}")
        st.markdown(f"**2분위수 (중앙값, Q2):** {results['age']['Q2_Median']:.2f}")
        st.markdown(f"**3분위수 (Q3):** {results['age']['Q3']:.2f}")
        st.error(f"**❗ 처리된 이상치 개수 (0~100세 기준):** {results['age']['Outliers_Count']}개")
    with col_a2:
        st.subheader("요금 (Fare) 분석 (VIP/일반인 분류)")
        st.markdown(f"**1분위수 (Q1):** {results['fare']['Q1']:.2f}")
        st.markdown(f"**3분위수 (Q3):** {results['fare']['Q3']:.2f}")
        st.markdown(f"**기준 요금 (Upper Bound):** {results['fare']['Upper_Bound']:.2f}")
        st.markdown("---")
        st.success(f"**👑 VIP (기준 요금 초과) 인원:** {results['fare']['VIP_Count']}명")
        st.info(f"**🚶 일반인 (기준 요금 이하) 인원:** {results['fare']['General_Count']}명")
        st.markdown("---")

# --- 메인 앱 ---
def main():
    data = load_data(FILE_PATH)
    if data is None: return
    data_raw = handle_missing_data(data.copy())
    data_raw = create_analysis_columns(data_raw)
    data = handle_missing_data(data)
    data = handle_outliers(data)
    data = handle_missing_data(data)
    data = create_analysis_columns(data)
    data = normalize_data(data)
    st.sidebar.title("메뉴 선택")
    graph_type = st.sidebar.radio(
        "📊 분석 유형 선택",
        ('종합 요약 (표)', '사망/구조자 수 분석 (그래프)', '상관관계 분석 (그래프)', '박스 플롯')
    )
    st.sidebar.markdown("---")
    if graph_type == '종합 요약 (표)':
        generate_summary_tables(data_raw)
    elif graph_type == '사망/구조자 수 분석 (그래프)':
        analysis_theme_kor = st.sidebar.radio("🔎 분석 주제 선택", ('사망자 수', '구조자 수'))
        target_col = 'Death' if analysis_theme_kor=='사망자 수' else 'Survival'
        target_name = 'Death Count' if target_col=='Death' else 'Survival Count'
        category_options = {'연령별':'age', '선실 등급별':'pclass'}
        selected_category_name = st.sidebar.selectbox("세부 분류 카테고리", options=list(category_options.keys()), index=0)
        selected_category_col = category_options[selected_category_name]
        st.sidebar.markdown("---")
        plot_style = st.sidebar.radio("📈 시각화 유형 선택", ('Bar Chart', 'Line Chart'))
        st.sidebar.markdown("---")
        extreme_select_kor = st.sidebar.radio("⬆️ 지점 강조 선택", ('가장 높은 지점', '가장 낮은 지점'), index=0)
        plot_counts(data_raw, selected_category_col, target_col, target_name, plot_style, extreme_select_kor)
    elif graph_type == '상관관계 분석 (그래프)':
        corr_type_kor = st.sidebar.radio("🔗 상관관계 방향 선택", ('양의 상관관계', '음의 상관관계'))
        st.sidebar.markdown("---")
        corr_plot_type = st.sidebar.radio("📊 시각화 유형 선택", ('Scatter Plot', 'Heatmap'))
        plot_correlation(data, corr_type_kor, corr_plot_type)
    elif graph_type == '박스 플롯':
        plot_boxplot(data)
        analyze_quantiles_and_outliers(data_raw)

if __name__ == "__main__":
    main()import streamlit as st
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
        df = pd.read_excel(file_path)
    except Exception as e:
        st.error(f"오류: 파일 경로('{FILE_PATH}')를 확인하거나 'xlrd' 라이브러리를 설치해 주세요. ({e})")
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
    df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])
    return df

# --- 박스 플롯 ---
def plot_boxplot(df):
    st.subheader("📊 박스 플롯: 나이 (Age)와 요금 (Fare)")
    fig, ax = plt.subplots(figsize=(4, 3), tight_layout=True)
    sns.boxplot(data=df[['age', 'fare']], ax=ax, palette="Set2")
    ax.set_title("Box Plot of Age and Fare (Normalized)", fontsize=10)
    ax.set_ylabel('Normalized Value', fontsize=8)
    st.pyplot(fig, use_container_width=False)

# --- 종합 요약 ---
def generate_summary_tables(df_raw):
    st.title("타이타닉 데이터 분석 종합 요약 표")
    st.markdown(f"**분석 데이터 파일:** {FILE_PATH}")
    st.markdown("---")
    total_people = len(df_raw)
    total_deaths = df_raw['Death'].sum()
    total_survival = df_raw['Survival'].sum()
    if 'age_group' not in df_raw.columns:
        st.error("오류: 'age_group' 컬럼이 데이터에 없습니다. 전처리 단계를 확인하세요.")
        return
    st.header(f"🚢 총 인원 수: {total_people}명")
    col_main1, col_main2 = st.columns(2)
    with col_main1:
        st.subheader(f"💔 총 사망자 수: {total_deaths}명")
        age_death_summary = df_raw.groupby('age_group')['Death'].sum().reset_index()
        age_death_summary = age_death_summary.rename(columns={'age_group': '연령대 (Age Group)', 'Death': '사망자 수'})
        st.dataframe(age_death_summary.set_index('연령대 (Age Group)'))
        class_death_summary = df_raw.groupby('pclass')['Death'].sum().reset_index()
        class_death_summary = class_death_summary.rename(columns={'pclass': '선실 등급', 'Death': '사망자 수'})
        class_death_summary['선실 등급'] = class_death_summary['선실 등급'].astype(str) + '등급'
        st.dataframe(class_death_summary.set_index('선실 등급'))
    with col_main2:
        st.subheader(f"✅ 총 구조된 사람 수: {total_survival}명")
        age_survival_summary = df_raw.groupby('age_group')['Survival'].sum().reset_index()
        age_survival_summary = age_survival_summary.rename(columns={'age_group': '연령대 (Age Group)', 'Survival': '구조자 수'})
        st.dataframe(age_survival_summary.set_index('연령대 (Age Group)'))
        class_survival_summary = df_raw.groupby('pclass')['Survival'].sum().reset_index()
        class_survival_summary = class_survival_summary.rename(columns={'pclass': '선실 등급', 'Survival': '구조자 수'})
        class_survival_summary['선실 등급'] = class_survival_summary['선실 등급'].astype(str) + '등급'
        st.dataframe(class_survival_summary.set_index('선실 등급'))
    st.markdown("---")

# --- 사망/구조자 수 분석 ---
def plot_counts(df_raw, category, target, target_name, plot_type, extreme_select):
    if 'age_group' not in df_raw.columns:
        st.error("오류: 'age_group' 컬럼이 데이터에 없습니다. 전처리 단계를 확인하세요.")
        return
    if category == 'age':
        plot_data = df_raw.groupby('age_group')[target].sum().reset_index()
        x_col = 'age_group'
        x_label = 'Age Group'
    else:
        plot_data = df_raw.groupby(category)[target].sum().reset_index()
        x_col = category
        x_label = 'Passenger Class'
        plot_data[x_col] = plot_data[x_col].astype(str) + ' Class'
    total_sum = plot_data[target].sum()
    st.info(f"**Total {target_name} Count by {x_label}:** {total_sum}")
    st.subheader(f"📊 {target_name} by {x_label}")
    y_max = plot_data[target].max()
    y_upper_limit = y_max * 1.10 if y_max > 0 else 10
    fig, ax = plt.subplots(figsize=(5, 3), tight_layout=True)
    if plot_type == 'Bar Chart':
        sns.barplot(x=x_col, y=target, data=plot_data, ax=ax, palette='YlGnBu', errorbar=None)
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width()/2., p.get_height()),
                        ha='center', va='center', xytext=(0,5), textcoords='offset points', fontsize=8)
    elif plot_type == 'Line Chart':
        sns.lineplot(x=x_col, y=target, data=plot_data, ax=ax, marker='o', color='blue')
        for x, y in zip(plot_data[x_col], plot_data[target]):
            ax.annotate(f'{int(y)}', (x, y), textcoords="offset points", xytext=(0,8), ha='center', fontsize=8)
    ax.set_title(f"{target_name} by {x_label} ({plot_type})", fontsize=10)
    ax.set_xlabel(x_label, fontsize=8)
    ax.set_ylabel(target_name, fontsize=8)
    ax.set_ylim(0, y_upper_limit)
    st.pyplot(fig, use_container_width=False)
    max_val = plot_data[target].max()
    min_val = plot_data[target].min()
    if extreme_select == '가장 높은 지점':
        extreme_data = plot_data[plot_data[target] == max_val]
        extreme_label = '가장 높은 지점'
        st.success(f"🥇 **{extreme_label}:** {extreme_data.reset_index(drop=True)[x_col].iloc[0]} ({max_val})")
    else:
        extreme_data = plot_data[plot_data[target] == min_val]
        extreme_label = '가장 낮은 지점'
        st.error(f"🥉 **{extreme_label}:** {extreme_data.reset_index(drop=True)[x_col].iloc[0]} ({min_val})")

# --- 상관관계 분석 ---
def plot_correlation(df, corr_type, plot_type):
    numeric_df = df[['survived', 'age', 'fare']].copy()
    numeric_df.dropna(inplace=True)
    corr_matrix, max_corr, min_corr = calculate_correlation(numeric_df)
    st.header(f"🔗 상관관계 분석 결과 ({plot_type})")
    if plot_type == 'Heatmap':
        fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
        col_names = ['Survived', 'Age', 'Fare']
        corr_matrix.columns = col_names
        corr_matrix.index = col_names
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='YlGnBu',
                    cbar=True, linewidths=0.5, linecolor='black', annot_kws={"size":9}, ax=ax)
        ax.set_title("Correlation Heatmap of Titanic Attributes", fontsize=10)
        st.pyplot(fig, use_container_width=False)
        if corr_type == '양의 상관관계':
            if not max_corr.empty:
                pair = max_corr.index[0]
                value = max_corr.values[0]
                st.success(f"📈 **가장 강한 양의 상관관계:** **{pair[0].capitalize()}**와 **{pair[1].capitalize()}** (계수: {value:.4f})")
            else:
                st.warning("분석할 수 있는 유효한 양의 상관관계 쌍이 없습니다.")
        else:
            if not min_corr.empty:
                pair = min_corr.index[0]
                value = min_corr.values[0]
                st.error(f"📉 **가장 강한 음의 상관관계:** **{pair[0].capitalize()}**와 **{pair[1].capitalize()}** (계수: {value:.4f})")
            else:
                st.warning("분석할 수 있는 유효한 음의 상관관계 쌍이 없습니다.")
    elif plot_type == 'Scatter Plot':
        st.subheader(f"산점도: pclass별 연령과 요금 (Normalized)")
        fig, ax = plt.subplots(figsize=(5, 3), tight_layout=True)
        df_plot = df.copy()
        df_plot['pclass_str'] = df_plot['pclass'].astype(str)
        sns.scatterplot(x='age', y='fare', data=df_plot, hue='pclass_str', style='pclass_str',
                        palette='deep', ax=ax, legend='full')
        ax.set_title(f"Scatter Plot: Age vs Fare (Grouped by Passenger Class)", fontsize=10)
        ax.set_xlabel('Age (Normalized)', fontsize=8)
        ax.set_ylabel('Fare (Normalized)', fontsize=8)
        st.pyplot(fig, use_container_width=False)

def calculate_correlation(df):
    corr_matrix = df.corr()
    np.fill_diagonal(corr_matrix.values, np.nan)
    corr_unstacked = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
    valid_corr = corr_unstacked.dropna()
    valid_corr = valid_corr[abs(valid_corr) < 0.999999]
    max_corr = valid_corr[valid_corr > 0].head(1)
    min_corr = valid_corr[valid_corr < 0].tail(1)
    return corr_matrix, max_corr, min_corr

# --- 분위수/이상치 ---
def analyze_quantiles_and_outliers(df_raw):
    st.markdown("---")
    st.header("📈 분위수 및 이상치 분석 결과")
    analysis_vars = ['age', 'fare']
    results = {}
    for var in analysis_vars:
        Q1 = df_raw[var].quantile(0.25)
        Q2 = df_raw[var].quantile(0.5)
        Q3 = df_raw[var].quantile(0.75)
        IQR = Q3 - Q1
        if var == 'age':
            outliers_count = len(df_raw[(df_raw[var].notna()) & ((df_raw[var]<0)|(df_raw[var]>100))])
            vip_count, general_count, upper_bound = None, None, None
        elif var == 'fare':
            lower_bound = Q1 - 1.5*IQR
            upper_bound = Q3 + 1.5*IQR
            vip_count = len(df_raw[(df_raw[var].notna()) & (df_raw[var] > upper_bound)])
            general_count = len(df_raw[df_raw[var].notna()]) - vip_count
            outliers_count = vip_count
        results[var] = {
            'Q1': Q1, 'Q2_Median': Q2, 'Q3': Q3, 'Outliers_Count': outliers_count,
            'Upper_Bound': upper_bound if var=='fare' else None,
            'VIP_Count': vip_count, 'General_Count': general_count
        }
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.subheader("나이 (Age) 분석")
        st.markdown(f"**1분위수 (Q1):** {results['age']['Q1']:.2f}")
        st.markdown(f"**2분위수 (중앙값, Q2):** {results['age']['Q2_Median']:.2f}")
        st.markdown(f"**3분위수 (Q3):** {results['age']['Q3']:.2f}")
        st.error(f"**❗ 처리된 이상치 개수 (0~100세 기준):** {results['age']['Outliers_Count']}개")
    with col_a2:
        st.subheader("요금 (Fare) 분석 (VIP/일반인 분류)")
        st.markdown(f"**1분위수 (Q1):** {results['fare']['Q1']:.2f}")
        st.markdown(f"**3분위수 (Q3):** {results['fare']['Q3']:.2f}")
        st.markdown(f"**기준 요금 (Upper Bound):** {results['fare']['Upper_Bound']:.2f}")
        st.markdown("---")
        st.success(f"**👑 VIP (기준 요금 초과) 인원:** {results['fare']['VIP_Count']}명")
        st.info(f"**🚶 일반인 (기준 요금 이하) 인원:** {results['fare']['General_Count']}명")
        st.markdown("---")

# --- 메인 앱 ---
def main():
    data = load_data(FILE_PATH)
    if data is None: return
    data_raw = handle_missing_data(data.copy())
    data_raw = create_analysis_columns(data_raw)
    data = handle_missing_data(data)
    data = handle_outliers(data)
    data = handle_missing_data(data)
    data = create_analysis_columns(data)
    data = normalize_data(data)
    st.sidebar.title("메뉴 선택")
    graph_type = st.sidebar.radio(
        "📊 분석 유형 선택",
        ('종합 요약 (표)', '사망/구조자 수 분석 (그래프)', '상관관계 분석 (그래프)', '박스 플롯')
    )
    st.sidebar.markdown("---")
    if graph_type == '종합 요약 (표)':
        generate_summary_tables(data_raw)
    elif graph_type == '사망/구조자 수 분석 (그래프)':
        analysis_theme_kor = st.sidebar.radio("🔎 분석 주제 선택", ('사망자 수', '구조자 수'))
        target_col = 'Death' if analysis_theme_kor=='사망자 수' else 'Survival'
        target_name = 'Death Count' if target_col=='Death' else 'Survival Count'
        category_options = {'연령별':'age', '선실 등급별':'pclass'}
        selected_category_name = st.sidebar.selectbox("세부 분류 카테고리", options=list(category_options.keys()), index=0)
        selected_category_col = category_options[selected_category_name]
        st.sidebar.markdown("---")
        plot_style = st.sidebar.radio("📈 시각화 유형 선택", ('Bar Chart', 'Line Chart'))
        st.sidebar.markdown("---")
        extreme_select_kor = st.sidebar.radio("⬆️ 지점 강조 선택", ('가장 높은 지점', '가장 낮은 지점'), index=0)
        plot_counts(data_raw, selected_category_col, target_col, target_name, plot_style, extreme_select_kor)
    elif graph_type == '상관관계 분석 (그래프)':
        corr_type_kor = st.sidebar.radio("🔗 상관관계 방향 선택", ('양의 상관관계', '음의 상관관계'))
        st.sidebar.markdown("---")
        corr_plot_type = st.sidebar.radio("📊 시각화 유형 선택", ('Scatter Plot', 'Heatmap'))
        plot_correlation(data, corr_type_kor, corr_plot_type)
    elif graph_type == '박스 플롯':
        plot_boxplot(data)
        analyze_quantiles_and_outliers(data_raw)

if __name__ == "__main__":
    main()
