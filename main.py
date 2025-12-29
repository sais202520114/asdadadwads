import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# --- 1. 환경 설정 ---
# 차트 내부는 기본 폰트를 사용하여 한글 깨짐 방지
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(
    page_title="타이타닉 데이터 분석 대시보드",
    layout="wide",
    initial_sidebar_state="expanded"
)

FILE_PATH = "titanic.xls"

# --- 2. 데이터 처리 함수 (오타 및 누락 기능 완벽 복구) ---
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_excel(file_path, engine='xlrd')
    except Exception:
        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            st.error(f"❌ 파일을 찾을 수 없습니다: {e}")
            return None
    return df[['pclass', 'survived', 'sex', 'age', 'fare']].copy()

def handle_missing_data(df):
    df = df.copy()
    df['pclass'] = df['pclass'].fillna(df['pclass'].mode()[0]).astype(int)
    df['survived'] = df['survived'].fillna(0).astype(int)
    df['age'] = df['age'].fillna(df['age'].median())
    df['fare'] = df['fare'].fillna(df['fare'].median())
    return df

def handle_outliers(df):
    df = df.copy()
    # 나이 이상치 (0~100세 범위를 벗어나는 데이터 처리)
    df['age'] = np.where((df['age'] < 0) | (df['age'] > 100), df['age'].median(), df['age'])
    # 요금(Fare) 이상치 처리 (IQR 기준)
    Q1_f = df['fare'].quantile(0.25)
    Q3_f = df['fare'].quantile(0.75)
    IQR_f = Q3_f - Q1_f
    df['fare'] = np.where((df['fare'] < (Q1_f - 1.5 * IQR_f)) | 
                          (df['fare'] > (Q3_f + 1.5 * IQR_f)), np.nan, df['fare'])
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
    # 결측치를 임시로 채워 스케일러 오류 방지
    temp_subset = df[['age', 'fare']].fillna(df[['age', 'fare']].median())
    df[['age', 'fare']] = scaler.fit_transform(temp_subset)
    return df

# --- 3. 시각화 및 분석 함수 ---
def generate_summary_tables(df_raw):
    st.title("🚢 타이타닉 데이터 분석 종합 요약")
    col1, col2, col3 = st.columns(3)
    col1.metric("총 인원", f"{len(df_raw)}명")
    col2.metric("총 사망자", f"{df_raw['Death'].sum()}명", delta="-사망", delta_color="inverse")
    col3.metric("총 구조자", f"{df_raw['Survival'].sum()}명", delta="+구조")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("💔 사망자 상세 요약")
        st.write("**연령대별**")
        st.dataframe(df_raw.groupby('age_group', observed=False)['Death'].sum(), use_container_width=True)
        st.write("**선실 등급별**")
        st.dataframe(df_raw.groupby('pclass')['Death'].sum(), use_container_width=True)
    with c2:
        st.subheader("✅ 구조자 상세 요약")
        st.write("**연령대별**")
        st.dataframe(df_raw.groupby('age_group', observed=False)['Survival'].sum(), use_container_width=True)
        st.write("**선실 등급별**")
        st.dataframe(df_raw.groupby('pclass')['Survival'].sum(), use_container_width=True)

def plot_counts(df_raw, category, target, target_name, plot_type, extreme_select):
    if category == 'age':
        plot_data = df_raw.groupby('age_group', observed=False)[target].sum().reset_index()
        x_col = 'age_group'
    else:
        plot_data = df_raw.groupby(category)[target].sum().reset_index()
        x_col = category
        plot_data[x_col] = "Class " + plot_data[x_col].astype(str)

    # 그래프 너비 제한 (화면의 약 60%)
    col_plot, _ = st.columns([1.5, 1])
    with col_plot:
        fig, ax = plt.subplots(figsize=(7, 4))
        if plot_type == 'Bar Chart':
            sns.barplot(x=x_col, y=target, data=plot_data, ax=ax, palette='viridis')
        else:
            sns.lineplot(x=x_col, y=target, data=plot_data, ax=ax, marker='o')
        ax.set_title(f"{target_name} Count by {category.capitalize()}", fontsize=12)
        st.pyplot(fig)

    if extreme_select == '가장 높은 지점':
        top = plot_data.loc[plot_data[target].idxmax()]
        st.success(f"🥇 최고치: {top[x_col]} ({top[target]}명)")
    else:
        bottom = plot_data.loc[plot_data[target].idxmin()]
        st.error(f"🥉 최저치: {bottom[x_col]} ({bottom[target]}명)")

def plot_correlation(df, corr_plot_type):
    st.subheader(f"🔗 상관관계 분석: {corr_plot_type}")
    col_corr, _ = st.columns([1.2, 1])
    with col_corr:
        fig, ax = plt.subplots(figsize=(6, 5))
        if corr_plot_type == 'Heatmap':
            sns.heatmap(df[['survived', 'age', 'fare']].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        else:
            sns.scatterplot(data=df, x='age', y='fare', hue='survived', alpha=0.6, ax=ax)
        ax.set_title(f"Correlation: {corr_plot_type}")
        st.pyplot(fig)

def plot_boxplot_with_stats(df_norm, df_raw):
    st.subheader("📊 박스 플롯 & 분위수 상세 분석")
    col_box, col_stat = st.columns([1.2, 1])
    with col_box:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df_norm[['age', 'fare']], ax=ax, palette="Set2")
        ax.set_title("Normalized Age & Fare Box Plot")
        st.pyplot(fig)
    with col_stat:
        st.write("**통계 상세 분석 (Quantiles)**")
        for col in ['age', 'fare']:
            q1, med, q3 = df_raw[col].quantile(0.25), df_raw[col].median(), df_raw[col].quantile(0.75)
            st.info(f"**{col.upper()}**\n\nQ1: {q1:.1f} | Median: {med:.1f} | Q3: {q3:.1f}")

# --- 4. 메인 실행 ---
def main():
    data = load_data(FILE_PATH)
    if data is None: return

    # 데이터 가공
    data_raw = create_analysis_columns(handle_missing_data(data))
    data_norm = normalize_data(handle_outliers(data_raw))

    st.sidebar.title("🔍 분석 메뉴")
    menu = st.sidebar.radio("항목 선택", ['종합 요약 (표)', '사망/구조자 분석 (그래프)', '상관관계 분석', '박스 플롯'])

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
        plot_boxplot_with_stats(data_norm, data_raw)

if __name__ == "__main__":
    main()
