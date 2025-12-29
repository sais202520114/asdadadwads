import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

# --- 1. 환경 설정 ---
# 그래프 내 영어 사용으로 폰트 오류 방지
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(
    page_title="타이타닉 분석 대시보드",
    layout="wide",
    initial_sidebar_state="expanded"
)

FILE_PATH = "titanic.xls"

# --- 2. 데이터 처리 함수 ---
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_excel(file_path, engine='xlrd')
    except Exception:
        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            st.error(f"❌ 파일 로드 오류: {e}")
            return None
    return df[['pclass', 'survived', 'sex', 'age', 'fare']].copy()

def handle_data(df):
    df = df.copy()
    df['pclass'] = df['pclass'].fillna(df['pclass'].mode()[0]).astype(int)
    df['survived'] = df['survived'].fillna(0).astype(int)
    df['age'] = df['age'].fillna(df['age'].median())
    df['fare'] = df['fare'].fillna(df['fare'].median())
    df['Death'] = 1 - df['survived']
    df['Survival'] = df['survived']
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 100]
    labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=True, include_lowest=True)
    return df

def normalize_data(df):
    df = df.copy()
    scaler = MinMaxScaler()
    # 이상치 처리 후 정규화
    df['age'] = np.where((df['age'] < 0) | (df['age'] > 100), df['age'].median(), df['age'])
    subset = df[['age', 'fare']].fillna(df[['age', 'fare']].median())
    df[['age', 'fare']] = scaler.fit_transform(subset)
    return df

# --- 3. 시각화 함수 (figsize 축소) ---
def generate_summary_tables(df_raw):
    st.title("🚢 타이타닉 종합 요약")
    col1, col2, col3 = st.columns(3)
    col1.metric("총 인원", f"{len(df_raw)}명")
    col2.metric("총 사망자", f"{df_raw['Death'].sum()}명", delta_color="inverse")
    col3.metric("총 구조자", f"{df_raw['Survival'].sum()}명")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("💔 사망자 요약")
        st.dataframe(df_raw.groupby('age_group', observed=False)['Death'].sum(), use_container_width=True)
    with c2:
        st.subheader("✅ 구조자 요약")
        st.dataframe(df_raw.groupby('pclass')['Survival'].sum(), use_container_width=True)

def plot_counts(df_raw, category, target, plot_type, extreme_select):
    if category == 'age':
        plot_data = df_raw.groupby('age_group', observed=False)[target].sum().reset_index()
        x_col = 'age_group'
    else:
        plot_data = df_raw.groupby(category)[target].sum().reset_index()
        x_col = category
        plot_data[x_col] = "C" + plot_data[x_col].astype(str)

    # 그래프 크기 대폭 축소 (5, 3)
    fig, ax = plt.subplots(figsize=(5, 3))
    if plot_type == 'Bar Chart':
        sns.barplot(x=x_col, y=target, data=plot_data, ax=ax, palette='magma')
    else:
        sns.lineplot(x=x_col, y=target, data=plot_data, ax=ax, marker='o')
    
    ax.set_title(f"{target} by {category.capitalize()}", fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=8)
    st.pyplot(fig)

    if extreme_select == '가장 높은 지점':
        top = plot_data.loc[plot_data[target].idxmax()]
        st.success(f"🥇 최고: {top[x_col]} ({top[target]}명)")
    else:
        bottom = plot_data.loc[plot_data[target].idxmin()]
        st.error(f"🥉 최저: {bottom[x_col]} ({bottom[target]}명)")

def plot_correlation(df, corr_plot_type):
    st.subheader(f"🔗 상관관계: {corr_plot_type}")
    fig, ax = plt.subplots(figsize=(4, 3)) # 크기 축소
    if corr_plot_type == 'Heatmap':
        sns.heatmap(df[['survived', 'age', 'fare']].corr(), annot=True, cmap='RdBu', fmt=".2f", ax=ax, annot_kws={"size": 7})
    else:
        sns.scatterplot(data=df, x='age', y='fare', hue='survived', alpha=0.5, ax=ax, s=20)
    ax.tick_params(labelsize=7)
    st.pyplot(fig)

def plot_boxplot(df):
    st.subheader("📊 Age & Fare Boxplot")
    fig, ax = plt.subplots(figsize=(4, 2.5)) # 크기 축소
    sns.boxplot(data=df[['age', 'fare']], ax=ax, palette="vlag")
    ax.tick_params(labelsize=7)
    st.pyplot(fig)

# --- 4. 메인 실행 ---
def main():
    raw_data = load_data(FILE_PATH)
    if raw_data is None: return

    df = handle_data(raw_data)
    df_norm = normalize_data(df)

    menu = st.sidebar.radio("메뉴", ['종합 요약', '분석 그래프', '상관관계', '박스 플롯'])

    if menu == '종합 요약':
        generate_summary_tables(df)
    elif menu == '분석 그래프':
        theme = st.sidebar.selectbox("대상", ['Death', 'Survival'])
        cat = st.sidebar.selectbox("기준", ['age', 'pclass'])
        style = st.sidebar.radio("형태", ['Bar Chart', 'Line Chart'])
        ext = st.sidebar.radio("강조", ['가장 높은 지점', '가장 낮은 지점'])
        plot_counts(df, cat, theme, style, ext)
    elif menu == '상관관계':
        style = st.sidebar.radio("방식", ['Heatmap', 'Scatter Plot'])
        plot_correlation(df_norm, style)
    elif menu == '박스 플롯':
        plot_boxplot(df_norm)
        # 분위수 분석은 텍스트로 간결하게
        st.write("**Stat Summary**")
        st.write(df[['age', 'fare']].describe().loc[['25%', '50%', '75%']])

if __name__ == "__main__":
    main()
