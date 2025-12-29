import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# --- 1. 환경 설정 ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="Titanic Analysis", layout="wide")

FILE_PATH = "titanic.xls"

# --- 2. 데이터 전처리 ---
@st.cache_data
def load_and_preprocess(file_path):
    try:
        df = pd.read_excel(file_path, engine='xlrd')
    except:
        df = pd.read_excel(file_path)
    
    df = df[['pclass', 'survived', 'sex', 'age', 'fare']].copy()
    df['pclass'] = df['pclass'].fillna(df['pclass'].mode()[0]).astype(int)
    df['survived'] = df['survived'].fillna(0).astype(int)
    df['age'] = df['age'].fillna(df['age'].median())
    df['fare'] = df['fare'].fillna(df['fare'].median())
    df['Death'] = 1 - df['survived']
    df['Survival'] = df['survived']
    
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 100]
    labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)
    return df

# --- 3. 메인 로직 ---
def main():
    df = load_and_preprocess(FILE_PATH)
    if df is None: return

    scaler = MinMaxScaler()
    df_norm = df.copy()
    df_norm[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])

    st.sidebar.title("🔍 대시보드 메뉴")
    menu = st.sidebar.radio("항목 선택", ['종합 요약', '분석 그래프', '상관관계/박스플롯'])

    if menu == '종합 요약':
        st.title("🚢 타이타닉 종합 데이터 요약")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Passengers", f"{len(df)}명")
        c2.metric("Deaths", f"{df['Death'].sum()}명")
        c3.metric("Survivors", f"{df['Survival'].sum()}명")
        
        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("💔 사망자 상세 통계")
            st.dataframe(df.groupby('age_group', observed=False)['Death'].sum(), use_container_width=True)
        with col_b:
            st.subheader("✅ 구조자 상세 통계")
            st.dataframe(df.groupby('pclass')['Survival'].sum(), use_container_width=True)

    elif menu == '분석 그래프':
        target_choice = st.sidebar.selectbox("분석 대상", ['사망자 수', '구조자 수'])
        # 내부 로직용 영어 변수 할당
        target = 'Death' if target_choice == '사망자 수' else 'Survival'
        cat = st.sidebar.selectbox("분류 기준", ['age_group', 'pclass'])
        
        plot_data = df.groupby(cat, observed=False)[target].sum().reset_index()
        
        # 그래프 적당한 크기 조절
        col_plot, _ = st.columns([1.5, 1])
        with col_plot:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(data=plot_data, x=cat, y=target, ax=ax, palette='viridis')
            
            # 🔥 핵심: 제목에서 한글 변수를 제거하고 영문으로만 표기
            ax.set_title(f"Passenger Count by {cat.replace('_', ' ').capitalize()}", fontsize=12)
            ax.set_xlabel(cat.upper())
            ax.set_ylabel("COUNT")
            st.pyplot(fig)

        # 강조 지점 (UI는 한글 유지)
        ext = st.sidebar.radio("강조 지점", ['최고치', '최저치'])
        if ext == '최고치':
            top = plot_data.loc[plot_data[target].idxmax()]
            st.success(f"🥇 최고 지점: {top[cat]} ({top[target]}명)")
        else:
            low = plot_data.loc[plot_data[target].idxmin()]
            st.error(f"🥉 최저 지점: {low[cat]} ({low[target]}명)")

    elif menu == '상관관계/박스플롯':
        st.subheader("📊 데이터 상관관계 및 분포 분석")
        c1, c2 = st.columns([1.2, 1])
        with c1:
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            sns.heatmap(df[['survived', 'age', 'fare']].corr(), annot=True, cmap='coolwarm', ax=ax1)
            ax1.set_title("Heatmap of Variables (Eng)")
            st.pyplot(fig1)
        with c2:
            st.write("**통계 분석 (Quantiles)**")
            for col_name in ['age', 'fare']:
                q1, med, q3 = df[col_name].quantile(0.25), df[col_name].median(), df[col_name].quantile(0.75)
                st.info(f"**{col_name.upper()}**\n\nQ1: {q1:.1f} | Med: {med:.1f} | Q3: {q3:.1f}")
            
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            sns.boxplot(data=df_norm[['age', 'fare']], ax=ax2)
            ax2.set_title("Boxplot (Normalized Data)")
            st.pyplot(fig2)

if __name__ == "__main__":
    main()
