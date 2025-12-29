import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 1. 그래프 한글 깨짐 방지 및 스타일 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

# 페이지 설정
st.set_page_config(page_title="Titanic Analysis Full Dashboard", layout="wide")

# 2. 데이터 로드 및 전처리
@st.cache_data
def load_full_data():
    try:
        df = pd.read_excel("titanic.xlsx", engine='openpyxl')
        
        cols = ['pclass', 'survived', 'sex', 'age', 'fare']
        df = df[cols].copy()

        df['pclass'] = df['pclass'].fillna(df['pclass'].mode()[0]).astype(int)
        df['survived'] = df['survived'].fillna(0).astype(int)
        df['age'] = df['age'].fillna(df['age'].median())
        df['fare'] = df['fare'].fillna(df['fare'].median())

        df['Death'] = 1 - df['survived']
        df['Survival'] = df['survived']

        bins = [0, 10, 20, 30, 40, 50, 60, 70, 100]
        labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71+']
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)
        df['age_group'] = pd.Categorical(df['age_group'], categories=labels, ordered=True)

        return df
    except Exception as e:
        st.error(f"데이터 로드 에러: {e}")
        return None

# 3. 메인 대시보드 실행
def main():
    df = load_full_data()
    if df is None:
        return

    scaler = MinMaxScaler()
    df_norm = df.copy()
    df_norm[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])

    st.sidebar.title("🚢 타이타닉 분석")
    menu = st.sidebar.radio("메뉴 선택", ['종합 대시보드', '사망/구조 분석 시각화', '심화 통계 분석'])

    if menu == '종합 대시보드':
        st.title("📊 타이타닉 데이터 종합 현황")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("전체 승객", f"{len(df)}명")
        m2.metric("총 사망자", f"{df['Death'].sum()}명")
        m3.metric("총 구조자", f"{df['Survival'].sum()}명")
        survival_rate = (df['Survival'].sum() / len(df)) * 100
        m4.metric("평균 생존율", f"{survival_rate:.1f}%")

        st.divider()
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("💀 사망자 상세 통계")
            death_age = df.groupby('age_group')['Death'].sum()
            death_pclass = df.groupby('pclass')['Death'].sum()
            st.dataframe(death_age, use_container_width=True)
            st.dataframe(death_pclass, use_container_width=True)
        with col_right:
            st.subheader("✅ 구조자 상세 통계")
            surv_age = df.groupby('age_group')['Survival'].sum()
            surv_pclass = df.groupby('pclass')['Survival'].sum()
            st.dataframe(surv_age, use_container_width=True)
            st.dataframe(surv_pclass, use_container_width=True)

    elif menu == '사망/구조 분석 시각화':
        st.title("📈 시각화 차트 분석")

        target_label = st.sidebar.radio("데이터 종류", ['사망자 수', '구조자 수'])
        target_col = 'Death' if target_label == '사망자 수' else 'Survival'
        category = st.sidebar.selectbox("분류 기준 (X축)", ['age_group', 'pclass', 'sex'])
        chart_type = st.sidebar.radio("차트 형태", ['Bar', 'Line', 'Histogram'])

        fig, ax = plt.subplots(figsize=(10, 5))

        plot_data = df.groupby(category)[target_col].sum().reset_index()

        if category == 'age_group':
            labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71+']
            plot_data[category] = pd.Categorical(plot_data[category], categories=labels, ordered=True)
            plot_data = plot_data.sort_values(category)

        if chart_type == 'Bar':
            sns.barplot(data=plot_data, x=category, y=target_col, ax=ax, palette='viridis')
            ax.set_title(f"{category}별 {target_label}", fontsize=15)

        elif chart_type == 'Line':
            if category == 'age_group':
                x_vals = range(len(plot_data))
                y_vals = plot_data[target_col].values
                ax.plot(x_vals, y_vals, marker='o', color='teal')
                ax.set_xticks(x_vals)
                ax.set_xticklabels(plot_data[category].astype(str))
                ax.set_title(f"{category}에 따른 {target_label} 변화", fontsize=15)
            else:
                sns.lineplot(data=plot_data, x=category, y=target_col, ax=ax, marker='o', color='teal')
                ax.set_title(f"{category}에 따른 {target_label} 변화", fontsize=15)

        elif chart_type == 'Histogram':
            if category in ['age_group', 'sex']:
                sns.countplot(data=df, x=category, hue='survived', palette='coolwarm', ax=ax)
            else:
                sns.histplot(data=df, x=category, hue='survived', multiple="stack", kde=True, palette='coolwarm', ax=ax)
            ax.set_title(f"생존 여부에 따른 {category} 분포", fontsize=15)

        plt.tight_layout()
        st.pyplot(fig)

    elif menu == '심화 통계 분석':
        st.title("🔍 수치 데이터 심화 분석")

        st.subheader("1. 변수 간 상관관계 (Heatmap)")
        corr_data = df[['survived', 'age', 'fare', 'pclass']].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_data, annot=True, cmap='RdBu', fmt=".2f", ax=ax_corr, center=0)
        st.pyplot(fig_corr)

        st.divider()
        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.subheader("2. 정규화 데이터 변동성 (Boxplot)")
            fig_box, ax_box = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=df_norm[['age', 'fare']], ax=ax_box, palette='Pastel1')
            ax_box.set_title("Normalized Age & Fare Distribution")
            st.pyplot(fig_box)
        with c2:
            st.subheader("3. 주요 수치 분위수")
            for col in ['age', 'fare']:
                q = df[col].quantile([0.25, 0.5, 0.75])
                with st.expander(f"📌 {col.upper()} 통계 보기"):
                    st.write(f"**1사분위 (Q1):** {q[0.25]:.2f}")
                    st.write(f"**중앙값 (Median):** {q[0.5]:.2f}")
                    st.write(f"**3사분위 (Q3):** {q[0.75]:.2f}")
                    st.write(f"**IQR:** {q[0.75]-q[0.25]:.2f}")

        st.divider()
        st.subheader("4. 나이와 요금의 상관관계 (Scatter Plot)")
        fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
        sns.scatterplot(df, x='age', y='fare', hue='survived', style='survived', alpha=0.6, palette='seismic', ax=ax_scatter)
        ax_scatter.set_title("Age vs Fare (Colored by Survival)", fontsize=15)
        st.pyplot(fig_scatter)

if __name__ == "__main__":
    main()
