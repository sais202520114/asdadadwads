import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 1. 그래프 한글 깨짐 방지 및 스타일 설정
# 환경에 따라 나눔고딕이나 맑은 고딕 등 설치된 폰트를 우선 적용하도록 설정합니다.
plt.rcParams['font.family'] = 'Malgun Gothic' # Windows용
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

# 페이지 설정
st.set_page_config(page_title="Titanic Analysis Full Dashboard", layout="wide")

# 2. 데이터 로드 및 전처리
@st.cache_data
def load_full_data():
    try:
        # 타이타닉 데이터 로드 (xlrd 엔진 사용)
        df = pd.read_excel("titanic.xls", engine='xlrd')
        
        # 분석 핵심 컬럼 추출
        cols = ['pclass', 'survived', 'sex', 'age', 'fare']
        df = df[cols].copy()

        # 결측치 처리
        df['pclass'] = df['pclass'].fillna(df['pclass'].mode()[0]).astype(int)
        df['survived'] = df['survived'].fillna(0).astype(int)
        df['age'] = df['age'].fillna(df['age'].median())
        df['fare'] = df['fare'].fillna(df['fare'].median())

        # 파생 변수 생성
        df['Death'] = 1 - df['survived']
        df['Survival'] = df['survived']

        # 연령대 그룹화
        bins = [0, 10, 20, 30, 40, 50, 60, 70, 100]
        labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71+']
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)
        
        return df
    except Exception as e:
        st.error(f"데이터를 로드하는 중 에러가 발생했습니다: {e}")
        return None

# 3. 메인 대시보드 실행
def main():
    df = load_full_data()
    
    if df is not None:
        # 데이터 정규화 (Min-Max Scaling)
        scaler = MinMaxScaler()
        df_norm = df.copy()
        df_norm[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])

        # 사이드바 메뉴
        st.sidebar.title("🚢 타이타닉 분석")
        menu = st.sidebar.radio("원하는 분석을 선택하세요", 
                                ['종합 대시보드', '사망/구조 분석 시각화', '심화 통계 분석'])

        # --- [메뉴 1: 종합 대시보드] ---
        if menu == '종합 대시보드':
            st.title("📊 타이타닉 데이터 종합 현황")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("전체 승객 수", f"{len(df)} 명")
            m2.metric("총 사망자", f"{df['Death'].sum()} 명", delta_color="inverse")
            m3.metric("총 구조자", f"{df['Survival'].sum()} 명")
            
            st.divider()
            
            col_left, col_right = st.columns(2)
            with col_left:
                st.subheader("💀 사망 통계 (연령대/등급)")
                st.write("**연령대별 사망자**")
                st.table(df.groupby('age_group', observed=False)['Death'].sum())
                st.write("**객실 등급별 사망자**")
                st.table(df.groupby('pclass')['Death'].sum())
            with col_right:
                st.subheader("✅ 구조 통계 (연령대/등급)")
                st.write("**연령대별 구조자**")
                st.table(df.groupby('age_group', observed=False)['Survival'].sum())
                st.write("**객실 등급별 구조자**")
                st.table(df.groupby('pclass')['Survival'].sum())

        # --- [메뉴 2: 사망/구조 분석 시각화] ---
        elif menu == '사망/구조 분석 시각화':
            st.title("📈 시각화 차트 분석")
            
            target_label = st.sidebar.radio("데이터 종류", ['사망자 수', '구조자 수'])
            target_col = 'Death' if target_label == '사망자 수' else 'Survival'
            category = st.sidebar.selectbox("분류 기준 (X축)", ['age_group', 'pclass', 'sex'])
            
            # 사용자 요구사항 반영: Pie 제거, Histogram 추가
            chart_type = st.sidebar.radio("차트 형태", ['Bar', 'Line', 'Histogram'])
            
            fig, ax = plt.subplots(figsize=(10, 5))
            
            if chart_type == 'Bar':
                plot_data = df.groupby(category, observed=False)[target_col].sum().reset_index()
                sns.barplot(data=plot_data, x=category, y=target_col, ax=ax, palette='magma', hue=category, legend=False)
                ax.set_title(f"{target_label} Distribution by {category.upper()}", fontsize=14)
            
            elif chart_type == 'Line':
                plot_data = df.groupby(category, observed=False)[target_col].sum().reset_index()
                sns.lineplot(data=plot_data, x=category, y=target_col, ax=ax, marker='o', color='teal', group=1)
                ax.set_title(f"{target_label} Trend by {category.upper()}", fontsize=14)
            
            elif chart_type == 'Histogram':
                # 히스토그램은 전체 연령 분포에서 생존 여부를 확인
                sns.histplot(data=df, x='age', hue='survived', multiple="stack", kde=True, ax=ax, palette='viridis')
                ax.set_title("Age Distribution by Survival Status", fontsize=14)
                ax.set_xlabel("Age")
                ax.set_ylabel("Count")

            st.pyplot(fig)

        # --- [메뉴 3: 심화 통계 분석] ---
        elif menu == '심화 통계 분석':
            st.title("🔍 수치 데이터 심화 분석")
            
            # 1. 히트맵 (Heatmap)
            st.subheader("1. 변수 간 상관관계 (Heatmap)")
            fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
            # 수치형 데이터만 선택하여 상관계수 산출
            numeric_df = df[['survived', 'age', 'fare', 'pclass']]
            corr = numeric_df.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
            st.pyplot(fig_corr)
            
            st.divider()

            # 2. 박스플롯 & 분위수
            c1, c2 = st.columns([1.5, 1])
            with c1:
                st.subheader("2. 정규화 데이터 분포 (Boxplot)")
                fig_box, ax_box = plt.subplots(figsize=(8, 5))
                sns.boxplot(data=df_norm[['age', 'fare']], ax=ax_box, orient='h', palette='Set2')
                st.pyplot(fig_box)
            with c2:
                st.subheader("3. 분위수 통계")
                for item in ['age', 'fare']:
                    q = df[item].quantile([0.25, 0.5, 0.75])
                    st.write(f"📍 **{item.upper()}**")
                    st.write(f"Q1 (25%): {q[0.25]:.2f}")
                    st.write(f"Med (50%): {q[0.5]:.2f}")
                    st.write(f"Q3 (75%): {q[0.75]:.2f}")
                    st.write("---")

            st.divider()

            # 4. 산점도 (Scatter Plot)
            st.subheader("4. 나이와 요금의 상관관계 (Scatter Plot)")
            fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df, x='age', y='fare', hue='survived', alpha=0.7, ax=ax_scatter, palette='coolwarm')
            ax_scatter.set_title("Age vs Fare Correlation", fontsize=15)
            st.pyplot(fig_scatter)

if __name__ == "__main__":
    main()
