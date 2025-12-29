import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 1. 그래프 한글 깨짐 방지 및 스타일 설정
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

# 페이지 설정
st.set_page_config(page_title="Titanic Analysis Full Dashboard", layout="wide")

# 2. 데이터 로드 및 전처리 (캐싱 적용)
@st.cache_data
def load_full_data():
    try:
        # 타이타닉 데이터 로드 (xls 파일)
        df = pd.read_excel("titanic.xls", engine='xlrd')
    except Exception as e:
        # 파일이 없거나 엔진 문제가 있을 경우 에러 메시지
        st.error(f"파일을 읽을 수 없습니다. 'titanic.xls' 파일이 같은 폴더에 있는지 확인하세요. 에러: {e}")
        return None

    # 필요한 컬럼만 추출
    cols = ['pclass', 'survived', 'sex', 'age', 'fare']
    df = df[cols].copy()

    # 결측치 처리 (최빈값 및 중앙값 사용)
    df['pclass'] = df['pclass'].fillna(df['pclass'].mode()[0]).astype(int)
    df['survived'] = df['survived'].fillna(0).astype(int)
    df['age'] = df['age'].fillna(df['age'].median())
    df['fare'] = df['fare'].fillna(df['fare'].median())

    # 분석용 파생 변수 (사망/구조 여부 명시)
    df['Death'] = 1 - df['survived']
    df['Survival'] = df['survived']

    # 연령대 그룹화 (0세부터 70세 이상까지)
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 100]
    labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)
    
    return df

# 3. 메인 대시보드 실행
def main():
    df = load_full_data()
    
    if df is not None:
        # 데이터 정규화 (Min-Max Scaling) - 분포 비교용
        scaler = MinMaxScaler()
        df_norm = df.copy()
        df_norm[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])

        # 사이드바 메뉴 구성
        st.sidebar.title("🚢 타이타닉 분석")
        menu = st.sidebar.radio("원하는 분석을 선택하세요", ['종합 대시보드', '상세 그래프', '심화 통계 분석'])

        # --- [메뉴 1: 종합 대시보드] ---
        if menu == '종합 대시보드':
            st.title("📊 타이타닉 데이터 종합 현황")
            
            # 상단 핵심 지표(Metrics)
            m1, m2, m3 = st.columns(3)
            m1.metric("전체 승객 수", f"{len(df)} 명")
            m2.metric("총 사망자", f"{df['Death'].sum()} 명", delta_color="inverse")
            m3.metric("총 구조자", f"{df['Survival'].sum()} 명")
            
            st.divider()
            
            # 테이블 요약
            col_left, col_right = st.columns(2)
            with col_left:
                st.subheader("💀 연령대별 사망 통계")
                st.table(df.groupby('age_group', observed=False)['Death'].sum())
                st.subheader("🏢 객실 등급별 사망 통계")
                st.table(df.groupby('pclass')['Death'].sum())
                
            with col_right:
                st.subheader("✅ 연령대별 구조 통계")
                st.table(df.groupby('age_group', observed=False)['Survival'].sum())
                st.subheader("🏢 객실 등급별 구조 통계")
                st.table(df.groupby('pclass')['Survival'].sum())

        # --- [메뉴 2: 상세 그래프] ---
        elif menu == '상세 그래프':
            st.title("📈 시각화 차트 분석")
            
            # 사용자 선택 인터페이스
            target_label = st.sidebar.radio("데이터 종류", ['사망자 수', '구조자 수'])
            target_col = 'Death' if target_label == '사망자 수' else 'Survival'
            
            category = st.sidebar.selectbox("분류 기준 (X축)", ['age_group', 'pclass', 'sex'])
            chart_type = st.sidebar.radio("차트 형태", ['Bar', 'Line', 'Pie'])
            
            # 그래프용 데이터 가공
            plot_data = df.groupby(category, observed=False)[target_col].sum().reset_index()
            
            # 메인 차트 출력
            fig, ax = plt.subplots(figsize=(10, 5))
            if chart_type == 'Bar':
                sns.barplot(data=plot_data, x=category, y=target_col, ax=ax, palette='magma')
            elif chart_type == 'Line':
                sns.lineplot(data=plot_data, x=category, y=target_col, ax=ax, marker='o', color='teal')
            elif chart_type == 'Pie':
                ax.pie(plot_data[target_col], labels=plot_data[category], autopct='%1.1f%%', startangle=90)
            
            ax.set_title(f"{target_col} Distribution by {category.upper()}", fontsize=14)
            st.pyplot(fig)
            
            # 분석 텍스트 요약
            st.divider()
            extreme = st.radio("특이 지점 확인", ['가장 높은 그룹', '가장 낮은 그룹'])
            if extreme == '가장 높은 그룹':
                top = plot_data.loc[plot_data[target_col].idxmax()]
                st.success(f"💡 분석 결과: **{top[category]}** 그룹에서 {target_label}가 **{top[target_col]}명**으로 가장 많습니다.")
            else:
                low = plot_data.loc[plot_data[target_col].idxmin()]
                st.warning(f"💡 분석 결과: **{low[category]}** 그룹에서 {target_label}가 **{low[target_col]}명**으로 가장 적습니다.")

        # --- [메뉴 3: 심화 통계 분석] ---
        elif menu == '심화 통계 분석':
            st.title("🔍 수치 데이터 심화 분석")
            
            c1, c2 = st.columns([1.5, 1])
            with c1:
                st.subheader("1. 변수 간 상관관계 (Heatmap)")
                fig_corr, ax_corr = plt.subplots()
                # 수치형 변수간 상관계수 계산
                corr = df[['survived', 'age', 'fare', 'pclass']].corr()
                sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
                st.pyplot(fig_corr)
                
            with c2:
                st.subheader("2. 주요 변수 분위수(Quantile)")
                for item in ['age', 'fare']:
                    q1 = df[item].quantile(0.25)
                    med = df[item].median()
                    q3 = df[item].quantile(0.75)
                    st.info(f"📍 **{item.upper()}** 통계\n- Q1 (25%): {q1:.2f}\n- 중앙값: {med:.2f}\n- Q3 (75%): {q3:.2f}")
            
            st.divider()
            st.subheader("3. 정규화 데이터 분포 비교 (Boxplot)")
            st.write("나이(Age)와 요금(Fare)의 분포를 동일한 스케일(0~1)로 비교합니다.")
            fig_box, ax_box = plt.subplots(figsize=(12, 4))
            sns.boxplot(data=df_norm[['age', 'fare']], ax=ax_box, orient='h', palette='Set2')
            st.pyplot(fig_box)

if __name__ == "__main__":
    main()
