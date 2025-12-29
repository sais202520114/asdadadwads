import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# --- 1. 기본 환경 설정 ---
# 차트 내 한글 깨짐 방지를 위해 차트 텍스트는 영문으로 작성
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(
    page_title="타이타닉 분석 대시보드",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 데이터 파일 경로 (파일이 같은 폴더에 있어야 합니다)
FILE_PATH = "titanic.xls"

# --- 2. 데이터 로드 및 전처리 함수 ---
@st.cache_data
def load_and_preprocess(file_path):
    try:
        # xlrd 엔진을 사용하여 xls 파일 로드
        df = pd.read_excel(file_path, engine='xlrd')
    except Exception:
        try:
            # 엔진 없이 재시도
            df = pd.read_excel(file_path)
        except Exception as e:
            st.error(f"데이터 파일을 불러오지 못했습니다: {e}")
            return None
    
    # 분석에 필요한 핵심 컬럼 선택
    cols = ['pclass', 'survived', 'sex', 'age', 'fare']
    df = df[cols].copy()
    
    # 결측치 처리 (최빈값/중앙값 활용)
    df['pclass'] = df['pclass'].fillna(df['pclass'].mode()[0]).astype(int)
    df['survived'] = df['survived'].fillna(0).astype(int)
    df['age'] = df['age'].fillna(df['age'].median())
    df['fare'] = df['fare'].fillna(df['fare'].median())
    
    # 분석용 파생 변수 생성 (사망/구조 여부)
    df['Death'] = 1 - df['survived']
    df['Survival'] = df['survived']
    
    # 연령대 그룹화 (0세부터 70세 이상까지)
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 100]
    labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)
    
    return df

# --- 3. 메인 애플리케이션 ---
def main():
    df = load_and_preprocess(FILE_PATH)
    
    if df is not None:
        # 데이터 정규화 (Min-Max Scaling) - 상관관계 분석용
        scaler = MinMaxScaler()
        df_norm = df.copy()
        df_norm[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])

        # 사이드바 메뉴
        st.sidebar.title("🚢 Titanic Dashboard")
        menu = st.sidebar.selectbox("메뉴 선택", ['데이터 요약', '시각화 분석', '심화 통계'])

        # --- 메뉴 1: 데이터 요약 ---
        if menu == '데이터 요약':
            st.title("📊 타이타닉 데이터 종합 요약")
            
            # 상단 지표
            m1, m2, m3 = st.columns(3)
            m1.metric("총 승객 수", f"{len(df)} 명")
            m2.metric("총 사망자 수", f"{df['Death'].sum()} 명")
            m3.metric("총 구조자 수", f"{df['Survival'].sum()} 명")
            
            st.markdown("---")
            
            # 상세 데이터 테이블
            col_left, col_right = st.columns(2)
            with col_left:
                st.subheader("💔 사망자 상세 통계")
                st.write("**연령대별 사망자**")
                st.table(df.groupby('age_group', observed=False)['Death'].sum())
                st.write("**객실 등급별 사망자**")
                st.table(df.groupby('pclass')['Death'].sum())
                
            with col_right:
                st.subheader("✅ 구조자 상세 통계")
                st.write("**연령대별 구조자**")
                st.table(df.groupby('age_group', observed=False)['Survival'].sum())
                st.write("**객실 등급별 구조자**")
                st.table(df.groupby('pclass')['Survival'].sum())

        # --- 메뉴 2: 시각화 분석 ---
        elif menu == '시각화 분석':
            st.title("📈 시각화 차트")
            
            # 사용자 선택 옵션
            target_label = st.sidebar.radio("데이터 선택", ['사망자 수', '구조자 수'])
            target_col = 'Death' if target_label == '사망자 수' else 'Survival'
            
            category = st.sidebar.selectbox("분류 기준", ['age_group', 'pclass'])
            chart_type = st.sidebar.radio("그래프 형태", ['Bar Chart', 'Line Chart'])
            
            # 데이터 그룹화
            plot_data = df.groupby(category, observed=False)[target_col].sum().reset_index()
            
            # 메인 그래프 출력 영역
            col_chart, col_empty = st.columns([2, 1])
            with col_chart:
                fig, ax = plt.subplots(figsize=(8, 5))
                if chart_type == 'Bar Chart':
                    sns.barplot(data=plot_data, x=category, y=target_col, ax=ax, palette='magma')
                else:
                    sns.lineplot(data=plot_data, x=category, y=target_col, ax=ax, marker='o', color='red')
                
                # 차트 내부는 영어로 설정 (한글 깨짐 방지)
                ax.set_title(f"{target_col} Counts by {category.upper()}", fontsize=14)
                ax.set_xlabel(category.upper())
                ax.set_ylabel("Count")
                st.pyplot(fig)

            # 분석 결과 텍스트 강조
            st.markdown("---")
            extreme = st.radio("특이 지점 확인", ['최고치 데이터', '최저치 데이터'])
            if extreme == '최고치 데이터':
                top_val = plot_data.loc[plot_data[target_col].idxmax()]
                st.success(f"💡 분석 결과: **{top_val[category]}** 그룹에서 {target_label}가 **{top_val[target_col]}명**으로 가장 많습니다.")
            else:
                low_val = plot_data.loc[plot_data[target_col].idxmin()]
                st.warning(f"💡 분석 결과: **{low_val[category]}** 그룹에서 {target_label}가 **{low_val[target_col]}명**으로 가장 적습니다.")

        # --- 메뉴 3: 심화 통계 ---
        elif menu == '심화 통계':
            st.title("🔍 상관관계 및 수치 분석")
            
            c1, c2 = st.columns([1.2, 1])
            with c1:
                st.subheader("1. 변수 간 상관관계 (Heatmap)")
                fig_corr, ax_corr = plt.subplots()
                sns.heatmap(df[['survived', 'age', 'fare']].corr(), annot=True, cmap='RdBu', ax=ax_corr)
                ax_corr.set_title("Correlation Heatmap")
                st.pyplot(fig_corr)
                
            with c2:
                st.subheader("2. 주요 변수 분위수 분석")
                for item in ['age', 'fare']:
                    q1 = df[item].quantile(0.25)
                    q2 = df[item].median()
                    q3 = df[item].quantile(0.75)
                    st.info(f"📍 **{item.upper()}** 통계\n- 25%(Q1): {q1:.2f}\n- 50%(중앙값): {q2:.2f}\n- 75%(Q3): {q3:.1f}")
            
            st.markdown("---")
            st.subheader("3. 정규화 데이터 분포 (Boxplot)")
            fig_box, ax_box = plt.subplots(figsize=(10, 4))
            sns.boxplot(data=df_norm[['age', 'fare']], ax=ax_box, orient='h', palette='Pastel1')
            ax_box.set_title("Normalized Distribution (Age & Fare)")
            st.pyplot(fig_box)

if __name__ == "__main__":
    main()
