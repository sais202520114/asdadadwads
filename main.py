import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import io

# --- 1. 환경 설정 (차트 내 영어 사용으로 폰트 오류 방지) ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(
    page_title="타이타닉 분석 대시보드",
    layout="wide",
    initial_sidebar_state="expanded"
)

FILE_PATH = "titanic.xls"

# --- 2. 데이터 처리 함수 (원본 기능 100% 복구) ---
@st.cache_data
def load_and_preprocess(file_path):
    try:
        # 엔진 우선순위 설정하여 로드 오류 방지
        df = pd.read_excel(file_path, engine='xlrd')
    except:
        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            st.error(f"❌ File Load Error: {e}")
            return None
    
    # 필요한 컬럼만 추출
    df = df[['pclass', 'survived', 'sex', 'age', 'fare']].copy()
    
    # 결측치 처리 (원본 로직 유지)
    df['pclass'] = df['pclass'].fillna(df['pclass'].mode()[0]).astype(int)
    df['survived'] = df['survived'].fillna(0).astype(int)
    df['age'] = df['age'].fillna(df['age'].median())
    df['fare'] = df['fare'].fillna(df['fare'].median())
    
    # 분석용 파생 컬럼 생성
    df['Death'] = 1 - df['survived']
    df['Survival'] = df['survived']
    
    # 연령대 그룹 생성 (원본 bins 유지)
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 100]
    labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)
    
    return df

# --- 3. 메인 앱 로직 ---
def main():
    df = load_and_preprocess(FILE_PATH)
    if df is None: return

    # 데이터 정규화 처리 (이상치 무시하고 스케일링 적용)
    scaler = MinMaxScaler()
    df_norm = df.copy()
    df_norm[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])

    st.sidebar.title("🔍 분석 메뉴")
    menu = st.sidebar.radio("항목 선택", ['종합 요약', '분석 그래프', '상관관계 및 통계'])

    if menu == '종합 요약':
        st.title("🚢 타이타닉 데이터 종합 요약")
        # 메트릭 표시
        m1, m2, m3 = st.columns(3)
        m1.metric("총 인원", f"{len(df)}명")
        m2.metric("총 사망자", f"{df['Death'].sum()}명", delta_color="inverse")
        m3.metric("총 구조자", f"{df['Survival'].sum()}명")
        
        st.markdown("---")
        # 요약 데이터프레임 출력 (원본 기능)
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("💔 사망자 상세 통계")
            st.write("**연령대별 사망자**")
            st.dataframe(df.groupby('age_group', observed=False)['Death'].sum(), use_container_width=True)
            st.write("**객실 등급별 사망자**")
            st.dataframe(df.groupby('pclass')['Death'].sum(), use_container_width=True)
        with col_right:
            st.subheader("✅ 구조자 상세 통계")
            st.write("**연령대별 구조자**")
            st.dataframe(df.groupby('age_group', observed=False)['Survival'].sum(), use_container_width=True)
            st.write("**객실 등급별 구조자**")
            st.dataframe(df.groupby('pclass')['Survival'].sum(), use_container_width=True)

    elif menu == '분석 그래프':
        st.title("📊 사망/구조자 시각화")
        target_choice = st.sidebar.selectbox("분석 대상", ['사망자 수', '구조자 수'])
        target = 'Death' if target_choice == '사망자 수' else 'Survival'
        category = st.sidebar.selectbox("분류 기준", ['age_group', 'pclass'])
        plot_type = st.sidebar.radio("그래프 형태", ['Bar Chart', 'Line Chart'])
        extreme_select = st.sidebar.radio("강조 지점", ['최고치', '최저치'])
        
        plot_data = df.groupby(category, observed=False)[target].sum().reset_index()
        
        # 그래프 크기 적당히 조절 (화면 60% 사용)
        c_plot, _ = st.columns([1.5, 1])
        with c_plot:
            fig, ax = plt.subplots(figsize=(7, 4))
            if plot_type == 'Bar Chart':
                sns.barplot(data=plot_data, x=category, y=target, ax=ax, palette='viridis')
            else:
                sns.lineplot(data=plot_data, x=category, y=target, ax=ax, marker='o')
            
            # 차트 내부 영어 설정
            ax.set_title(f"{target} Count by {category.capitalize()}", fontsize=12)
            ax.set_xlabel(category.upper())
            ax.set_ylabel("COUNT")
            st.pyplot(fig)

        # 강조 지점 표시 기능 (원본 복구)
        if extreme_select == '최고치':
            top = plot_data.loc[plot_data[target].idxmax()]
            st.success(f"🥇 최고 지점: {top[category]} ({top[target]}명)")
        else:
            low = plot_data.loc[plot_data[target].idxmin()]
            st.error(f"🥉 최저 지점: {low[category]} ({low[target]}명)")

    elif menu == '상관관계 및 통계':
        st.title("📈 상관관계 및 분위수 분석")
        col_1, col_2 = st.columns([1.2, 1])
        
        with col_1:
            st.subheader("상관관계 히트맵 (Heatmap)")
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            sns.heatmap(df[['survived', 'age', 'fare']].corr(), annot=True, cmap='coolwarm', ax=ax1)
            ax1.set_title("Correlation Matrix", fontsize=12)
            st.pyplot(fig1)
            
        with col_2:
            st.subheader("통계 상세 분석")
            # 분위수(Quantile) 분석 기능 (원본 복구)
            for col_name in ['age', 'fare']:
                q1 = df[col_name].quantile(0.25)
                median = df[col_name].median()
                q3 = df[col_name].quantile(0.75)
                st.info(f"**{col_name.upper()}**\n\nQ1: {q1:.1f} | Median: {median:.1f} | Q3: {q3:.1f}")
            
            st.markdown("---")
            st.write("**정규화 데이터 박스 플롯 (Boxplot)**")
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            sns.boxplot(data=df_norm[['age', 'fare']], ax=ax2, palette="Set3")
            ax2.set_title("Normalized Data Distribution", fontsize=10)
            st.pyplot(fig2)

if __name__ == "__main__":
    main()
