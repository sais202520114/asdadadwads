import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# --- 1. 설정 및 데이터 로드 ---
FILE_PATH = "titanic.xls"

# Matplotlib 한글 폰트 설정 (OS별 호환성 고려)
plt.rcParams['font.family'] = 'Malgun Gothic' if 'Windows' in st.runtime.exists() else 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="타이타닉 데이터 분석 대시보드", layout="wide")

@st.cache_data
def load_and_preprocess(file_path):
    try:
        # 엔진을 'xlrd'로 명시하거나 openpyxl을 사용 (xls 확장자 대응)
        df = pd.read_excel(file_path)
    except Exception as e:
        st.error(f"파일 로드 실패: {e}")
        return None

    # 결측치 처리 및 기본 전처리
    df['pclass'] = df['pclass'].fillna(df['pclass'].mode()[0]).astype(int)
    df['survived'] = df['survived'].fillna(0).astype(int)
    df['age'] = df['age'].fillna(df['age'].median())
    df['fare'] = df['fare'].fillna(df['fare'].median())
    
    # 분석용 열 추가
    df['Death'] = 1 - df['survived']
    df['Survival'] = df['survived']
    
    # 연령대 그룹화
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 100]
    labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=True, include_lowest=True)
    
    return df

# --- 2. 시각화 함수 ---
def plot_counts(df, category, target, target_name, plot_type, extreme_select):
    # groupby 시 observed=False를 추가하여 카테고리 에러 방지
    if category == 'age':
        plot_data = df.groupby('age_group', observed=False)[target].sum().reset_index()
        x_col = 'age_group'
    else:
        plot_data = df.groupby(category, observed=False)[target].sum().reset_index()
        x_col = category
        plot_data[x_col] = plot_data[x_col].astype(str) + "등석"

    fig, ax = plt.subplots(figsize=(8, 5))
    if plot_type == 'Bar Chart':
        sns.barplot(x=x_col, y=target, data=plot_data, ax=ax, palette='muted')
    else:
        sns.lineplot(x=x_col, y=target, data=plot_data, ax=ax, marker='o', linewidth=2)
    
    ax.set_title(f"{category}별 {target_name} 분포", fontsize=15)
    st.pyplot(fig)

    # 강조 지점 표시
    if extreme_select == '가장 높은 지점':
        top = plot_data.loc[plot_data[target].idxmax()]
        st.success(f"🥇 최고치: {top[x_col]} ({top[target]}명)")
    else:
        bottom = plot_data.loc[plot_data[target].idxmin()]
        st.error(f"🥉 최저치: {bottom[x_col]} ({bottom[target]}명)")

# --- 3. 메인 앱 실행 ---
def main():
    df_raw = load_and_preprocess(FILE_PATH)
    if df_raw is None: return

    # 사이드바
    st.sidebar.title("🚢 타이타닉 메뉴")
    menu = st.sidebar.radio("메뉴 선택", ['종합 요약', '사망/구조자 분석', '상관관계 & 박스플롯'])

    if menu == '종합 요약':
        st.title("🚢 타이타닉 데이터 종합 요약")
        m1, m2, m3 = st.columns(3)
        m1.metric("총 승객", f"{len(df_raw)}명")
        m2.metric("총 사망자", f"{df_raw['Death'].sum()}명", delta_color="inverse")
        m3.metric("총 생존자", f"{df_raw['Survival'].sum()}명")
        
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("💀 연령대별 사망 요약")
            st.table(df_raw.groupby('age_group', observed=False)['Death'].sum())
        with c2:
            st.subheader("🏥 선실별 생존 요약")
            st.table(df_raw.groupby('pclass', observed=False)['Survival'].sum())

    elif menu == '사망/구조자 분석':
        theme = st.sidebar.selectbox("분석 대상", ['사망자 수', '구조자 수'])
        target = 'Death' if theme == '사망자 수' else 'Survival'
        cat = st.sidebar.selectbox("분류 기준", ['age', 'pclass'])
        style = st.sidebar.radio("그래프 형태", ['Bar Chart', 'Line Chart'])
        extreme = st.sidebar.radio("강조 지점", ['가장 높은 지점', '가장 낮은 지점'])
        
        plot_counts(df_raw, cat, target, theme, style, extreme)

    elif menu == '상관관계 & 박스플롯':
        st.header("📈 통계 분석")
        
        # 정규화 및 이상치 처리 (시각화용)
        df_norm = df_raw.copy()
        scaler = MinMaxScaler()
        df_norm[['age', 'fare']] = scaler.fit_transform(df_norm[['age', 'fare']])
        
        tab1, tab2 = st.tabs(["상관계수 히트맵", "변수별 박스플롯"])
        
        with tab1:
            fig, ax = plt.subplots()
            sns.heatmap(df_raw[['survived', 'age', 'fare']].corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
            
        with tab2:
            fig, ax = plt.subplots()
            sns.boxplot(data=df_norm[['age', 'fare']], ax=ax)
            st.pyplot(fig)
            st.write("※ 데이터는 0~1 사이로 정규화되었습니다.")

if __name__ == "__main__":
    main()
