import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit 페이지 설정
st.set_page_config(
    page_title="타이타닉 데이터 상관관계 분석",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 데이터 로드 및 전처리 함수 ---
@st.cache_data
def load_data(file_path):
    """CSV 파일을 로드하고 필요한 전처리를 수행합니다."""
    # 데이터 로드 (업로드된 파일명을 가정)
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"파일을 찾을 수 없습니다: {file_path}")
        return None
    except Exception as e:
        st.error(f"데이터 로딩 중 오류 발생: {e}")
        return None

    # 분석에 필요한 열 선택 및 수치형 변수만 남기기
    # 'pclass', 'survived', 'age', 'sibsp', 'parch', 'fare', 'body' 등 수치형 변수 선택
    # 'pclass', 'survived'는 범주형이지만 분석을 위해 수치형으로 유지
    numeric_df = df.select_dtypes(include=['number']).copy()
    
    # 일부 데이터프레임에는 'body' 열이 없을 수 있으므로, 있으면 사용하고 없으면 무시합니다.
    # 또한, 분석의 명확성을 위해 고유값(ID/티켓번호 등) 성격이 강한 열은 제외합니다.
    if 'ticket' in numeric_df.columns:
         numeric_df = numeric_df.drop('ticket', axis=1, errors='ignore')
    if 'boat' in numeric_df.columns:
         numeric_df = numeric_df.drop('boat', axis=1, errors='ignore')

    # 상관관계 계산을 위해 결측값을 0으로 임시 대체 (분석 목적에 따라 다른 대체 방법 고려 가능)
    numeric_df = numeric_df.fillna(numeric_df.mean())
    
    return numeric_df

# --- 상관관계 분석 함수 ---
def calculate_correlation(df):
    """데이터프레임의 상관관계를 계산하고 결과를 반환합니다."""
    corr_matrix = df.corr()
    
    # 자기 자신과의 상관관계(1)를 제외
    np.fill_diagonal(corr_matrix.values, float('nan'))
    
    # 시리즈 형태로 변환
    corr_unstacked = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
    
    # 가장 높은 양의 상관관계 (1에 가까운 값)
    max_corr = corr_unstacked.dropna().head(1)
    
    # 가장 높은 음의 상관관계 (-1에 가까운 값)
    min_corr = corr_unstacked.dropna().tail(1)
    
    return corr_matrix, max_corr, min_corr

# --- 메인 앱 로직 ---
def main():
    st.title("🚢 타이타닉호 속성 간 상관관계 분석기")
    st.markdown("""
        이 앱은 업로드된 타이타닉 데이터(`titanic.xls - titanic3.csv`)를 사용하여 
        수치형 속성 간의 **상관관계**를 분석하고 시각화합니다.
        가장 강한 양의 상관관계와 음의 상관관계를 가진 속성 쌍을 확인해보세요.
    """)

    # 파일 경로 (업로드된 파일명을 사용)
    file_path = "titanic.xls - titanic3.csv"
    
    data = load_data(file_path)

    if data is not None:
        st.header("🔢 수치형 변수 간 상관관계 히트맵")
        
        corr_matrix, max_corr, min_corr = calculate_correlation(data)

        # 히트맵 시각화
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap='coolwarm', 
            cbar=True,
            linewidths=0.5,
            linecolor='black',
            ax=ax
        )
        ax.set_title("타이타닉호 속성 간 상관관계")
        st.pyplot(fig)
        #  # 상관관계 히트맵 이미지 태그

        st.markdown("---")
        st.header("🔎 주요 상관관계 분석 결과")

        # 두 개의 컬럼을 사용하여 버튼과 결과를 나란히 배치
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📈 가장 강한 양의 상관관계 보기"):
                if not max_corr.empty:
                    st.success("✨ **가장 강한 양의 상관관계**")
                    pair = max_corr.index[0]
                    value = max_corr.values[0]
                    st.metric(
                        label=f"변수 쌍: **{pair[0]}**와 **{pair[1]}**", 
                        value=f"{value:.4f}",
                        delta="양의 상관관계"
                    )
                    st.info(f"👉 **{pair[0]}**의 값이 증가하면 **{pair[1]}**의 값도 증가하는 경향이 가장 강합니다.")
                else:
                    st.warning("분석할 수 있는 수치형 변수 쌍이 충분하지 않습니다.")

        with col2:
            if st.button("📉 가장 강한 음의 상관관계 보기"):
                if not min_corr.empty:
                    st.error("💔 **가장 강한 음의 상관관계**")
                    pair = min_corr.index[0]
                    value = min_corr.values[0]
                    st.metric(
                        label=f"변수 쌍: **{pair[0]}**와 **{pair[1]}**", 
                        value=f"{value:.4f}",
                        delta="음의 상관관계"
                    )
                    st.info(f"👈 **{pair[0]}**의 값이 증가하면 **{pair[1]}**의 값은 감소하는 경향이 가장 강합니다.")
                else:
                    st.warning("분석할 수 있는 수치형 변수 쌍이 충분하지 않습니다.")

        st.markdown("---")
        with st.expander("📊 데이터 미리보기 (상관관계 분석에 사용된 데이터)"):
            st.dataframe(data.head())
            st.caption(f"총 {len(data)}개의 행과 {len(data.columns)}개의 수치형 속성을 사용했습니다.")

    else:
        st.warning("데이터 로드에 실패하여 앱을 실행할 수 없습니다. 파일 경로 및 형식 확인이 필요합니다.")

if __name__ == "__main__":
    main()
