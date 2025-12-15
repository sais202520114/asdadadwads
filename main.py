import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 사용자님이 요청하신 파일명으로 정확히 설정
FILE_PATH = "titanic.xls"

# Streamlit 페이지 설정
st.set_page_config(
    page_title="타이타닉 데이터 상관관계 분석",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 데이터 로드 및 전처리 함수 ---
@st.cache_data
def load_data(file_path):
    """엑셀(.xls) 파일을 로드하고 수치형 변수만 선택합니다."""
    try:
        # 파일명이 .xls이므로 pd.read_excel을 사용하여 엑셀 파일을 로드합니다.
        # 사용자님이 업로드하신 파일이 실제로는 CSV 데이터 구조를 가지고 있지만,
        # 파일명을 존중하여 .xls 처리를 시도합니다.
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        st.error(f"오류: 파일을 찾을 수 없습니다. '{file_path}' 파일이 앱과 같은 위치에 있는지 확인해 주세요.")
        return None
    except ImportError:
        st.error("오류: 엑셀 파일 로드를 위해 'openpyxl' 또는 'xlrd' 라이브러리가 필요합니다. 'requirements.txt'를 확인해 주세요.")
        return None
    except Exception as e:
        st.error(f"데이터 로딩 중 오류 발생: {e}")
        return None

    # 수치형 변수만 선택
    numeric_df = df.select_dtypes(include=['number']).copy()
    
    # 결측값(NaN)을 해당 열의 평균값으로 대체
    numeric_df = numeric_df.fillna(numeric_df.mean())
    
    if numeric_df.shape[1] < 2:
        st.warning("분석할 수 있는 수치형 변수가 2개 미만입니다.")
        return None
        
    return numeric_df

# --- 상관관계 분석 함수 ---
def calculate_correlation(df):
    """데이터프레임의 상관관계를 계산하고 결과를 반환합니다."""
    corr_matrix = df.corr()
    
    # 자기 자신과의 상관관계(1)를 NaN으로 처리
    np.fill_diagonal(corr_matrix.values, float('nan'))
    
    # 상관 행렬을 시리즈 형태로 변환하고 중복되는 쌍을 제거
    corr_unstacked = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
    
    # NaN 값 제거
    valid_corr = corr_unstacked.dropna()
    
    # 가장 높은 양의 상관관계
    max_corr = valid_corr.head(1)
    
    # 가장 높은 음의 상관관계
    min_corr = valid_corr.tail(1)
    
    return corr_matrix, max_corr, min_corr

# --- 메인 앱 로직 ---
def main():
    st.title("🚢 타이타닉호 속성 간 상관관계 분석기")
    st.markdown(f"**현재 코드에 사용된 데이터 파일명:** `{FILE_PATH}`")
    st.markdown("---")
    
    data = load_data(FILE_PATH)

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

        st.markdown("---")
        st.header("🔎 버튼으로 주요 상관관계 확인하기")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📈 가장 강한 양의 상관관계 보기"):
                if not max_corr.empty:
                    pair = max_corr.index[0]
                    value = max_corr.values[0]
                    st.success("✨ **가장 강한 양의 상관관계**")
                    st.metric(
                        label=f"변수 쌍: **{pair[0]}**와 **{pair[1]}**", 
                        value=f"{value:.4f}",
                        delta="양의 상관관계"
                    )
                else:
                    st.warning("분석할 수 있는 유효한 양의 상관관계 쌍이 없습니다.")

        with col2:
            if st.button("📉 가장 강한 음의 상관관계 보기"):
                if not min_corr.empty:
                    pair = min_corr.index[0]
                    value = min_corr.values[0]
                    st.error("💔 **가장 강한 음의 상관관계**")
                    st.metric(
                        label=f"변수 쌍: **{pair[0]}**와 **{pair[1]}**", 
                        value=f"{value:.4f}",
                        delta="음의 상관관계"
                    )
                else:
                    st.warning("분석할 수 있는 유효한 음의 상관관계 쌍이 없습니다.")

        st.markdown("---")
        with st.expander("📊 데이터 미리보기"):
            st.dataframe(data.head())

    else:
        st.warning("데이터 로드에 실패하여 앱을 실행할 수 없습니다.")

if __name__ == "__main__":
    main()
