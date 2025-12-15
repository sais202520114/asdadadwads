# ... (앞부분 생략)

def plot_correlation(df, corr_type, plot_type):
    """상관관계를 산점도 또는 히트맵으로 그립니다. (내부 라벨은 영어)"""
    
    # 연속형 변수 + survived 만 상관관계 행렬에 포함
    numeric_df = df[['survived', 'age', 'fare']].copy() 
    
    corr_matrix, max_corr, min_corr = calculate_correlation(numeric_df)
    
    st.header(f"🔗 상관관계 분석 결과 ({plot_type})")
    
    if plot_type == 'Heatmap':
        # 히트맵 로직 (변동 없음)
        plt.figure(figsize=(6, 6))
        fig, ax = plt.subplots(figsize=(6, 6))
        
        col_names = ['Survived', 'Age', 'Fare']
        corr_matrix.columns = col_names
        corr_matrix.index = col_names
        
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap='YlGnBu', 
            cbar=True,
            linewidths=0.5,
            linecolor='black',
            annot_kws={"size": 9},
            ax=ax
        )
        ax.set_title("Correlation Heatmap of Titanic Attributes", fontsize=12)
        st.pyplot(fig, use_container_width=False) 
        
        # 강한 상관관계 텍스트 출력
        if corr_type == '양의 상관관계':
            if not max_corr.empty:
                pair = max_corr.index[0]
                value = max_corr.values[0]
                st.success(f"📈 **가장 강한 양의 상관관계:** **{pair[0].capitalize()}**와 **{pair[1].capitalize()}** (계수: {value:.4f})")
            else:
                st.warning("분석할 수 있는 유효한 양의 상관관계 쌍이 없습니다.")
        else: # 음의 상관관계
            if not min_corr.empty:
                pair = min_corr.index[0]
                value = min_corr.values[0]
                st.error(f"📉 **가장 강한 음의 상관관계:** **{pair[0].capitalize()}**와 **{pair[1].capitalize()}** (계수: {value:.4f})")
            else:
                st.warning("분석할 수 있는 유효한 음의 상관관계 쌍이 없습니다.")

    elif plot_type == 'Scatter Plot':
        # === 산점도 로직: 이진 변수 축 사용 금지 (X=Age, Y=Fare로 고정) ===
        
        x_var, y_var = 'age', 'fare' 
        
        if corr_type == '양의 상관관계':
            # 양의 상관관계는 주로 Age와 Fare 사이에서 발생합니다.
            title_prefix = "Strongest Positive Correlation (Age vs Fare)"
        else: # 음의 상관관계
            # 음의 상관관계는 주로 Survived와 Age/Fare 사이에서 발생합니다.
            # 산점도에서는 Age와 Fare의 분포를 Survived로 색칠하여 간접적으로 확인합니다.
            title_prefix = "Distribution Analysis for Negative Correlation (Age vs Fare)"
        
        # 2. 산점도 시각화
        st.subheader(f"산점도: {title_prefix}")
        
        plt.figure(figsize=(6, 4))
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # X, Y 축에 연속형 변수 Age와 Fare만 사용, Survived는 색상(hue)으로만 사용
        sns.scatterplot(x=x_var, y=y_var, data=df, ax=ax, hue='survived', palette='deep', legend='full') 
        
        # 3. 축 라벨과 포맷팅
        ax.set_title(f"Scatter Plot: {x_var.capitalize()} vs {y_var.capitalize()} (Grouped by Survival)", fontsize=12)
        ax.set_xlabel(x_var.capitalize(), fontsize=10)
        ax.set_ylabel(y_var.capitalize(), fontsize=10)
        
        ax.ticklabel_format(style='plain', useOffset=False, axis='x')
        ax.ticklabel_format(style='plain', useOffset=False, axis='y')
            
        st.pyplot(fig, use_container_width=False) 

def calculate_correlation(df):
# ... (calculate_correlation 함수 변동 없음)
    corr_matrix = df.corr()
    np.fill_diagonal(corr_matrix.values, np.nan) 
    corr_unstacked = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
    valid_corr = corr_unstacked.dropna()
    valid_corr = valid_corr[abs(valid_corr) < 0.999999] 
    max_corr = valid_corr.head(1)
    min_corr = valid_corr.tail(1)
    return corr_matrix, max_corr, min_corr

# ... (나머지 main 함수 및 기타 함수 변동 없음)

# --- 메인 앱 로직 (UI는 한국어) ---
def main():
    
    data = load_data(FILE_PATH)
    if data is None:
        return

    st.sidebar.title("메뉴 선택")
    
    graph_type = st.sidebar.radio(
        "📊 분석 유형 선택",
        ('종합 요약 (표)', '사망/구조자 수 분석 (그래프)', '상관관계 분석 (그래프)')
    )
    
    st.sidebar.markdown("---")
    
    if graph_type == '종합 요약 (표)':
        generate_summary_tables(data)

    elif graph_type == '사망/구조자 수 분석 (그래프)':
        
        analysis_theme_kor = st.sidebar.radio(
            "🔎 분석 주제 선택",
            ('사망자 수', '구조자 수')
        )

        if analysis_theme_kor == '사망자 수':
            target_col = 'Death'
            target_name = 'Death Count' # 그래프 라벨용
        else: 
            target_col = 'Survival'
            target_name = 'Survival Count' # 그래프 라벨용
            
        category_options = {
            f'연령별': 'age',
            f'선실 등급별': 'pclass'
        }
            
        selected_category_name = st.sidebar.selectbox(
            f"세부 분류 카테고리",
            options=list(category_options.keys()),
            index=0
        )
        selected_category_col = category_options[selected_category_name]
        
        st.sidebar.markdown("---")
        
        plot_style = st.sidebar.radio(
            "📈 시각화 유형 선택",
            ('Bar Chart', 'Line Chart')
        )
        
        st.sidebar.markdown("---")

        extreme_select_kor = st.sidebar.radio(
            "⬆️ 지점 강조 선택",
            ('가장 높은 지점', '가장 낮은 지점'),
            index=0 
        )
        
        plot_counts(data, selected_category_col, target_col, target_name, plot_style, extreme_select_kor)


    elif graph_type == '상관관계 분석 (그래프)':
        
        corr_type_kor = st.sidebar.radio(
            "🔗 상관관계 방향 선택",
            ('양의 상관관계', '음의 상관관계')
        )
        
        st.sidebar.markdown("---")
        
        corr_plot_type = st.sidebar.radio(
            "📊 시각화 유형 선택",
            ('Scatter Plot', 'Heatmap')
        )
        
        plot_correlation(data, corr_type_kor, corr_plot_type)
        
        
if __name__ == "__main__":
    main()
