import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Define the file path
FILE_PATH = "titanic.xls"

# --- Matplotlib Font Reset for maximum stability ---
# Removed all Korean font search logic as requested.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False 

# Streamlit Page Setup
st.set_page_config(
    page_title="Titanic Data Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Loading and Preprocessing Function ---
@st.cache_data
def load_data(file_path):
    """Loads the Excel file and performs necessary data cleaning."""
    try:
        df = pd.read_excel(file_path)
    except Exception:
        st.error(f"Error: Could not find file or 'xlrd' library is not installed. Check file path ('{file_path}') and requirements.txt.")
        return None
    
    # Select key columns and handle missing values
    df_clean = df[['pclass', 'survived', 'sex', 'age', 'fare']].copy()

    # Imputation and type conversion
    df_clean['pclass'] = df_clean['pclass'].fillna(df_clean['pclass'].mode()[0]).astype(int)
    df_clean['survived'] = df_clean['survived'].fillna(0).astype(int)
    df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())
    df_clean['fare'] = df_clean['fare'].fillna(df_clean['fare'].median())
    
    # Create Age Group for counting plots
    bins = [0, 10, 20, 30, 40, 50, 60, 100]
    labels = ['0-10s', '10-20s', '20-30s', '30-40s', '40-50s', '50-60s', '60s+']
    df_clean['age_group'] = pd.cut(df_clean['age'], bins=bins, labels=labels, right=False)

    # Create target columns
    df_clean['Death'] = 1 - df_clean['survived']
    df_clean['Survival'] = df_clean['survived']
    
    return df_clean

# --- Summary Table Function ---
def generate_summary_tables(df):
    st.title("Titanic Data Analysis Summary Tables")
    st.markdown(f"**Data File:** `{FILE_PATH}`")
    st.markdown("---")
    
    total_deaths = df['Death'].sum()
    st.header(f"💔 Total Deaths: {total_deaths}")
    st.subheader("Detailed Death Analysis")
    
    col_d1, col_d2 = st.columns(2)
    
    # Age Group Death Summary
    age_death_summary = df.groupby('age_group')['Death'].sum().reset_index()
    age_death_summary = age_death_summary.rename(columns={'age_group': 'Age Group', 'Death': 'Death Count'})
    with col_d1:
        st.caption("Death Count by Age Group")
        st.dataframe(age_death_summary.set_index('Age Group'))
        
    # PClass Death Summary
    class_death_summary = df.groupby('pclass')['Death'].sum().reset_index()
    class_death_summary = class_death_summary.rename(columns={'pclass': 'Passenger Class', 'Death': 'Death Count'})
    class_death_summary['Passenger Class'] = class_death_summary['Passenger Class'].astype(str) + ' Class'
    with col_d2:
        st.caption("Death Count by Passenger Class")
        st.dataframe(class_death_summary.set_index('Passenger Class'))

    st.markdown("---")

    total_survival = df['Survival'].sum()
    st.header(f"✅ Total Survivors: {total_survival}")
    st.subheader("Detailed Survival Analysis")
    
    col_s1, col_s2 = st.columns(2)

    # Age Group Survival Summary
    age_survival_summary = df.groupby('age_group')['Survival'].sum().reset_index()
    age_survival_summary = age_survival_summary.rename(columns={'age_group': 'Age Group', 'Survival': 'Survival Count'})
    with col_s1:
        st.caption("Survival Count by Age Group")
        st.dataframe(age_survival_summary.set_index('Age Group'))
        
    # PClass Survival Summary
    class_survival_summary = df.groupby('pclass')['Survival'].sum().reset_index()
    class_survival_summary = class_survival_summary.rename(columns={'pclass': 'Passenger Class', 'Survival': 'Survival Count'})
    class_survival_summary['Passenger Class'] = class_survival_summary['Passenger Class'].astype(str) + ' Class'
    with col_s2:
        st.caption("Survival Count by Passenger Class")
        st.dataframe(class_survival_summary.set_index('Passenger Class'))
    
    st.markdown("---")

# --- Visualization Function ---

def plot_counts(df, category, target, target_name, plot_type, extreme_select):
    """Plots Death/Survival counts as bar or line charts."""
    
    if category == 'age':
        plot_data = df.groupby('age_group')[target].sum().reset_index()
        x_col = 'age_group'
        x_label = 'Age Group'
    else: # pclass
        plot_data = df.groupby(category)[target].sum().reset_index()
        x_col = category
        x_label = 'Passenger Class'
        plot_data[x_col] = plot_data[x_col].astype(str) + ' Class'

    total_sum = plot_data[target].sum()
    st.info(f"**Total {target_name} Count by {x_label}:** `{total_sum}`")
    
    st.subheader(f"📊 {target_name} by {x_label}")

    # === Fixed Plot Size: (6, 4) ===
    fig, ax = plt.subplots(figsize=(6, 4))
    
    if plot_type == 'Bar Chart':
        sns.barplot(x=x_col, y=target, data=plot_data, ax=ax, palette='YlGnBu', errorbar=None)
        
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 5), 
                        textcoords='offset points', 
                        fontsize=8)
            
    elif plot_type == 'Line Chart':
        sns.lineplot(x=x_col, y=target, data=plot_data, ax=ax, marker='o', color='blue')
        
        for x, y in zip(plot_data[x_col], plot_data[target]):
            ax.annotate(f'{int(y)}', (x, y), 
                        textcoords="offset points", 
                        xytext=(0, 8), 
                        ha='center', 
                        fontsize=8)
        
    ax.set_title(f"{target_name} by {x_label} ({plot_type})", fontsize=12)
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(target_name, fontsize=10)
    st.pyplot(fig) 
    
    max_val = plot_data[target].max()
    min_val = plot_data[target].min()
    
    if extreme_select == 'Highest Point':
        extreme_data = plot_data[plot_data[target] == max_val]
        extreme_label = 'Highest Point'
        st.success(f"🥇 **{extreme_label}:** {extreme_data[x_col].iloc[0]} ({max_val})")
    else:
        extreme_data = plot_data[plot_data[target] == min_val]
        extreme_label = 'Lowest Point'
        st.error(f"🥉 **{extreme_label}:** {extreme_data[x_col].iloc[0]} ({min_val})")


def plot_correlation(df, corr_type, plot_type):
    """Plots correlation as a scatter plot or heatmap."""
    
    numeric_df = df[['survived', 'pclass', 'age', 'fare']].copy()
    
    corr_matrix, max_corr, min_corr = calculate_correlation(numeric_df)
    
    st.header(f"🔗 Correlation Analysis Results ({plot_type})")
    
    if plot_type == 'Heatmap':
        # 1. Heatmap visualization
        # === Fixed Plot Size: (6, 6) ===
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # English labels for heatmap
        col_names = ['Survived', 'PClass', 'Age', 'Fare']
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
        st.pyplot(fig) 
        
        # 2. Print strongest correlation
        if corr_type == 'Positive Correlation':
            if not max_corr.empty:
                pair = max_corr.index[0]
                value = max_corr.values[0]
                st.success(f"📈 **Strongest Positive Correlation:** **{pair[0]}** and **{pair[1]}** (Coefficient: {value:.4f})")
            else:
                st.warning("No valid positive correlation pairs found.")
        else: # Negative Correlation
            if not min_corr.empty:
                pair = min_corr.index[0]
                value = min_corr.values[0]
                st.error(f"📉 **Strongest Negative Correlation:** **{pair[0]}** and **{pair[1]}** (Coefficient: {value:.4f})")
            else:
                st.warning("No valid negative correlation pairs found.")

    elif plot_type == 'Scatter Plot':
        # Scatter Plot visualization
        
        if corr_type == 'Positive Correlation':
            if not max_corr.empty:
                pair = max_corr.index[0]
                x_var, y_var = pair[0], pair[1]
                title_prefix = "Strongest Positive Correlation"
            else:
                # Fallback: Fare vs Age (Commonly positive)
                x_var, y_var = 'fare', 'age'
                title_prefix = "Positive Correlation (Fallback: Fare vs Age)"

        else: # Negative Correlation
            if not min_corr.empty:
                pair = min_corr.index[0]
                x_var, y_var = pair[0], pair[1]
                title_prefix = "Strongest Negative Correlation"
            else:
                # Fallback: Pclass vs Fare (Commonly negative)
                x_var, y_var = 'pclass', 'fare'
                title_prefix = "Negative Correlation (Fallback: PClass vs Fare)"

        st.subheader(f"Scatter Plot: {title_prefix} - {x_var} vs {y_var}")
        # === Fixed Plot Size: (6, 4) ===
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Use a more descriptive hue for survival status
        sns.scatterplot(x=x_var, y=y_var, data=df, ax=ax, hue='survived', palette='deep', legend='full') 
        
        ax.set_title(f"Relationship between {x_var} and {y_var} (Grouped by Survival)", fontsize=12)
        ax.set_xlabel(x_var, fontsize=10)
        ax.set_ylabel(y_var, fontsize=10)
        st.pyplot(fig) 

def calculate_correlation(df):
    """Calculates correlation matrix and extracts strongest non-trivial pairs."""
    corr_matrix = df.corr()
    
    # Fill diagonal (self-correlation) with NaN
    np.fill_diagonal(corr_matrix.values, np.nan) 
    
    corr_unstacked = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
    
    valid_corr = corr_unstacked.dropna()
    
    # Filter out values extremely close to 1 or -1 (addressing the 1/-1 issue)
    valid_corr = valid_corr[abs(valid_corr) < 0.999999] 

    max_corr = valid_corr.head(1)
    min_corr = valid_corr.tail(1)
    
    return corr_matrix, max_corr, min_corr

# --- Main App Logic ---
def main():
    
    data = load_data(FILE_PATH)
    if data is None:
        return

    # ------------------
    # 1. Sidebar Menu Setup
    # ------------------

    st.sidebar.title("Menu Selection")
    
    graph_type = st.sidebar.radio(
        "📊 Select Analysis Type",
        ('Summary Tables', 'Death/Survival Count Analysis', 'Correlation Analysis')
    )
    
    st.sidebar.markdown("---")
    
    # ------------------
    # 2. Main Screen Layout
    # ------------------
    
    if graph_type == 'Summary Tables':
        generate_summary_tables(data)

    elif graph_type == 'Death/Survival Count Analysis':
        
        analysis_theme = st.sidebar.radio(
            "🔎 Select Analysis Subject",
            ('Death Count', 'Survival Count')
        )

        if analysis_theme == 'Death Count':
            target_col = 'Death'
            target_name = 'Death Count'
        else: 
            target_col = 'Survival'
            target_name = 'Survival Count'
            
        category_options = {
            f'By Age Group ({target_name})': 'age',
            f'By PClass ({target_name})': 'pclass'
        }
            
        selected_category_name = st.sidebar.selectbox(
            f"Select Breakdown Category",
            options=list(category_options.keys()),
            index=0
        )
        selected_category_col = category_options[selected_category_name]
        
        st.sidebar.markdown("---")
        
        plot_style = st.sidebar.radio(
            "📈 Select Visualization Type",
            ('Bar Chart', 'Line Chart')
        )
        
        st.sidebar.markdown("---")

        extreme_select = st.sidebar.radio(
            "⬆️ Highlight Point",
            ('Highest Point', 'Lowest Point'),
            index=0 
        )
        
        plot_counts(data, selected_category_col, target_col, target_name, plot_style, extreme_select)


    elif graph_type == 'Correlation Analysis':
        
        corr_type = st.sidebar.radio(
            "🔗 Select Correlation Direction",
            ('Positive Correlation', 'Negative Correlation')
        )
        
        st.sidebar.markdown("---")
        
        corr_plot_type = st.sidebar.radio(
            "📊 Select Visualization Type",
            ('Scatter Plot', 'Heatmap')
        )
        
        plot_correlation(data, corr_type, corr_plot_type)
        
        
if __name__ == "__main__":
    main()
