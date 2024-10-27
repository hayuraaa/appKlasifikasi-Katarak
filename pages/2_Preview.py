import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit_extras.jupyterlite import jupyterlite

# Baca dataset
@st.cache_data
def load_data():
    df = pd.read_csv('dataset.csv')
    return df


st.title("📈 History Pelatihan Model")

# Overview pelatihan model
st.markdown("""
<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
    <h4>🔬 Proses Pengembangan Model</h4>
    <p>Halaman ini menampilkan proses lengkap pengembangan model klasifikasi katarak menggunakan algoritma Random Forest.
    Anda dapat melihat tahapan preprocessing data, pelatihan model, dan evaluasi performa.</p>
</div>
""", unsafe_allow_html=True)

df = load_data()

# Tab untuk berbagai bagian pelatihan
tab1, tab2, tab3 = st.tabs(["💾 Dataset", "🤖 Model Training", "📊 Evaluasi"])

with tab1:
    st.header("Data yang Digunakan")
    
    # Tampilkan dataset
    st.markdown("### 📋 Preview Dataset")
    st.dataframe(df.head(10))
    
    # Informasi dataset
    st.markdown("### ℹ️ Informasi Dataset")
    buffer = st.expander("Lihat informasi detail dataset")
    with buffer:
        df_info = pd.DataFrame({
            'Kolom': df.columns,
            'Tipe Data': df.dtypes,
            'Null Values': df.isnull().sum(),
            'Unique Values': df.nunique()
        })
        st.dataframe(df_info)
    
    # Distribusi data
    st.markdown("### 📊 Visualisasi Data")
    
    # Pilih fitur untuk visualisasi
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    selected_feature = st.selectbox("Pilih fitur untuk visualisasi:", numeric_columns)
    
    # Plot distribusi
    fig = px.histogram(df, x=selected_feature, color='katarak',
                      title=f'Distribusi {selected_feature} berdasarkan Diagnosis',
                      marginal="box")
    st.plotly_chart(fig)
    
with tab2:
    st.header("Proses Pelatihan Model")
    st.markdown("""
    ### 🛠️ Parameter Model
    ```python
    param_grid = {
        'n_estimators': [200, 300, 400, 500],
        'max_depth': [15, 20, 25, 30, None],
        'min_samples_split': [2, 4, 6, 8],
        'min_samples_leaf': [1, 2, 3],
        'criterion': ['entropy', 'gini'],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    ```

    ### 📊 Parameter Terbaik
    - n_estimators: 100
    - max_depth: 20
    - min_samples_split: 2
    - criterion: entropy
    - class_weight: balanced
    """)
    
    # Tampilkan kode training
    code_training = '''
    # Kode training model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV

    # Initialize model
    rf_model = RandomForestClassifier(random_state=42)

    # Perform Grid Search
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )

    # Train model
    grid_search.fit(X_train, y_train)
    '''
    st.code(code_training, language='python')
    
with tab3:
    st.header("Evaluasi Model")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", "0.95")
    with col2:
        st.metric("Precision", "1.00")
    with col3:
        st.metric("Recall", "0.9630")
    with col4:
        st.metric("F1 Score", "1.000")
    
    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    code_matrix = '''
    # Visualisasi Confusion Matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    '''
    st.code(code_matrix, language='python')
    
    # Feature Importance
    st.markdown("### Feature Importance")
    code_importance = '''
    # Visualisasi Feature Importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance)
    plt.title('Feature Importance dalam Prediksi Katarak')
    plt.show()
    '''
    st.code(code_importance, language='python')

# Tambahkan link ke notebook lengkap
st.markdown("""
---
### 📚 Resources
- [Lihat Notebook Lengkap di Google Colab](https://colab.research.google.com/drive/1z5dVyMgu-MpvsDq7r8zt_IjmQrIqZHUE#scrollTo=8H6kfgc5e9_M)
- [Download Dataset](LINK_DATASET_ANDA)
""")