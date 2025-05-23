import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

st.title('Iris model inference!')

with st.sidebar:
    st.header('Data requirements')
    st.caption('To inference the model you need to upload a dataframe in csv format with four columns/features (columns names are not important)')
    with st.expander('Data format'):
        st.markdown(' - utf-8')
        st.markdown(' - separated by coma')
        st.markdown(' - delimited by "."')
        st.markdown(' - first row - header')       
    st.divider() 
    st.caption("<p style = 'text-align:center'>Developed by me</p>", unsafe_allow_html = True)

if 'clicked' not in st.session_state:
    st.session_state.clicked = {1:False}

def clicked(button):
    st.session_state.clicked[button] = True

st.button("Let's get started", on_click = clicked, args = [1])

if st.session_state.clicked[1]:
    uploaded_file = st.file_uploader("Choose a file", type='csv')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, low_memory = True)
        st.header('Uploaded data sample')
        st.write(df.head())
        model = joblib.load('model.joblib')
        pred = model.predict_proba(df)
        pred = pd.DataFrame(pred, columns = ['setosa_probability', 'versicolor_probability', 'virginica_probability'])
        st.header('Predicted values')
        st.write(pred.head())

        pred = pred.to_csv(index=False).encode('utf-8')
        st.download_button('Download prediction',
                        pred,
                        'prediction.csv',
                        'text/csv',
                        key='download-csv')
