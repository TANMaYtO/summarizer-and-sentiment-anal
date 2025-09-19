import streamlit as st
from transformers import pipeline

st.set_page_config(page_title='Summarizer',layout='wide')
@st.cache_resource
def load_model():
    models= {}
    models['summ']= pipeline('summarization', model='facebook/bart-large-cnn')
    models['sent']= pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    return models

st.title('Summarizer and Sentiment Anal!!')
st.write('paste the text to get the summary as well as sentiment anal')
col1, col2= st.columns([3,1])
with col2:
    max_len= st.slider('Max Summary length: ',30,400,120)
    min_len= st.slider('Min summary length:', 10,200,30)

text= col1.text_area('Text to summarize', height=300, placeholder='enter text here...')
if st.button('Anal!!'):
    if not text.strip():
        st.warning('PLEAES ENTER TEXT FIRST')
    else:
        with st.spinner('Summarizing...'):
            models= load_model()
            try:
                summary= models['summ']()