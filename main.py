import streamlit as st
from transformers import pipeline

# Load the summarizer model once (outside the app flow for performance)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

st.title("📝 Text Summarizer")

# User input
text = st.text_area("Enter text to summarize:", height=200)

# When the user clicks the button
if st.button("Summarize"):
    if text.strip():
        with st.spinner('Summarizing...'):
            # Generate summary
            summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
            result = summary[0]['summary_text']

        st.subheader("Summary")
        st.write(result)
    else:
        st.warning("Please enter some text first.")
