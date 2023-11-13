import streamlit as st
from pipeline import llm_pipeline 
import base64
from langdetect import detect
from gtts import gTTS
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import os
st.set_page_config(layout="wide")

# Custom Color Theme
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #4F8BF9;  # Change the background color
    }
    .sidebar .sidebar-content {
        background-color: #FAFAFA;  # Change the sidebar background color
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_data()
def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width='100%' height=720 type="application/pdf"></iframe>"""
    return pdf_display

@st.cache()
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud)
    plt.axis("off")
    return plt

def download_summary(summary):
    st.download_button('Download Summary', summary, file_name='summary.txt')
def main():
    st.title("Document Summarization App")

    uploaded_file = st.file_uploader("Upload your file", type=['pdf', 'txt', 'docx', 'html'])

    if uploaded_file is not None:
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.getbuffer())

        if st.button("Summarize"):
            with st.spinner('Summarizing...'):
                try:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.info("Uploaded File")
                        if uploaded_file.type == "application/pdf":
                            st.markdown(display_pdf(file_path), unsafe_allow_html=True)
                        else:
                            st.text(f'Uploaded {uploaded_file.type} file.')

                    with col2:
                        summary = llm_pipeline(file_path)
                        st.info("Summarization Complete")
                        st.success(summary)

                        # Word Cloud
                        st.subheader("Key Concepts Visualization")
                        plt = generate_wordcloud(summary)
                        st.pyplot(plt)

                        # Download Summary
                        st.subheader("Download Summary")
                        download_summary(summary)

                        # Text-to-Speech
                        if st.button('Listen to Summary'):
                            tts = gTTS(summary)
                            tts.save('summary.mp3')
                            audio_file = open('summary.mp3', 'rb')
                            audio_bytes = audio_file.read()
                            st.audio(audio_bytes, format='audio/mp3')
                            os.remove('summary.mp3')

                except Exception as e:
                    st.error(f"An error occurred: {e}")
        os.remove(file_path)

if __name__ == "__main__":
    main()
