# Streamlit 패키지 추가
import streamlit as st
# OpenAI 패키기 추가
import openai

with open('api.txt', 'r') as f:
    api_key = f.read().strip()

#메인 함수
def main():
    st.set_page_config(page_title="광고 문구 생성 프로그램")


# session state 초기화
    if "OPENAI_API" not in st.session_state:
        st.session_state["OPENAI_API"] = ""

# 사이드바
with st.sidebar:
    # Open AI API 키 입력받기
	# open_apikey = st.text_input(label='API KEY', placeholder='Enter Your API Key', value='',type='password')
    open_apikey = api_key
    
#메인공간
st.header("🎸광고 문구 생성 프로그램")
st.markdown('---')