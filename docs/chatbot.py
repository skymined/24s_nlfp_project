from io import BytesIO

import streamlit as st
from assistant import EnhancedChatbot
from openai import OpenAI
import os

# if __name__ == "__main__":
#     api_key = os.getenv('api_key')
#     model_name = "jhgan/ko-sbert-multitask"
#     db_path = "./data/dataset_and_embeddings_text_test.pkl"
#     # today_conversation = []

#     chatbot = EnhancedChatbot(api_key, model_name, db_path)  # class 가지고 옴

#     st.title("Chatbot Application")
#     user_input = st.text_input("User:")

#     if 'conversation' not in st.session_state:
#         st.session_state.conversation = []

#     if user_input:
#         response = chatbot.get_response(user_input)
#         st.session_state.conversation.append((user_input, response))
#         user_input = ""

#     for user, bot in st.session_state.conversation:
#         st.text(f"User: {user}")
#         st.text(f"Chatbot: {bot}")

#     if st.button("이제 요약해줘!><"):
#         summary = chatbot.summarize_conversation()  # 요약 실행
#         st.text_area("Summary:", summary)
#         st.session_state.conversation = []
#         st.experimental_rerun()


# streamlit run app.py 로 실행

# 챗봇 사용 예시
if __name__ == "__main__":
    api_key = os.getenv('api_key')
    model_name = "jhgan/ko-sbert-multitask"
    db_path = "./data/dataset_and_embeddings_text_test.pkl"

    chatbot = EnhancedChatbot(api_key, model_name, db_path)  # 클래스 인스턴스 생성

    st.set_page_config(layout="wide")  # 화면 레이아웃을 와이드로 설정

    st.title("Chatbot Application")

    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    if 'summaries' not in st.session_state:
        st.session_state.summaries = []

    # 사이드바 요약 저장 및 보기
    with st.sidebar:
        st.header("Saved Summaries")
        for summary in st.session_state.summaries:
            st.text(summary)

        if st.button("이제 요약해줘!><"):
            summary = chatbot.summarize_conversation()
            st.session_state.summaries.append(summary)
            chatbot.save_database()  # save_database 호출
            st.experimental_rerun()

    # CSS 스타일 적용
    st.markdown("""
        <style>
        .chat-container {
            max-width: 800px;
            margin: auto;
            padding: 20px;
            overflow-y: auto;
            height: calc(100vh - 150px); /* Adjust height to make room for input */
        }
        .user-message {
            background-color: #1E88E5;
            color: #ffffff;
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
            text-align: left;
            width: fit-content;
            max-width: 80%;
        }
        .bot-message {
            background-color: #424242;
            color: #ffffff;
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
            text-align: left;
            width: fit-content;
            max-width: 80%;
        }
        .message-row {
            display: flex;
            align-items: flex-start;
            margin: 5px 0;
        }
        .message-row.user {
            justify-content: flex-end;
        }
        .message-row.bot {
            justify-content: flex-start;
        }
        .input-container {
            position: sticky;
            bottom: 0;
            width: 100%;
            background: white;
            padding: 10px 0;
            box-shadow: 0 -1px 5px rgba(0,0,0,0.1);
            z-index: 10; /* Ensure the input container is above other content */
        }
        </style>
    """, unsafe_allow_html=True)

    # 채팅 창 표시
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for user, bot in st.session_state.conversation:
            st.markdown(f'<div class="message-row user"><div class="message user-message">{user}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="message-row bot"><div class="message bot-message">{bot}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 입력 필드와 전송 버튼을 페이지 하단에 고정
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("User:", key="user_input", label_visibility='collapsed')
        submitted = st.form_submit_button("Send")

        if submitted and user_input:
            # 사용자 메시지를 먼저 추가
            st.session_state.conversation.append((user_input, ""))

            # 챗봇 응답 가져오기
            response = chatbot.get_response(user_input)

            # 마지막에 있는 사용자 메시지에 챗봇 응답을 추가
            st.session_state.conversation[-1] = (user_input, response)
            chatbot.save_conversation(user_input, response)  # save_conversation 호출

            # 리렌더링
            st.experimental_rerun()
    st.markdown('</div>', unsafe_allow_html=True)