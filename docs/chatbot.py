from io import BytesIO

import streamlit as st
from assistant import EnhancedChatbot
from openai import OpenAI
import os
import time

if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = "jhgan/ko-sbert-multitask"
    db_path = "./data/dataset_and_embeddings_text_test.pkl"

    chatbot = EnhancedChatbot(api_key, model_name, db_path)  # 클래스 인스턴스 생성

    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    if 'summaries' not in st.session_state:
        st.session_state.summaries = []

    # 사이드바 요약 저장 및 보기
    with st.sidebar:
        st.header("F의 운동일기")
        for summary in st.session_state.summaries:
            st.text(summary)

        if st.button("이제 요약해줘!><"):
            # 대화 내용을 chatbot의 current_conversation에 전달
            chatbot.current_conversation = st.session_state.conversation
            summary = chatbot.summarize_conversation()
            st.session_state.summaries.append(summary)
            chatbot.save_database()  # save_database 호출
            st.experimental_rerun()

    # CSS 스타일 적용
    st.markdown("""
        <style>
        .main-container {
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        .input-container {
            padding: 10px;
            background: white;
            box-shadow: 0 -1px 5px rgba(0,0,0,0.1);
        }
        .input-container form {
            display: flex;
            width: 100%;
            max-width: 800px;
            margin: auto;
        }
        .input-container input[type="text"] {
            flex-grow: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin-right: 10px;
        }
        .input-container button {
            padding: 10px 20px;
            background-color: #1E88E5;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
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
        </style>
    """, unsafe_allow_html=True)

    # 전체 컨테이너
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # 채팅 창 표시
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for user, bot in st.session_state.conversation:
            st.markdown(f'<div class="message-row user"><div class="user-message">{user}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="message-row bot"><div class="bot-message">{bot}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 입력 필드와 전송 버튼을 페이지 하단에 고정
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("User:", key="user_input", label_visibility='collapsed')
        submitted = st.form_submit_button("Send")

        if submitted and user_input:
            # 사용자 메시지를 먼저 추가
            st.session_state.conversation.append((user_input, ""))

            # 응답 생성 전체 시간 측정 시작
            total_start_time = time.time()

            # 챗봇 응답 가져오기
            response = chatbot.get_response(user_input)

            # 응답 생성 전체 시간 측정 종료
            total_end_time = time.time()
            total_elapsed_time = total_end_time - total_start_time

            print(f"응답 생성 전체 시간: {total_elapsed_time:.2f}초")

             # 화면에 나타나는 시간 측정 시작
            render_start_time = time.time()


            # 마지막에 있는 사용자 메시지에 챗봇 응답을 추가
            st.session_state.conversation[-1] = (user_input, response)
            chatbot.current_conversation = st.session_state.conversation
            chatbot.save_conversation(user_input, response)  # save_conversation 호출
            
            # 화면에 나타나는 시간 측정 종료
            render_end_time = time.time()
            render_elapsed_time = render_end_time - render_start_time
            print(f"화면에 나타나는 시간: {render_elapsed_time:.2f}초")

            # 리렌더링
            st.experimental_rerun()
    st.markdown('</div>', unsafe_allow_html=True)