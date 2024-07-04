from io import BytesIO

import streamlit as st
from assistant import EnhancedChatbot
from openai import OpenAI
import os

api_key = os.getenv('api_key')
model_name = "jhgan/ko-sbert-multitask"
db_path = "./data/dataset_and_embeddings_text_test.pkl"
today_conversation =[]

chatbot = EnhancedChatbot(api_key, model_name, db_path) # class 가지고 옴

user_input = input("User: ")
if user_input.lower() == 'exit':
    print("오늘의 운동을 요약해줄게~~~><")
    # Accumulate conversation in today_conversation
    break

response = chatbot.get_response(user_input)
today_conversation.append((user_input, response))
print("User:", user_input)
print("Chatbot:", response)

# while True:
#     user_input = input("User: ")
#     if user_input.lower() == 'exit':
#         print("Chatbot: 대화를 종료합니다.")
#         # Accumulate conversation in today_conversation

#         break
#     response = chatbot.get_response(user_input)
#     today_conversation.append((user_input, response))
#     print("User:", user_input)
#     print("Chatbot:", response)




instructions = None
name = "성동일"
pdf_file_name = None
txt_file_name = None

client = ChatAssistant(name="성동일")
client.set_assistant(instructions=instructions, pdf_file=pdf_file_name, txt_file=txt_file_name) 

st.title(f"🤖 Chat Assistant Demo")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = client._retrieve_assistant(client.assistant_id).model

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

client._create_thread()

if prompt := st.chat_input(f"{name}씨와 대화를 나눠보세요!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        responses = client.get_answers(prompt)

        for response in responses:
            full_response += response
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})