import os
import streamlit as st
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
st.title("Legal Assistant")
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
embeddings=GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
vectorstore=FAISS.load_local("civilLaw_database", embeddings, allow_dangerous_deserialization=True)
print(vectorstore)
retriever = vectorstore.as_retriever()

system_prompt = (
    "You are an assistant specialized in Indian civil law. Only answer questions that are related to Indian civil law"
    "Provide a concise answer ONLY BASED ON CONTEXT. If the information is not in context, say 'I don't know'. "
    "List down the references with REFERENCES - format at the end of each response"
    "\n\n"
    "CONTEXT: {context}"
)

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is. ONLY answer questions that are related to Indian civil law"
    "List down the references with REFERENCES - format at the end of each response"
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

if "messages" not in st.session_state:
    st.session_state.messages = []

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if "session_id" not in st.session_state:
        st.session_state.session_id = ChatMessageHistory()
    return st.session_state.session_id

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


input=st.chat_input("Ask something")
if input:
    st.chat_message("user").markdown(input)
    st.session_state.messages.append({"role": "user", "content": input})
    output=conversational_rag_chain.invoke(
        {"input": input},
        config={
            "configurable": {"session_id": "abc123"}
        }, 
    )["answer"]
    with st.chat_message("assistant"):
        st.markdown(output)
    
    st.session_state.messages.append({"role": "assistant", "content": output})