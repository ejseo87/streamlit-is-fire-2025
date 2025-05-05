import os
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


st.set_page_config(
    page_title="Streamlit is 🔥",
    page_icon=":fire:",
)

# Create necessary directories in /tmp
CACHE_DIR = "temp/.cache"
os.makedirs(os.path.join(CACHE_DIR, "files"), exist_ok=True)
os.makedirs(os.path.join(CACHE_DIR, "embeddings"), exist_ok=True)


with st.sidebar:
    st.markdown("GitHub: [https://github.com/ejseo87/streamllit-is-fire-2025.git](https://github.com/ejseo87/streamllit-is-fire-2025.git)")
    st.markdown("---")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    file = st.file_uploader("Upload a .txt .pdf or .docx file", type=[
        "pdf", "docx", "txt"])

# Initialize ChatOpenAI with API key from user input
if openai_api_key:
    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[
            ChatCallbackHandler(),
        ],
        model="gpt-3.5-turbo",
        openai_api_key=openai_api_key
    )
else:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = os.path.join(CACHE_DIR, "files", file.name)
    # Save the file
    with open(file_path, "wb") as f:
        f.write(file_content)
    st.success(f"File [{file.name}] uploaded successfully!")
    cache_dir = LocalFileStore(os.path.join(
        CACHE_DIR, "embeddings", file.name))
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir)
    vectorstore = FAISS.from_documents(
        documents=docs,
        embedding=cached_embeddings
    )
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append(
        {"role": role, "message": message})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
        if save:
            save_message(message, role)


def paint_history():
    for message in st.session_state.messages:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
     Answer the question using ONLY the following context.
     If you don't know the answer, say "I don't know".
     Don't make anything up.
     ---
     Context: {context}
     """,
     ),
    ("human", "{question}"),
])

st.title("Streamlit is 🔥")

st.markdown("""
Welcome!

Use this chatbot to ask questions to an AI about your file!

Upload your files on the sidebar.
""")

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about the document")
    if message:
        send_message(message, "human")
        chain = {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        } | prompt | llm
        with st.chat_message("ai"):
            response = chain.invoke(message)
else:
    st.session_state["messages"] = []
