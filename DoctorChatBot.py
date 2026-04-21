import os
import streamlit as st

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"


@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    return db


def main():
    st.title("Ask Doctor ChatBot!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            groq_api_key = os.environ.get("GROQ_API_KEY")
            groq_model_name = "llama-3.1-8b-instant"

            llm = ChatGroq(
                model=groq_model_name,
                temperature=0.5,
                max_tokens=512,
                api_key=groq_api_key,
            )

            retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

            combine_docs_chain = create_stuff_documents_chain(
                llm,
                retrieval_qa_chat_prompt,
            )

            rag_chain = create_retrieval_chain(
                vectorstore.as_retriever(search_kwargs={"k": 3}),
                combine_docs_chain,
            )

            response = rag_chain.invoke({"input": prompt})

            result = response["answer"]
            source_documents = response["context"]

            st.chat_message("assistant").markdown(result)
            st.session_state.messages.append(
                {"role": "assistant", "content": result}
            )

            with st.expander("Source documents"):
                for doc in source_documents:
                    st.write(doc.metadata)
                    st.write(doc.page_content[:300] + "...")

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()