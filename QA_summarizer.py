import streamlit as st
import os
import getpass
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA

def main():
    """
    Main function for the Streamlit application.
    """
    st.title("📄 PDF Q&A with LangChain")
    st.markdown("Upload a PDF, ask a question, and get an answer.")

    # --- API Key Input ---
   
    if not os.environ.get("OPENAI_API_KEY"):
        st.info("Enter your OpenAI API key to get started.")
        api_key = st.text_input("OpenAI API Key:", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("API key set!")
        else:
            st.warning("Please enter your API key to continue.")
            return

    # --- File Uploader ---
 
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        with st.spinner("Processing PDF..."):
            # Save the uploaded file temporarily
            with open("temp_doc.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # --- Document Loading and Splitting ---
            loader = PyPDFLoader("temp_doc.pdf")
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )
            split_docs = text_splitter.split_documents(docs)

            # --- Vector Store and Retriever ---
            try:
                embeddings = OpenAIEmbeddings()
                vector_store = InMemoryVectorStore.from_documents(split_docs, embeddings)
                retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            except Exception as e:
                st.error(f"Error creating vector store: {e}")
                st.stop()

            st.success("PDF processed successfully! You can now ask a question.")

            # --- LLM and QA Chain Setup ---
            llm = ChatOpenAI(model_name="gpt-5", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

            # --- User Input and Answer Display ---
            query = st.text_input("Your question:")
            if st.button("Get Answer"):
                if query:
                    try:
                        with st.spinner("Finding answer..."):
                            response = qa_chain.invoke({"query": query})
                            st.subheader("Answer")
                            st.write(response["result"])
                    except Exception as e:
                        st.error(f"An error occurred while getting the answer: {e}")
                else:
                    st.warning("Please enter a question.")

if __name__ == "__main__":
    main()