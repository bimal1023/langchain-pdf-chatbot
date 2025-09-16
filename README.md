# Streamlit PDF Q&A App

This is a Streamlit web application that allows users to upload a PDF document and ask questions about its content. The application uses a Retrieval-Augmented Generation (RAG) pipeline built with LangChain to provide accurate, document-grounded answers.

## Features

-   **PDF Upload:** Upload any PDF file directly through the web interface.
-   **Intelligent Q&A:** Ask questions about the content of the PDF.
-   **LangChain Integration:** Utilizes LangChain for document loading, splitting, and the RAG pipeline.
-   **Streamlit UI:** A simple and intuitive user interface built entirely with Python.
-   **OpenAI Embeddings:** Converts document chunks into vectors for semantic search.

## Technologies Used

-   **[Streamlit](https://streamlit.io/)**: For the web application framework.
-   **[LangChain](https://www.langchain.com/)**: For building the LLM application pipeline.
-   **[OpenAI](https://openai.com/)**: Provides the embedding and chat models (`gpt-3.5-turbo`).
-   **[PyPDFLoader](https://python.langchain.com/docs/integrations/document_loaders/pypdfloader/)**: For loading PDF documents.

## How to Run the Application Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git
    cd <YOUR_REPO_NAME>
    ```

2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up your OpenAI API key:**
    The application will prompt you for your key in the web UI. Alternatively, you can set it as an environment variable:
    ```bash
    export OPENAI_API_KEY="your_api_key_here"
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your browser.

## License

This project is licensed under the MIT License.