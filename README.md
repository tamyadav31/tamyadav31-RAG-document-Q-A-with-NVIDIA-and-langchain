# RAG Document Q&A with NVIDIA NIM and LangChain

## Overview

This project demonstrates a **retrieval-augmented generation (RAG)** workflow for document-based question answering, leveraging the powerful combination of NVIDIA’s NIM (NVIDIA Inference Microservices) language models, advanced embeddings, and the LangChain framework. Users can load a directory of PDF documents, create an efficient vector database using NVIDIA embeddings and FAISS, and ask complex questions—getting highly relevant, context-aware answers via an intuitive Streamlit interface.

---

## Features

- **NVIDIA NIM Language Models & Embeddings:** Utilizes state-of-the-art LLMs (`meta/llama3-70b-instruct`) and embedding models from NVIDIA for powerful, modern natural language understanding.
- **PDF Document Loader:** Seamlessly loads and ingests multiple PDF files for analysis.
- **Intelligent Chunking:** Documents are automatically split into overlapping, context-preserving chunks, ensuring accurate retrieval even from large files.
- **FAISS Vector Store:** Embeds and indexes documents using FAISS, enabling fast, accurate similarity search and retrieval.
- **RAG Pipeline:** Advanced retrieval-augmented generation ensures responses are always grounded in your uploaded documents.
- **Interactive Web App:** Streamlit frontend for workflow control, Q&A, and live display of supporting document snippets.
- **Expander View:** Review which document segments informed each answer for transparency and auditing.

---

## Technology Stack

- **Python 3.x**
- [NVIDIA NIM LLMs](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-foundation/models/meta_llama3_70b_instruct)
- [LangChain](https://github.com/langchain-ai/langchain) core and community modules
- **FAISS** (vector similarity search)
- **Streamlit** (user interface)
- **PyPDF** (PDF document loading)
- **python-dotenv** (for secure API credential management)

---

## Installation

1. **Clone the repository**
git clone https://github.com/tamyadav31/tamyadav31-RAG-document-Q-A-with-NVIDIA-and-langchain.git
cd tamyadav31-RAG-document-Q-A-with-NVIDIA-and-langchain

text

2. **Create and activate a Python virtual environment (optional but recommended)**
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

text

3. **Install all dependencies**
pip install -r requirements.txt

text

4. **Configure NVIDIA API key**
- Register or obtain credentials for the NVIDIA NIM API service.
- Create a `.env` file in the project root with:
  ```
  NVIDIA_API_KEY=your_nvidia_api_key_here
  ```

5. **Prepare your document directory**
- Place all PDF files to be used for Q&A inside a folder named `us_census` in the project root (`./us_census`).

---

## Usage

1. **Start the Streamlit app**
streamlit run finalapp.py

text

2. **Workflow**
- Click the **document embedding** button to load, split, embed, and index your documents.
- Enter your question in the input field and receive a context-aware answer, grounded in the loaded documents.
- Use the "document similarity search" expander to see which text segments informed the response.

---

## File Structure

| File             | Purpose                                                                          |
|------------------|----------------------------------------------------------------------------------|
| `finalapp.py`    | Main Streamlit web application—RAG workflow, user interface, doc indexing, QA    |
| `requirements.txt` | Project dependencies                                                           |
| `README.md`      | (You’re reading it!)                                                             |

---

## Acknowledgements

- [NVIDIA AI Foundation](https://www.nvidia.com/en-us/ai-data-science/ai-foundation-models/)
- [LangChain](https://langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)

---

## License

This project is open source and available under the MIT License.

---

_Build retrieval-augmented, document-grounded AI apps with the best from NVIDIA and LangChain!_
