# Website Chat Assistant

This project implements a conversational assistant that allows users to chat with the content of a specified website. It leverages vector embeddings, retrieval-based models, and a user-friendly Streamlit interface to provide context-aware and intelligent responses.

## Features

- **Website Content Understanding**:
  - Extracts and processes content from a user-specified website.
  - Creates a vector store of the content for efficient retrieval.

- **Context-Aware Conversation**:
  - Maintains conversation history for personalized and contextually relevant responses.
  - Uses a retriever to find information based on conversation flow.

- **Interactive Interface**:
  - Simple Streamlit-powered chat UI for ease of use.
  - Sidebar for setting website URL and starting new sessions.

## Workflow

1. **Input Website URL**:
   - User provides a website URL through the sidebar input.

2. **Content Processing**:
   - The webpage content is loaded, split into chunks, and indexed using a vector store.

3. **Conversation**:
   - Users type queries in the chat.
   - The assistant retrieves relevant information from the indexed content and provides responses.

4. **Conversation History**:
   - Maintains history to enable contextually relevant interactions.

## Technologies Used

- **LangChain**: For creating vector stores and retrieval-based chains.
- **Streamlit**: For building an interactive web-based chat application.
- **OpenAI**: For generating vector embeddings and conversational responses.
- **Chroma**: For vector store implementation.
- **dotenv**: For environment variable management.



![image](https://github.com/user-attachments/assets/c5bf6edc-6dd7-4b4f-843b-ad9db8c85e5c)


