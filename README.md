# Info Agent Chatbot - HR Policy Documents (RAG Framework)

This repository contains the implementation of an intelligent chatbot that allows users to interact with HR policy documents using the Retrieval-Augmented Generation (RAG) framework. The chatbot enables querying HR policies, retrieving relevant information, and generating contextually accurate responses.

## Features

- **HR Policy Integration**: The chatbot is integrated with HR policy documents to provide accurate responses.
- **RAG Framework**: Utilizes the Retrieval-Augmented Generation (RAG) framework to retrieve the most relevant sections from policy documents based on user queries.
- **Real-Time Updates**: Seamless integration with the HR system for real-time updates to policy documents.
- **User-Friendly Interface**: Chatbot designed to be easy for HR personnel and employees to interact with and retrieve policy information.

## Installation

### Prerequisites

- Python 3.x
- Required libraries (listed in `requirements.txt`)

### Steps to Setup

1. **Clone the repository**:
    ```bash
    git clone <repo_url>
    cd <repo_name>
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up environment variables** (if needed):
    - Ensure that any API keys or service credentials are set up correctly for integration with your HR policy document database.

4. **Run the chatbot**:
    ```bash
    python chatbot.py
    ```

## Usage

Once the chatbot is running, you can interact with it through the provided interface to ask questions related to HR policies. The chatbot will use the RAG framework to search and generate relevant responses from the HR policy documents.

### Example Queries:
- "What is the company's leave policy?"
- "How can I request parental leave?"
- "Tell me about the dress code policy."

## Contributing

Contributions are welcome! If you have suggestions, improvements, or bug fixes, feel free to submit a pull request.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

