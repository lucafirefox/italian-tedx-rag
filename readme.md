# 🎙️ TEDx Italian Speeches RAG Engine

This repository contains the code for a Retrieval-Augmented Generation (RAG) engine that answers questions based on the content of TEDx Italian speeches. It leverages a vector database (ChromaDB), OpenAI's embedding and language models, and the LlamaIndex library.

## 🗂️ Project Structure

The repository is organized into the following directories:

-   **`data_loader`**: Contains the code to load and process transcripts, generate embeddings, and store them in the vector database.
    -   **`src`**: Includes Python modules for data handling, embedding, and vector database operations.
    -   **`tests`**: Contains tests for the data loader's core functionalities.
    -   **`Dockerfile`**: Defines the docker image build process for the data loader.
    -   **`entrypoint.sh`**: Defines the entrypoint for the data loader docker container.
    -   **`pyproject.toml`**: Project configuration file for Poetry.
    -    **`poetry.lock`**: lock file for poetry dependencies
-   **`query_engine`**: Contains the code for querying the vector database and generating responses using an LLM.
    -   **`src`**: Includes Python modules for querying the vector database and interacting with the LLM.
    -   **`tests`**: Contains tests for the query engine's core functionalities.
    -   **`Dockerfile`**: Defines the docker image build process for the query engine.
    -   **`entrypoint.sh`**: Defines the entrypoint for the query engine docker container.
    -   **`pyproject.toml`**: Project configuration file for Poetry.
    -    **`poetry.lock`**: lock file for poetry dependencies
-   **`transcripts`**: A directory that contains the transcript text files to be loaded by the data loader. (This directory should contain `.txt` files).
-   **`docker-compose.yml`**: Docker Compose configuration file for deploying the entire application.
-   **`ruff.toml`**: Configuration file for Ruff, a Python linter.
-   **`.env`**: Environment variables (note that this file is not pushed to git repo for security reasons).

## ✨ Key Technologies

This project utilizes the following technologies:

-   **🐍 Python 3.12**: The backbone of our codebase.
-   **📦 Poetry**: For dependency management and packaging.
-   **🐳 Docker**: For consistent and containerized deployments.
-   **🚢 Docker Compose**: For managing multi-container applications.
-   **🗄️ ChromaDB**: A robust vector database for storing and retrieving embeddings.
-   **🧠 OpenAI API**: Empowering the project with top-tier embedding and language models (including `gpt-4o`).
-   **🦙 LlamaIndex**: A framework for building RAG applications.
-   **🖼️ Gradio**: A Python library to build interactive web app.

## 🚀 Setup and Installation

### Prerequisites

1.  **Docker and Docker Compose**: Ensure you have Docker and Docker Compose installed on your machine. You can download them from [Docker's website](https://docs.docker.com/get-docker/).
2.  **Poetry**: Ensure you have Poetry installed on your machine. You can download it from [Poetry's website](https://python-poetry.org/docs/#installation).
3.  **OpenAI API Key**: You need a valid OpenAI API key. Get it from the [OpenAI website](https://platform.openai.com/api-keys).

### Installation Steps

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create a `.env` file in the root directory:**
    - create a copy of .env file example and rename the copy to `.env`
     - copy the following and update with your values:

      ```bash
      OPENAI_API_KEY=<your_openai_api_key>
      COLLECTION_NAME=TEDx_italian_speeches
      EMBEDDING_MODEL=text-embedding-3-large
      LLM_MODEL=gpt-4o-mini-2024-07-18
      CHUNK_SIZE=512
      CHUNK_OVERLAP=40
      ```
    -   Replace `<your_openai_api_key>` with your actual OpenAI API key.


3.  **Build and run the application using Docker Compose:**

    ```bash
    docker-compose up --build
    ```

    This command will:
    - Build the Docker images for the `data_loader` and `query_engine`.
    - Create the necessary networks and volumes.
    - Start the containers.
    - First, the `data_loader` container will process all transcripts and store them in the vector database.
    - After successful execution, the `query_engine` container starts and exposes the application on `http://localhost:8000`.

4.  **Access the Query Engine**: Once the query_engine container is running, you can access the interactive web interface in your browser by navigating to: `http://localhost:8000`.

## 📚 Usage

1.  **Data Loading:**
    -   The `data_loader` service automatically starts first, processes the transcripts in the `/transcripts` directory, creates embeddings and adds them to the vector store.
    -   This process is executed only once. If you need to update the vector db you will need to restart the `data_loader` container and let it finish the execution (or remove the volume in order to force reload of the database).

2.  **Querying:**
    -   Once the `data_loader` has finished, you can access the query engine through the specified port `http://localhost:8000`.
    -   Enter your question about the TEDx talks in the interface.
    -   The engine will use RAG to generate an answer.

## 🎬 Demo

[![Demo Video](https://img.youtube.com/vi/up3DNARO-Bk/0.jpg)](https://www.youtube.com/watch?v=up3DNARO-Bk)

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 👨‍💻 Author

-   [Luca Ferrario](https://github.com/lucafirefox)
