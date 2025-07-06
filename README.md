# Mock Data Generator (MDG) 4.0

MDG 4.0 is a sophisticated, AI-powered tool for generating high-quality, realistic mock data. It leverages a local, instruction-tuned LLM to understand input JSON structures and produce new data that is contextually relevant and structurally identical. This full-stack application features a React-based frontend for a seamless user experience and a powerful Python backend built with FastAPI.

## Key Features

- **AI-Powered Data Generation**: Uses a local, fine-tuned LLM to generate data that matches the schema and style of your input examples.
- **High-Performance Backend**: Built with FastAPI and `llama-cpp-python` for efficient, high-throughput data generation.
- **Advanced Caching**: Implements a sophisticated partial-caching mechanism with Redis to instantly serve previously generated data, reducing costs and latency.
- **Multiple Inference Modes**: Supports different generation architectures (`local`, `sequential`, `runpod`) to balance performance and stability.
- **Dynamic Instruction & Parameter Control**: Utilizes varied instructional prompts and adjustable temperature settings to prevent repetitive, predictable output.
- **Automatic Schema Validation**: Ensures every generated data object strictly conforms to the structure of the input examples.
- **Intelligent Image Enrichment**: Automatically identifies fields containing image keywords and enriches them with relevant, real image URLs from Openverse.
- **Modern Frontend**: A sleek and intuitive UI built with React, Vite, and Shadcn UI for easy interaction.

## Project Structure

- **`backend/`**: The FastAPI application that handles API requests, data generation, caching, and all core logic.
- **`frontend/`**: The React/Vite frontend application that provides the user interface.
- **`model/`**: Contains the GGUF model file used for local inference.
- **`model_server/`**: A separate FastAPI server for model-specific tasks like moderation (can be run alongside the main backend).
- **`start-dev.sh`**: A convenience script to launch the full stack environment.

## Prerequisites

- Python 3.9+
- Node.js and npm (or Bun)
- Redis (for caching)
- MongoDB (for logging)

## Backend Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <repository-name>
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**
    Create a `.env` file in the project root directory. This is required to specify the path to your GGUF model.

    ```env
    # .env

    # --- Core Settings ---
    # REQUIRED: Absolute or relative path to your GGUF model file.
    GGUF_MODEL_PATH="model/phi35-finetuned-Q4_K_M.gguf"

    # --- Inference Mode ---
    # Defines the generation architecture. Options:
    # 'local': High-performance batching with a pool of model instances. (Default)
    # 'sequential': Error-free single-threaded mode for debugging.
    # 'runpod': For connecting to a serverless RunPod endpoint.
    LLM_INFERENCE_MODE="local"

    # --- Services ---
    # Ensure these services are running and accessible.
    REDIS_URL="redis://localhost:6379"
    MONGO_URI="mongodb://localhost:27017/"
    MODEL_SERVER_URL="http://localhost:8001"

    # --- RunPod (Optional) ---
    # Required only if LLM_INFERENCE_MODE is set to 'runpod'.
    # RUNPOD_ENDPOINT_ID="your-runpod-endpoint-id"
    # RUNPOD_API_KEY="your-runpod-api-key"
    ```

5.  **Download the Model:**
    The required model is hosted on Hugging Face. Use the following command from your project's root directory to download it. The command will automatically create the `model` directory for you.

    ```bash
    curl -L "https://huggingface.co/VishwasB/phi-3.5-finetuned-gguf/resolve/main/phi35-finetuned-Q4_K_M.gguf" --create-dirs -o "model/phi35-finetuned-Q4_K_M.gguf"
    ```
    _Note: The model file is approximately 2.4 GB._

## Frontend Setup

1.  **Navigate to the Frontend Directory:**
    ```bash
    cd frontend
    ```

2.  **Install Dependencies:**
    ```bash
    npm install
    ```
    *(Alternatively, you can use `bun install`)*

## Running the Application

The easiest way to run the full stack application is using the provided script from the **project root directory**.

1.  **Make the script executable:**
    ```bash
    chmod +x start-dev.sh
    ```

2.  **Run the script:**
    ```bash
    ./start-dev.sh
    ```

This script will:
- Start the Model Server on `http://localhost:8001`.
- Start the Backend API server on `http://localhost:8000`.
- Start the Frontend development server on `http://localhost:3000`.

You can now access the application at **http://localhost:3000**.

---

## Instruction Fine-Tuning

The LLM used in this project is a fine-tuned version of `microsoft/Phi-3.5-mini-instruct`. The fine-tuning process adapts the base model to our specific synthetic data generation task, improving its ability to follow complex instructions and generate high-quality, structured JSON output.

The `Instruction_tuning_phase/` directory contains all the necessary scripts and data for this process.

### Tuning Process Overview

The fine-tuning is performed using QLoRA (Quantized Low-Rank Adaptation), an efficient technique that dramatically reduces memory usage. The process is orchestrated by the `Instruction_tuning_phase/phi-3.5-mini-Instruct/tuner_phi3.5.py` script, which leverages Hugging Face's `transformers`, `peft`, and `trl` libraries.

Key steps in the tuning script:
1.  **Data Loading**: The script loads the custom dataset from `Instruction_tuning_phase/dataset/dataset.json`.
2.  **Formatting**: Each data sample is formatted into the specific chat template required by the Phi-3 model.
3.  **Model Quantization**: The base model is loaded in 4-bit precision (`bnb_4bit_quant_type="nf4"`) to reduce its memory footprint.
4.  **LoRA Configuration**: PEFT (Parameter-Efficient Fine-Tuning) is configured to inject trainable LoRA adapters into the model.
5.  **Training**: The `SFTTrainer` (Supervised Fine-Tuning Trainer) runs the training process, saving the resulting adapters.

## Here is the complete Documentation of my project
https://copper-pepper-543.notion.site/Documentation-2112fc3aeb488083bc9fdfcec504c4e5?source=copy_link