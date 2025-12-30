
# Mikasa AI Project

A modular AI assistant with support for both text and voice input.

## Features

- Text-based chat using NVIDIA's Nemotron-3 Nano 30B model
- Modular architecture for easy extension


## Prerequisites

- Python 3.7+
- NVIDIA API key (get it from [NVIDIA API Catalog](https://build.nvidia.com/))

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Mikasa
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your NVIDIA API key

## Usage

Run the main program:
```bash
python main.py
```

You'll be presented with a menu to choose between:
1. Text Input Chat
2. Voice Input Chat (not implemented yet)
3. Exit

### Text Chat Mode
- Type your message and press Enter
- The AI response will be displayed in real-time
- Type 'quit', 'exit', or 'bye' to return to the main menu

## Project Structure

```
Mikasa/
├── main.py              # Main entry point
├── requirements.txt     # Project dependencies
├── .env.example         # Environment variable template
├── llm/
│   ├── api.py           # NVIDIA API client
│   └── local.py         # Local LLM implementation (planned)
├── stt/                 # Speech-to-text module (planned)
├── rag/                 # Retrieval-augmented generation (planned)
├── avatar/              # Avatar/visual interface (planned)
├── behavior/            # Behavior control (planned)
└── contracts/           # Contracts/interfaces (planned)
```

## Modules

### LLM Module
- `api.py`: Implements the NVIDIA API client for cloud-based inference




