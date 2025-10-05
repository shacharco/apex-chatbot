# LangGraph-style CLI Chatbot (Python)

## Getting Started

### 1. Clone the Repository

```bash
gh repo clone shacharco/apex-chatbot
cd apex-chatbot
```

### 2. Install pyenv

This project uses [pyenv](https://github.com/pyenv/pyenv) to manage Python versions.

```bash
brew install pyenv
```

After installation, add pyenv to your shell configuration:

```bash
# For zsh (default on macOS)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc

# For bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
source ~/.bash_profile
```

### 3. Install Python

Install the required Python version (specified in `.python-version`):

```bash
pyenv install
```

Verify the correct version is active:

```bash
python --version
```

### 4. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure API Key

Get a free API key from [Mistral AI](https://mistral.ai) and create a `.env` file:

```bash
echo "MISTRAL_API_KEY=your-mistral-api-key-here" > .env
```

### 7. Download and Extract Documentation

Download the documentation files from [Google Drive](https://drive.google.com/file/d/17-QhEen1juz9lXoKw0qb6Fffy6edFYhN/view) and extract them to the docs directory:

```bash
# After downloading harel_public_information.zip to the project root
unzip harel_public_information.zip
```

### 8. Preprocess Documents

Build indices from your documents:

```bash
python preprocess.py --input_dir harel_public_information
```

### 9. Run the Chatbot

```bash
python cli.py
```

Type your questions and type `exit` or `quit` to stop.

## Files

- `preprocess.py` - Preprocess documents into FAISS + TF-IDF indices per category
- `chatbot.py` - Graph-style chatbot implementation (Router → Retriever → Generator)
- `cli.py` - Simple CLI entrypoint
- `prompts/` - Prompts used by LLM steps (instruction, few-shot, formatting)
- `tests/` - Test runner that computes average precision/recall/F1 (word-overlap-based)
- `requirements.txt` - Python packages
- `.python-version` - Required Python version for pyenv

# Notes

- This is a lightweight, self-contained "langgraph-style" pipeline (no external langgraph dependency).
- You need to set MISTRAL_API_KEY environment variable for the generator step, or adapt LLM call in chatbot.py.
- Preprocessing will create a folder called `indices/` with files per category.
