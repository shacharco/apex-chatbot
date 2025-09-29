
LangGraph-style CLI Chatbot (Python)


Files:
- preprocess.py : preprocess documents into FAISS + TF-IDF indices per category
- chatbot.py : graph-style chatbot implementation (Router -> Retriever -> Generator)
- cli.py : simple CLI entrypoint
- prompt-version-1.txt: prompts used by LLM steps (instruction, few-shot, formatting)
- validation-data.txt : sample validation pairs (q:..., a:...)
- test_main.py : test runner that computes average precision/recall/F1 (word-overlap-based)
- requirements.txt : python packages

run:
1. pip install -r ./requirements
2. create .env file with MISTRAL_API_KEY=your-mistral-api-key (its free to generate one in their website)
3. load the .env to you env
4. run python preprocess.py
5. run python cli.py

Notes:
- This is a lightweight, self-contained "langgraph-style" pipeline (no external langgraph dependency).
- You need to set MISTRAL_API_KEY environment variable for the generator step, or adapt LLM call in chatbot.py.
- Preprocessing will create a folder called `indices/` with files per category.
