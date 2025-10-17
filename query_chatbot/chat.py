import ast
import json
import time
import uuid
import argparse
import urllib.request
import urllib.error

from chatbot import ChatbotGraph


def query_chatbot(bot, question):
    """
    Sends a single question to the chatbot API (OpenAI Completions API compatible)
    and returns the assistant's reply and latency.
    """

    start_time = time.time()
    try:
        result = bot.respond(question)
        generated = result["generated"]
        answer = generated['answer']
        sources = generated['sources']
    except urllib.error.URLError as e:
        print(f"Error querying chatbot: {e}")
        return "Error", time.time() - start_time

    latency = time.time() - start_time
    return generated, latency


def main():
    parser = argparse.ArgumentParser(description="Query a local chatbot via REST API.")
    parser.add_argument("--input", required=True, help="Path to the questions TXT file")

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = f.read()
        questions = ast.literal_eval(data)
    bot = ChatbotGraph(model="mistral-medium")  # initialize once

    # Generate submission ID
    submission_id = str(uuid.uuid4()).split("-")[0]
    output_filename = f"{submission_id}_conversations.json"

    conversations = []

    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] Querying chatbot...")
        answer, bot_latency = query_chatbot(bot, q)

        conversation = {
            "turns": [
                {
                    "role": "user",
                    "content": q,
                    "latency": 0.0  # No latency for first user turn
                },
                {
                    "role": "assistant",
                    "content": answer,
                    "latency": bot_latency
                }
            ],
            "metadata": {}
        }
        conversations.append(conversation)
        print(f"[{i}/{len(questions)}] Done ({bot_latency:.2f}s)")

    # Save output
    with open(output_filename, "w", encoding="utf-8") as out:
        json.dump(conversations, out, ensure_ascii=False, indent=2)

    print(f"Saved {len(conversations)} conversations to {output_filename}")


if __name__ == "__main__":
    main()
