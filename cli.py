import sys
from chatbot import ChatbotGraph


def main():
    bot = ChatbotGraph(model="gpt-4.1")  # initialize once
    print("Insurance Chatbot (type 'exit' to quit)")
    while True:
        try:
            user_input = input("You: ")
        except EOFError:
            break
        if user_input.strip().lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        result = bot.respond(user_input)
        generated = result.get("generated")
        answer = generated['answer']
        sources = generated['sources']
        if sources:
            answer = f"{answer}\nBased on: {sources}"
        print(f"Bot: {answer}")


if __name__ == "__main__":
    main()
