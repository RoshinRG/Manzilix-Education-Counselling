"""
test_llm.py — Standalone test script for the Manzilix LLM module.

Run directly to chat with the model from the terminal:
    python test_llm.py

Requires NVIDIA_API_KEY to be set in your environment (see llm.py docstring).
"""

from llm import ManzilixLLM


def main():
    llm = ManzilixLLM()

    print("Manzilix LLM test — type 'exit' to quit.\n")

    history = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        print("Manzilix: ", end="", flush=True)
        full_response = ""
        for token in llm.stream_chat(user_input, history=history):
            print(token, end="", flush=True)
            full_response += token
        print("\n")

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
