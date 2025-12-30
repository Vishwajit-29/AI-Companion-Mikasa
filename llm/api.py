import os

from dotenv import load_dotenv
from openai import OpenAI


class NvidiaChatClient:
    """NVIDIA API chat client for text-based conversations"""

    def __init__(self):
        env_path = os.path.abspath("D:\PROJECTS\AI\Mikasa\.env")
        load_dotenv(dotenv_path=env_path, override=True)
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY environment variable is not set")

        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1", api_key=api_key
        )

    def chat(self, user_input):
        """Send a chat message and return the response"""
        try:
            completion = self.client.chat.completions.create(
                model="nvidia/nemotron-3-nano-30b-a3b",
                messages=[{"role": "user", "content": user_input}],
                temperature=1,
                top_p=1,
                max_tokens=16384,
                extra_body={
                    "reasoning_budget": 16384,
                    "chat_template_kwargs": {"enable_thinking": True},
                },
                stream=True,
            )

            response = ""
            for chunk in completion:
                reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
                if reasoning:
                    print(reasoning, end="")
                if chunk.choices[0].delta.content is not None:
                    response += chunk.choices[0].delta.content

            return response

        except Exception as e:
            return f"Error: {str(e)}"

    def chat_streaming(self, user_input):
        """Send a chat message and stream the response"""
        try:
            completion = self.client.chat.completions.create(
                model="nvidia/nemotron-3-nano-30b-a3b",
                messages=[{"role": "user", "content": user_input}],
                temperature=1,
                top_p=1,
                max_tokens=16384,
                extra_body={
                    "reasoning_budget": 16384,
                    "chat_template_kwargs": {"enable_thinking": True},
                },
                stream=True,
            )

            print("Assistant: ", end="", flush=True)
            for chunk in completion:
                """ Reasoning printing off!
                reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
                if reasoning:
                    print(reasoning, end="", flush=True)
                    """
                if chunk.choices[0].delta.content is not None:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print()  # New line after response

        except Exception as e:
            print(f"Error: {str(e)}")


# Test function
if __name__ == "__main__":
    client = NvidiaChatClient()
    print("NVIDIA Chat Client")
    print("-" * 20)

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            break
        client.chat_streaming(user_input)
