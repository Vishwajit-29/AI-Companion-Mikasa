import asyncio
import os
import threading

from dotenv import load_dotenv
from openai import OpenAI

from tts.tts_client import TTSClient


class NvidiaChatClient:
    """NVIDIA API chat client integrated with TTS capabilities."""

    def __init__(self, tts_client: TTSClient = None):
        env_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", ".env")
        )
        load_dotenv(dotenv_path=env_path, override=True)
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY environment variable is not set")

        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1", api_key=api_key
        )
        self.tts_client = tts_client

    async def _llm_stream_generator(self, user_input):
        """An async generator that yields chunks from the LLM stream."""
        try:
            completion = self.client.chat.completions.create(
                model="nvidia/nemotron-3-nano-30b-a3b",
                messages=[{"role": "user", "content": user_input}],
                temperature=1,
                top_p=1,
                max_tokens=16384,
                stream=True,
            )
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"🚨 LLM Error: {e}")
            yield ""

    async def chat_streaming(self, user_input: str):
        """
        Sends a chat message, streams the response, and handles TTS output.
        """
        print("Assistant: ", end="", flush=True)

        async def text_iterator():
            async for chunk in self._llm_stream_generator(user_input):
                print(chunk, end="", flush=True)
                yield chunk

        # If TTS is enabled, run it concurrently with printing
        if self.tts_client and self.tts_client.is_initialized:
            await self.tts_client.speak(text_iterator())
        else:
            # If no TTS, just iterate through and print
            async for _ in text_iterator():
                pass

        print()  # New line after the full response


# ============================================
# TEST CODE
# ============================================
if __name__ == "__main__":

    async def main_test():
        print("NVIDIA Chat Client with TTS")
        print("-" * 30)
        print("Enter text to chat. Type 'quit', 'exit', or 'bye' to end.")
        print("While the AI is speaking, press ENTER to interrupt.")
        print("-" * 30)

        # Initialize TTS client
        tts_client = TTSClient()
        client = NvidiaChatClient(tts_client=tts_client)

        # --- Keyboard Interruption Handling ---
        interrupted = threading.Event()

        def listen_for_interrupt():
            while True:
                input()  # Wait for user to press Enter
                if tts_client.audio_player.is_active():
                    print("\n🚨 User interrupted. Stopping audio...")
                    tts_client.interrupt()
                    interrupted.set()
                else:
                    print("(Press Enter while audio is playing to interrupt)")

        interrupt_thread = threading.Thread(target=listen_for_interrupt, daemon=True)
        interrupt_thread.start()
        # ------------------------------------

        try:
            while True:
                interrupted.clear()
                user_input = await asyncio.to_thread(input, "You: ")

                if user_input.lower() in ["quit", "exit", "bye"]:
                    break

                if interrupted.is_set():
                    print("You interrupted your own message. Try again.")
                    continue

                await client.chat_streaming(user_input)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
        finally:
            if tts_client:
                tts_client.shutdown()
            print("Client shut down.")

    asyncio.run(main_test())
