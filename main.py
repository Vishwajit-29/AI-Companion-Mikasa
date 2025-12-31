#!/usr/bin/env python3
"""
Main entry point for the Mikasa AI project.
Handles sequential execution and modular code handling.
"""

import os
import sys

# Add the project root and tts directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "tts"))

import asyncio
import os
import sys
import threading
import time

from llm.api import NvidiaChatClient
from tts.tts_client import TTSClient


def text_chat_mode():
    """Handles text-based chat interaction with integrated TTS."""
    print("\n=== Text Chat Mode (TTS Enabled) ===")
    print("Type 'quit', 'exit', or 'bye' to return to the main menu.")
    print("While the AI is speaking, press ENTER to interrupt.")
    print("-" * 40)

    tts_client = None
    client = None
    exit_event = threading.Event()

    try:
        # Initialize clients
        tts_client = TTSClient()
        client = NvidiaChatClient(tts_client=tts_client)

        # --- Input and Interruption Handling ---
        def user_input_handler():
            while not exit_event.is_set():
                user_input = input("You: ")
                if user_input.lower() in ["quit", "exit", "bye"]:
                    exit_event.set()
                    break

                if tts_client and tts_client.audio_player.is_active():
                    print("\n[User interrupted. Stopping audio...]")
                    tts_client.interrupt()

                asyncio.run(client.chat_streaming(user_input))

        input_thread = threading.Thread(target=user_input_handler, daemon=True)
        input_thread.start()

        # Keep the main thread alive
        while not exit_event.is_set():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nReturning to the main menu...")
    except Exception as e:
        print(f"An error occurred in text chat mode: {e}")
    finally:
        # Signal the input thread to exit and clean up resources
        exit_event.set()
        if tts_client:
            tts_client.shutdown()
        print("Exited text chat mode.")


def voice_chat_mode():
    """Handle voice-based chat interaction (placeholder)"""
    print("\n=== Voice Chat Mode ===")
    print("Voice input is not implemented yet!")
    print("Returning to main menu...\n")


def main():
    """Main entry point"""
    print("Welcome to Mikasa AI Project")
    print("=" * 30)
    while True:
        print("\nPlease select an option:")
        print("1. Text Input Chat")
        print("2. Voice Input Chat")
        print("3. Exit")

        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == "1":
            text_chat_mode()
        elif choice == "2":
            voice_chat_mode()
        elif choice == "3":
            print("Bye")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
