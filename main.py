#!/usr/bin/env python3
"""
Main entry point for the Mikasa AI project.
Handles sequential execution and modular code handling.
"""

import os
import sys

from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from llm.api import NvidiaChatClient


def text_chat_mode():
    """Handle text-based chat interaction"""
    print("\n=== Text Chat Mode ===")
    print("Type 'quit', 'exit', or 'bye' to return to main menu\n")

    try:
        client = NvidiaChatClient()
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit", "bye"]:
                break
            client.chat_streaming(user_input)
    except KeyboardInterrupt:
        print("\nReturning to main menu...")
    except Exception as e:
        print(f"Error in text chat mode: {e}")


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
