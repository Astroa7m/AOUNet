import json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import time

# ANSI colors
BLUE = "\033[94m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def pretty_print_messages(messages):
    print(f"\n{BOLD}{MAGENTA}{'=' * 80}")
    print("📜  CONVERSATION HISTORY")
    print(f"{'=' * 80}{RESET}\n")

    for i, message in enumerate(messages, 1):
        # Determine role
        if isinstance(message, HumanMessage):
            role = f"{BOLD}{BLUE}👤 Human"
        elif isinstance(message, AIMessage):
            role = f"{BOLD}{GREEN}🤖 Assistant"
        elif isinstance(message, SystemMessage):
            role = f"{BOLD}{YELLOW}⚙️ System"
        else:
            role = f"{BOLD}{CYAN}🧩 Other"

        print(f"{role} {RESET}(message {i})")
        print(f"{'-' * 60}")

        # Optional reasoning
        reasoning = None
        if hasattr(message, "metadata") and "reasoning" in message.metadata:
            reasoning = message.metadata["reasoning"]
        elif hasattr(message, "additional_kwargs") and "reasoning_content" in message.additional_kwargs:
            reasoning = message.additional_kwargs["reasoning_content"]

        if reasoning:
            print(f"{DIM}{YELLOW}💭 Reasoning:{RESET}\n{reasoning}\n")

        # Content
        content = getattr(message, "content", str(message))
        if isinstance(content, (dict, list)):
            content = json.dumps(content, indent=2, ensure_ascii=False)

        print(f"{CYAN}🗣️ Content:{RESET}\n{content}\n")
        print(f"{DIM}{'=' * 80}{RESET}\n")
