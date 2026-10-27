import sys
from pathlib import Path

import agent as core

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HELP = """
Commands:
  /new               start a new chat
  /chats             list your chats
  /open N            open chat number N
  /delete            delete the current chat
  /docs              list uploaded documents (RAG)
  /upload PATH       add a PDF, TXT or MD file
  /cleardocs         remove all documents
  /graph             show the graph again
  /help              show this help
  /exit              quit
Anything else is sent to the chatbot.
"""


def show_graph(mermaid: bool = False):
    graph = core.get_agent()[0].get_graph()
    print("\n========== GRAPH ==========")
    try:
        graph.print_ascii()
    except Exception as e:
        print(f"(ASCII drawing needs: pip install grandalf)  [{e}]")
    if mermaid:
        print("\n--- Mermaid text (paste into https://mermaid.live) ---")
        print(graph.draw_mermaid())
    try:
        Path("graph.png").write_bytes(graph.draw_mermaid_png())
        print("Saved graph.png")
    except Exception as e:
        print(f"(graph.png skipped, it needs internet) [{e}]")
    print("===========================\n")


def show_history(thread_id: str, last: int = 6):
    items = core.get_history(thread_id)[-last:]
    for it in items:
        label = {"user": "You", "assistant": "AI", "tool": "  [tool]"}[it["role"]]
        print(f"{label}: {it['text']}")
    if items:
        print()


def ask_approval(calls: list[dict]) -> dict:
    print("\n🛑 APPROVAL NEEDED: nothing happens until you approve.")
    for c in calls:
        print(f"Action: {c['name']}")
        for k, v in c["args"].items():
            print(f"  {k}: {v}")
    ans = input("Approve? [y]es / [n]o / [e]dit: ").strip().lower()
    if ans.startswith("e"):
        edits = {}
        for c in calls:
            new = {}
            for k, v in c["args"].items():
                val = input(f"  {k} [{str(v)[:40]}] (Enter = keep): ")
                new[k] = val if val.strip() else v
            edits[c["id"]] = new
        return {"approved": True, "args": edits}
    if ans.startswith("y"):
        return {"approved": True}
    return {"approved": False}


def run_turn(thread_id: str, user_text: str | None = None, resume: dict | None = None):
    """Print one turn as it streams. Returns the approval request if the graph paused."""
    pending, started = None, False
    for ev in core.stream_turn(thread_id, user_text=user_text, resume=resume):
        kind = ev["type"]
        if kind == "token":
            if not started:
                print("AI: ", end="", flush=True)
                started = True
            print(ev["text"], end="", flush=True)
            continue
        if started:
            print()
            started = False
        if kind == "tool":
            print(f"  [tool] {ev['text']}")
        elif kind == "tool_result":
            print(f"  [result] {ev['text']}")
        elif kind == "approval":
            pending = ev
        elif kind == "error":
            print(f"  [error] {ev['text']}")
    if started:
        print()
    print()
    return pending


def main():
    print("Loading the agent (the first start can take a little while)...")
    core.get_agent()
    show_graph()
    print(HELP)

    chats = core.list_threads()
    current = chats[0]["id"] if chats else core.new_thread_id()
    if chats:
        print(f"Opened your latest chat: {chats[0]['title']}\n")
        show_history(current)

    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not text:
            continue

        if text.startswith("/"):
            cmd, _, arg = text.partition(" ")
            cmd, arg = cmd.lower(), arg.strip()
            if cmd in ("/exit", "/quit"):
                print("Bye!")
                break
            elif cmd == "/help":
                print(HELP)
            elif cmd == "/new":
                current = core.new_thread_id()
                print("New chat started.\n")
            elif cmd == "/chats":
                chats = core.list_threads()
                if not chats:
                    print("No saved chats yet.\n")
                for i, c in enumerate(chats, start=1):
                    print(f"  {i}. {'●' if c['id'] == current else ' '} {c['title']}")
                print()
            elif cmd == "/open":
                chats = core.list_threads()
                if arg.isdigit() and 1 <= int(arg) <= len(chats):
                    current = chats[int(arg) - 1]["id"]
                    print(f"Opened: {chats[int(arg) - 1]['title']}\n")
                    show_history(current)
                else:
                    print("Use /chats to see the numbers, then /open N.\n")
            elif cmd == "/delete":
                core.delete_thread(current)
                current = core.new_thread_id()
                print("Chat deleted. Started a new one.\n")
            elif cmd == "/docs":
                files = core.get_kb().files
                if not files:
                    print("No documents yet. Use /upload PATH.\n")
                for name, n in files.items():
                    print(f"  📄 {name} · {n} chunks")
                print()
            elif cmd == "/upload":
                path = Path(arg.strip('"').strip("'"))
                if not path.is_file():
                    print("File not found.\n")
                else:
                    print("Indexing... (the first time it downloads the embedding model)")
                    print(core.get_kb().add(path.name, path.read_bytes())[1], "\n")
            elif cmd == "/cleardocs":
                core.get_kb().clear()
                print("All documents removed.\n")
            elif cmd == "/graph":
                show_graph(mermaid=True)
            else:
                print("Unknown command. Type /help.\n")
            continue

        pending = run_turn(current, user_text=text)
        while pending:                                       # human-in-the-loop
            decision = ask_approval(pending["calls"])
            pending = run_turn(current, resume=decision)


if __name__ == "__main__":
    main()