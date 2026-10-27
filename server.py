import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import agent as core

app = FastAPI(title="LangGraph Chatbot")
INDEX = Path(__file__).parent / "index.html"
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class MessageIn(BaseModel):
    text: str


class DecisionIn(BaseModel):
    approved: bool
    args: dict | None = None          # {tool_call_id: {edited arguments}}


def sse(events):
    for ev in events:
        yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"


@app.get("/")
def home():
    return FileResponse(INDEX)


# ----- chats -----
@app.get("/api/chats")
def chats():
    return core.list_threads()


@app.post("/api/chats")
def new_chat():
    return {"id": core.new_thread_id()}


@app.get("/api/chats/{thread_id}")
def chat(thread_id: str):
    return {"messages": core.get_history(thread_id), "pending": core.get_pending(thread_id)}


@app.delete("/api/chats/{thread_id}")
def remove_chat(thread_id: str):
    core.delete_thread(thread_id)
    return {"ok": True}


@app.post("/api/chats/{thread_id}/message")
def message(thread_id: str, body: MessageIn):
    if not body.text.strip():
        raise HTTPException(400, "Empty message")
    if core.get_pending(thread_id):
        raise HTTPException(409, "Approve or reject the pending action first")
    return StreamingResponse(
        sse(core.stream_turn(thread_id, user_text=body.text)),
        media_type="text/event-stream", headers=SSE_HEADERS,
    )


@app.post("/api/chats/{thread_id}/decision")
def decision(thread_id: str, body: DecisionIn):
    if not core.get_pending(thread_id):
        raise HTTPException(409, "Nothing is waiting for approval")
    resume = {"approved": body.approved, "args": body.args or {}}
    return StreamingResponse(
        sse(core.stream_turn(thread_id, resume=resume)),
        media_type="text/event-stream", headers=SSE_HEADERS,
    )



# ----- documents (RAG) -----
@app.get("/api/docs")
def docs():
    return {"files": core.get_kb().files}


@app.post("/api/docs")
def upload_docs(files: list[UploadFile] = File(...)):
    kb = core.get_kb()
    results = []

    for f in files:
        try:
            ok, msg = kb.add(
                f.filename or "uploaded_file.txt",
                f.file.read()
            )
            results.append({"ok": ok, "message": msg})
        finally:
            f.file.close()

    return {"results": results, "files": kb.files}


@app.delete("/api/docs")
def clear_docs():
    """Clear the listing/index without deleting saved documents."""
    success, message = core.get_kb().clear_index()

    if not success:
        raise HTTPException(status_code=500, detail=message)

    return {"success": True, "message": message}


@app.delete("/api/docs/permanent")
def permanently_delete_docs():
    """Permanently delete all saved documents and chunks."""
    success, message = core.get_kb().delete_all_documents()

    if not success:
        raise HTTPException(status_code=500, detail=message)

    return {"success": True, "message": message}




if __name__ == "__main__":
    import uvicorn
    core.get_agent()                              # load everything before the first request
    uvicorn.run(app, host="127.0.0.1", port=8000)  # localhost only: this server has no login