"""Main entrypoint for the app."""
import logging
from pathlib import Path
import sys
from fastapi import  FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain.callbacks.base import AsyncCallbackHandler
from schemas import ChatResponse
from dotenv import load_dotenv
from typing import Any
from starloom_chain import StarLoomChain


logging.basicConfig(level=logging.INFO)

app = FastAPI()

if getattr(sys, "frozen", False):
    script_location = Path(sys.executable).parent.resolve()
else:
    script_location = Path(__file__).parent.resolve()
load_dotenv(dotenv_path=script_location / ".env")

templates = Jinja2Templates(directory="starloom-templates")


# @app.on_event("startup")
# async def startup_event():


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class StreamingLLMCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(self, websocket):
        self.websocket = websocket

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        resp = ChatResponse(sender="bot", message=token, type="stream")
        await self.websocket.send_json(resp.dict())

from langchain.chat_models import ChatOpenAI



@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    chat_history = []
    # Use the below line instead of the above line to enable tracing
    # Ensure `langchain-server` is running
    # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)
    from langchain.callbacks.manager import AsyncCallbackManager
    gpt4 = ChatOpenAI(
        model_name="gpt-4", 
        temperature=0.9, 
        streaming=True,
        callback_manager=AsyncCallbackManager([StreamingLLMCallbackHandler(websocket=websocket)]),
        verbose=True
    )
    conversationChain = StarLoomChain.from_llm(
        llm=gpt4,
        verbose=True,
    )

    last_input = ""
    prompt = ""
    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())
            await conversationChain.arun(question)

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=9000)
