"""Main entrypoint for the app."""
import logging
from pathlib import Path
import sys
from typing import Dict
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain import ConversationChain
from schemas import ChatResponse
from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
import json
from langchain.memory import ConversationBufferMemory, ConversationTokenBufferMemory
from langchain.callbacks.base import BaseCallbackManager, BaseCallbackHandler
from langchain.callbacks.base import LLMResult
from typing import Any, Dict, Optional
from sqlalchemy import UUID
from langchain.schema import AgentAction, AgentFinish
from llama2_70b_chat import SagemakerLLMChat, LLama270bChatChain
import asyncio


logging.basicConfig(level=logging.INFO)

if getattr(sys, "frozen", False):
    script_location = Path(sys.executable).parent.resolve()
else:
    script_location = Path(__file__).parent.resolve()
load_dotenv(dotenv_path=script_location / ".env")

app = FastAPI()
templates = Jinja2Templates(directory="llama2-70bc-templates")


# @app.on_event("startup")
# async def startup_event():


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


llm = SagemakerLLMChat(
    endpoint_name="meta-textgeneration-llama-2-70b-f-2023-08-03-07-02-32-301",
    # credentials_profile_name="default",
    region_name="us-east-1",
    model_kwargs={
        "parameters": {
            "top_p": 0.9,
            "max_new_tokens": 2048,
            "temperature": 0.9,
        },
    },
    verbose=True,
)

PREFIX_PRESETTING_CHARACTER = "!character:"


class CallbackHandler(BaseCallbackHandler):
    """Callback handler for question generation."""

    def __init__(self, websocket):
        self.websocket = websocket

    async def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        print(f"###########{outputs}")
        resp = ChatResponse(sender="bot", message=outputs["text"], type="stream")
        await self.websocket.send_json(resp.dict())


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    chat_history = []
    # Use the below line instead of the above line to enable tracing
    # Ensure `langchain-server` is running
    # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)
    conversationChain = LLama270bChatChain.from_system_message(
        callback_manager=BaseCallbackManager([CallbackHandler(websocket=websocket)]),
        verbose=True,
    )

    last_input = ""
    prompt = ""
    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            if question.startswith(PREFIX_PRESETTING_CHARACTER):
                sta = question.find(PREFIX_PRESETTING_CHARACTER) + len(
                    PREFIX_PRESETTING_CHARACTER
                )
                system_message_txt = question[sta:]
                from langchain.prompts import SystemMessagePromptTemplate

                conversationChain = LLama270bChatChain.from_system_message(
                    message=SystemMessagePromptTemplate.from_template(
                        system_message_txt
                    ),
                    callback_manager=BaseCallbackManager(
                        [CallbackHandler(websocket=websocket)]
                    ),
                    verbose=True,
                )
                resp = ChatResponse(
                    sender="bot", message="Changing my character...", type="info"
                )
                await websocket.send_json(resp.dict())
                await asyncio.sleep(3)
                end_resp = ChatResponse(sender="bot", message="", type="end")
                await websocket.send_json(end_resp.dict())
            else:
                resp = ChatResponse(sender="you", message=question, type="stream")
                await websocket.send_json(resp.dict())

                # Construct a response
                start_resp = ChatResponse(sender="bot", message="", type="start")
                await websocket.send_json(start_resp.dict())
                # if user_inputs=="" and prompt=="":
                #     break
                # if question!="":
                #     last_input=question
                # if prompt=="":
                #     prompt=PROMPT.format(input=question)
                # else:
                #     prompt=prompt+"\n\nHuman: "+question+"\nAI:"
                await conversationChain.apredict(input=question)
                # prompt=resp
                # logging.info(f"Response fromm LLM:\n{resp}")
                # sta=resp.rfind("Human: "+last_input)+len("Human: "+last_input)+1
                # delta_str=resp[sta:]
                # if delta_str.find("Human")!=-1:
                #     ai_res=delta_str[:delta_str.find("Human")]
                # else:
                #     ai_res=delta_str
                # result=ai_res[3:]
                # print(f"Result: {result}")
                # resp = ChatResponse(sender="bot", message=resp, type="stream")
                # await websocket.send_json(resp.dict())
                # chat_history.append((question, result))

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
