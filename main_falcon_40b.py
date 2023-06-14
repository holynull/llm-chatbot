"""Main entrypoint for the app."""
import logging
from pathlib import Path
import sys
from typing import Dict
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from schemas import ChatResponse
from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
import json
from falcon_llm import FalconLLM,LLMContentHandler

logging.basicConfig(level=logging.INFO)

if getattr(sys, 'frozen', False):
    script_location = Path(sys.executable).parent.resolve()
else:
    script_location = Path(__file__).parent.resolve()
load_dotenv(dotenv_path=script_location / '.env')

app = FastAPI()
templates = Jinja2Templates(directory="templates")


# @app.on_event("startup")
# async def startup_event():


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"inputs": prompt, **model_kwargs})
        return input_str.encode('utf-8')
    
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        ai_conversation=response_json[0]["generated_text"]
        return ai_conversation

content_handler = ContentHandler()

llm=FalconLLM(
        endpoint_name="huggingface-pytorch-tgi-inference-2023-06-08-09-34-28-551", 
        # credentials_profile_name="default1", 
        region_name="us-east-1", 
        model_kwargs={
			"parameters":{
				"do_sample":True,
            	# "top_p": 0.9,
            	# "top_k": 10,
            	"repetition_penalty": 1.03,
            	"max_new_tokens":512,
            	"temperature":0.8,
            	# "max_length":1024,
            	# "num_return_sequences":10,
            	"stop": ["\n\n","<|endoftext|>","</s>"],
				}
            },
        content_handler=content_handler,
    )

DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
System: You are helpful assistant for human.
AI: Yes.

Human: Hi, Nice to meet you.
AI: Nice to meet you too.

Human: {input}
AI:"""
PROMPT = PromptTemplate(input_variables=["input"], template=DEFAULT_TEMPLATE)

@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    chat_history = []
    # Use the below line instead of the above line to enable tracing
    # Ensure `langchain-server` is running
    # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)

    last_input=""
    prompt=""
    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            # resp = ChatResponse(sender="you", message=question, type="stream")
            # await websocket.send_json(resp.dict())

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
            resp=llm.predict(question)
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
            resp = ChatResponse(sender="bot", message=resp, type="stream")
            await websocket.send_json(resp.dict())
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
