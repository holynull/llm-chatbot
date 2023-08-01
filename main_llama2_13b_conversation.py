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
from falcon_llm import FalconLLM,LLMContentHandler
from langchain.memory import ConversationBufferMemory,ConversationTokenBufferMemory
from langchain.callbacks.base import BaseCallbackManager,BaseCallbackHandler
from langchain.callbacks.base import LLMResult
from typing import Any, Dict, Optional
from sqlalchemy import UUID
from langchain.schema import AgentAction,AgentFinish


logging.basicConfig(level=logging.INFO)

if getattr(sys, 'frozen', False):
    script_location = Path(sys.executable).parent.resolve()
else:
    script_location = Path(__file__).parent.resolve()
load_dotenv(dotenv_path=script_location / '.env')

app = FastAPI()
templates = Jinja2Templates(directory="llama2-13b-chatbot-templates")


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
        generated_text=response_json[0]["generated_text"]
        logging.info(f"Generated Text:\n{generated_text}")
        sta=generated_text.rfind("AI:")+3;
        end=generated_text.rfind("\nHuman:");
        logging.info(f"Start at: {sta}, end at {end}")
        if end==-1 or end<=sta:
            resp= generated_text[sta:]
        else:
            resp= generated_text[sta:end]
        logging.info(f"Response str:{resp}")
        return resp

content_handler = ContentHandler()

llm=FalconLLM(
        endpoint_name="huggingface-pytorch-tgi-inference-2023-07-24-07-23-15-93", 
        # credentials_profile_name="default1", 
        region_name="us-east-1", 
        model_kwargs={
            "parameters":{
                "do_sample":True,
                # "top_p": 0.9,
                # "top_k": 10,
                "repetition_penalty": 1.03,
                "max_new_tokens":1024,
                "temperature":0.8,
                # "max_length":1024,
                # "num_return_sequences":10,
                "stop": ["\nHuman:"],
                }
            },
        content_handler=content_handler,
    )



class CallbackHandler(BaseCallbackHandler):
    """Callback handler for question generation."""

    def __init__(self, websocket):
        self.websocket = websocket

    # def on_llm_start(
    #     self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    # ) -> None:
    #     """Run when LLM starts running."""
    #     print(f"ON_LLM_START")
        # resp = ChatResponse(
        #     sender="bot", message="Synthesizing question...", type="info"
        # )
        # self.websocket.send_json(resp.dict())

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM ends running.""" 
        print(f"ON_LLM_END: {response.dict}")
    # def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
    #     print(f"llm new token: {token}")
    #     resp = ChatResponse(sender="bot", message=token, type="stream")
    #     self.websocket.send_json(resp.dict())

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        print(f"ON_CHAIN_START: Inputs: {inputs}")

    async def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        print(f"ON_CHAIN_END: Outputs: {outputs}")
        resp = ChatResponse(sender="bot", message=outputs['response'], type="stream")
        await self.websocket.send_json(resp.dict())
        # resp = ChatResponse(sender="bot", message=outputs['output'], type="stream")
        # self.websocket.send_json(resp.dict())
        # if outputs['answer'] != None:
        #     resp = ChatResponse(
        #         sender="bot", message=outputs['answer'], type="stream")
        # else:
        #     resp = ChatResponse(
        #         sender="bot", message="Synthesizing question...", type="info"
        #     )
        # self.websocket.send_json(resp.dict())
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool starts running."""
        print(f"ON_TOOL_START: input: {input_str}")

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool ends running."""
        print(f"ON_TOOL_END: output: {output}")
    
    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run on agent end."""
        print(f"ON_AGENT_FINISH: {finish.return_values}")
        # resp = ChatResponse(sender="bot", message=finish.return_values['output'], type="stream")
        # await self.websocket.send_json(resp.dict())
    
    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run on agent action."""
        print(f"ON_AGENT_ACTION: tool: {action.tool}")

DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to 
a question, it truthfully says it does not know.

Current conversation:
Human: Hi. Nice to meet you.
AI: Hi, Nice to meet you, too.
{history}
Human: {input}
AI:"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=DEFAULT_TEMPLATE)

memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=200)

@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    chat_history = []
    # Use the below line instead of the above line to enable tracing
    # Ensure `langchain-server` is running
    # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)
    conversationChain=ConversationChain(llm=llm,memory=memory,prompt=PROMPT,verbose=True,callback_manager=BaseCallbackManager([CallbackHandler(websocket=websocket)]))

    last_input=""
    prompt=""
    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
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
