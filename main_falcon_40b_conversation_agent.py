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
from langchain.memory import ConversationTokenBufferMemory
from typing import  Dict 
from langchain.agents import initialize_agent,AgentType,AgentExecutor
from langchain.vectorstores.base import VectorStore
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.chains import RetrievalQA
from langchain.agents import Tool
import pickle
from callback import AgentCallbackHandler 
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.chat_models import ChatOpenAI
from chain.cmc_quotes_chain import CMCQuotesChain
import os


logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

if getattr(sys, 'frozen', False):
    script_location = Path(sys.executable).parent.resolve()
else:
    script_location = Path(__file__).parent.resolve()
load_dotenv(dotenv_path=script_location / '.env')

app = FastAPI()
templates = Jinja2Templates(directory="templates")

with open("data-swft.pkl", "rb") as f:
        vcs_swft = pickle.load(f)
with open("data-path.pkl", "rb") as f:
        vcs_path = pickle.load(f)

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
        return generated_text

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
                "max_new_tokens":500,
                "temperature":0.8,
                "return_full_text":False,
                # "max_length":1512,
                # "num_return_sequences":10,
                # "stop": ["\nHuman:"],
                }
            },
        content_handler=content_handler,
        verbose=True,
    )

def get_agent(
    vcs_swft: VectorStore,
    vcs_path: VectorStore, 
    agent_cb_handler) -> AgentExecutor:
    llm_agent = ChatOpenAI(
        temperature=0.9, 
        # model="gpt-4",
        verbose=True,
    )
    agent_cb_manager = AsyncCallbackManager([agent_cb_handler])
    search = GoogleSerperAPIWrapper()
    combine_prompt_template = """Given the following extracted parts of a long document and a question, create a final answer. 
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    QUESTION: {question}
    ANSWER:{summaries}
    FINAL ANSWER:"""
    COMBINE_PROMPT = PromptTemplate(
        template=combine_prompt_template, input_variables=["summaries", "question"]
    )
    doc_search_swft = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="map_reduce", 
        retriever=vcs_swft.as_retriever(search_kwargs={"k":8}),
        chain_type_kwargs={
          "combine_prompt":COMBINE_PROMPT,
          "verbose":True,
        },
        verbose=True
        )
    doc_search_path = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="map_reduce", 
        chain_type_kwargs={
            "combine_prompt":COMBINE_PROMPT,
            "verbose":True,
         },
        retriever=vcs_path.as_retriever(search_kwargs={"k":8}),
        verbose=True
        )
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': os.getenv("CMC_API_KEY"),
    }
    cmc_quotes_api=CMCQuotesChain.from_llm(llm=llm_agent,headers=headers,verbose=True)
    tools = [
        Tool(
            name = "Cryptocurrency Latest Quotes, Exchange rate and Price System",
            func=cmc_quotes_api.run,
            description="When you need to inquire about the latest cryptocurrency market trends or the latest cryptocurrency prices, you can use this tool. The input should be a complete question, and use the original language.",
            coroutine=cmc_quotes_api.arun
        ),
        Tool(
            name = "QA SWFT System",
            func=doc_search_swft.run,
            description="useful for when you need to answer questions about swft. Input should be a fully formed question, and use the original language.",
            coroutine=doc_search_swft.arun
        ),
         Tool(
            name = "QA Metapath System",
            func=doc_search_path.run,
            description="useful for when you need to answer questions about metapath. Input should be a fully formed question, and use the original language.",
            coroutine=doc_search_path.arun
        ),
        Tool(
            name = "Current Search",
            func=search.run,
            description="""
            useful for when you need to answer questions about current events or the current state of the world or you need to ask with search. 
            the input to this should be a single search term.
            """,
            coroutine=search.arun
        ),
    ]
    
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory = ConversationTokenBufferMemory(llm=llm_agent,memory_key="chat_history",max_token_limit=3000,return_messages=True)
    agent_excutor = initialize_agent(
        tools=tools,
        llm=llm_agent, 
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True, 
        memory=memory,
        callback_manager=agent_cb_manager,
        agent_kwargs={
             "verbose":True,
        },
    )
    return agent_excutor

@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    agent_cb_handler = AgentCallbackHandler(websocket)
    await websocket.accept()
    chat_history = []
    # Use the below line instead of the above line to enable tracing
    # Ensure `langchain-server` is running
    agent=get_agent(vcs_swft=vcs_swft,vcs_path=vcs_path,agent_cb_handler=agent_cb_handler)
    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            result = await agent.arun(input=question)
            print(f"Result: {result}")
            # resp = ChatResponse(sender="bot", message=result, type="stream")
            # await websocket.send_json(resp.dict())
            # chat_history.append((question, result))

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logger.info("websocket disconnect")
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
