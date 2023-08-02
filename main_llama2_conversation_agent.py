"""Main entrypoint for the app."""
import logging
from pathlib import Path
import sys
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from schemas import ChatResponse
from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationTokenBufferMemory
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.vectorstores.base import VectorStore
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool
import pickle
from callback import AgentCallbackHandler,LLMAgentCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.chat_models import ChatOpenAI
from datetime import datetime
import llama2


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if getattr(sys, "frozen", False):
    script_location = Path(sys.executable).parent.resolve()
else:
    script_location = Path(__file__).parent.resolve()
load_dotenv(dotenv_path=script_location / ".env")

app = FastAPI()
templates = Jinja2Templates(directory="llama2-13b-chatbot-templates")

# @app.on_event("startup")
# async def startup_event():


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def get_agent(agent_cb_handler,websocket) -> AgentExecutor:
    llm_gpt4 = ChatOpenAI(
        model="gpt-4",
        temperature=0.6,
		callbacks=[LLMAgentCallbackHandler(websocket=websocket)],
        verbose=True,
    )
    agent_cb_manager = AsyncCallbackManager([agent_cb_handler])
    search = GoogleSerperAPIWrapper()
    mtia_tool = llama2.MarketTrendAndInvestmentAdviseToolChain.from_create(
        websocket=websocket,
        verbose=True
    )
    tools = [
        Tool(
            name="Cryptocurrency Research Report System",
            func=mtia_tool.run,
            description="You can use this tool when you need to generate the latest Cryptocurrency research report. The input should be a complete question about the request to generate the latest market research report of Cryptocurrency. This tool will generate the textual content of the market research report for you.",
            coroutine=mtia_tool.arun,
        ),
        Tool(
            name="Current Search",
            func=search.run,
            description="""
            useful for when you need to answer questions about current events or the current state of the world or you need to ask with search. 
            the input to this should be a single search term.
            """,
            coroutine=search.arun,
        ),
    ]

    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory = ConversationTokenBufferMemory(
        llm=llm_gpt4,
        memory_key="chat_history",
        max_token_limit=3000,
        return_messages=True,
    )
    PREFIX = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
It is {system_time} now."""
    system_template = PromptTemplate(input_variables=["system_time"], template=PREFIX)
    agent_excutor = initialize_agent(
        tools=tools,
        llm=llm_gpt4,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        system_message=system_template.format(
            system_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ),
        verbose=True,
        memory=memory,
        callback_manager=agent_cb_manager,
        agent_kwargs={
            "verbose": True,
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
    agent = get_agent(agent_cb_handler=agent_cb_handler,websocket=websocket)
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
