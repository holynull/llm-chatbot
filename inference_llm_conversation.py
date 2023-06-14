from ast import List
from langchain import  ConversationChain, SagemakerEndpoint
from langchain.memory import ConversationBufferMemory,ConversationTokenBufferMemory
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from typing import Any, Dict, Optional
import json
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.callbacks.base import LLMResult
from sqlalchemy import UUID
from langchain.schema import AgentAction,AgentFinish
from langchain.callbacks.base import BaseCallbackManager,BaseCallbackHandler
from langchain.prompts.prompt import PromptTemplate
import logging

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger("Falcon-40B-Conversation")

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

llm=SagemakerEndpoint(
        endpoint_name="huggingface-pytorch-tgi-inference-2023-06-08-09-34-28-551", 
        credentials_profile_name="default", 
        region_name="us-east-1", 
        model_kwargs={
            "do_sample":True,
            "top_p": 0.9,
            "top_k": 10,
            "repetition_penalty": 1.03,
            "max_new_tokens":1024,
            "temperature":1e-10,
            "max_length":1024,
            "num_return_sequences":10,
            # "stop": ["\nUser:","<|endoftext|>","</s>"],
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

prompt=""

last_input=""

while True:
    user_inputs=input("Prompt: ")
    print("="*70)
    if user_inputs=="" and prompt=="":
        break
    if user_inputs!="":
        last_input=user_inputs
        if prompt=="":
            prompt=PROMPT.format(input=user_inputs)
        else:
            prompt=prompt+"\n\nHuman: "+user_inputs+"\nAI:"
    logger.debug(f"Prompt to LLM: {prompt}")
    resp=llm.predict(prompt)
    logger.info(f"LLM Response: {resp}")
    prompt=resp
    sta=resp.rfind("Human: "+last_input)+len("Human: "+last_input)+1
    delta_str=resp[sta:]
    if delta_str.find("Human")!=-1:
        ai_res=delta_str[:delta_str.find("Human")]
    else:
        ai_res=delta_str
    answer=ai_res[3:]

    print(f"AI answer: {answer}")