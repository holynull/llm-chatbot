from dotenv import load_dotenv
from langchain import  SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from typing import Dict, Optional
import json
from langchain.chains import RetrievalQA
import os
import pickle
from langchain.vectorstores import VectorStore


# if getattr(sys, 'frozen', False):
#     script_location = pathlib.Path(sys.executable).parent.resolve()
# else:
#     script_location = pathlib.Path(__file__).parent.resolve()
load_dotenv(dotenv_path= '.env')

chain_type = os.getenv("CHAIN_TYPE")

vectorstore: Optional[VectorStore] = None
with open("data-swft.pkl", "rb") as f:
        vectorstore = pickle.load(f)

class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"inputs": prompt, **model_kwargs})
        return input_str.encode('utf-8')
    
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generated_text"]

content_handler = ContentHandler()

llm=SagemakerEndpoint(
        endpoint_name="huggingface-pytorch-tgi-inference-2023-06-08-09-34-28-551", 
        credentials_profile_name="default", 
        region_name="us-east-1", 
        model_kwargs={
            "do_sample":True,
            "top_p": 0.9,
            "repetition_penalty": 1.03,
            "max_new_tokens":1024,
            "temperature":0.8,
            # "stop": ["\nUser:","<|endoftext|>","</s>"],
            },
		content_handler=content_handler,
)
doc_search_swft = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_reduce", 
    retriever=vectorstore.as_retriever(search_type="mmr",search_kwargs={"k":1}),
    verbose=True,
)
user_inputs=input()
print(doc_search_swft(user_inputs))