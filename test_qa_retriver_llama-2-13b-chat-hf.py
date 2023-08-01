from dotenv import load_dotenv
from falcon_llm import FalconLLM
from falcon_llm import LLMContentHandler
from typing import Dict, Optional
import json
from langchain.chains import RetrievalQA
import os
import pickle
from langchain.vectorstores import VectorStore
import logging
from langchain.prompts.prompt import PromptTemplate

logging.basicConfig(level=logging.INFO)


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

llm=FalconLLM(
        endpoint_name="huggingface-pytorch-tgi-inference-2023-07-24-02-28-45-647", 
        credentials_profile_name="default", 
        region_name="us-east-1", 
        model_kwargs={
            "parameters":{
                "do_sample":True,
                # "top_p": 0.9,
                # "top_k": 10,
                "repetition_penalty": 1.03,
                "max_new_tokens":200,
                "temperature":0.8,
				"return_full_text":False,
                # "max_length":1024,
                # "num_return_sequences":10,
                # "stop": ["\nHuman:"],
                }
            },
		content_handler=content_handler,
		verbose=True,
)
# combine_prompt_template = """Given the following extracted parts of a long document and a question, create a final answer. 
# If you don't know the answer, just say that you don't know. Don't try to make up an answer.
# QUESTION: {question}
# ANSWER:{summaries}
# FINAL ANSWER:"""
# COMBINE_PROMPT = PromptTemplate(
#     template=combine_prompt_template, input_variables=["summaries", "question"]
# )
doc_search_swft = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_reduce", 
	chain_type_kwargs={
        #   "combine_prompt":COMBINE_PROMPT,
		  "verbose":True,
	},
    # retriever=vectorstore.as_retriever(search_type="mmr",search_kwargs={"k":1}),
    retriever=vectorstore.as_retriever(),
    verbose=True,
)
user_inputs=input("Prompt:")
print(doc_search_swft(user_inputs))