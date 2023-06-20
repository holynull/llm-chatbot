import os
from dotenv import load_dotenv
from chain import cmc_quotes_chain 
from langchain.chat_models import ChatOpenAI
from langchain.chains import APIChain
import chain.all_templates
from langchain.prompts import PromptTemplate
from falcon_llm import FalconLLM
from falcon_llm import LLMContentHandler
from typing import Dict
import json
import logging


logging.basicConfig(level=logging.INFO)

# if getattr(sys, 'frozen', False):
#     script_location = pathlib.Path(sys.executable).parent.resolve()
# else:
#     script_location = pathlib.Path(__file__).parent.resolve()
load_dotenv(dotenv_path= '.env')
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
        endpoint_name="huggingface-pytorch-tgi-inference-2023-06-08-09-34-28-551", 
        credentials_profile_name="default", 
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
                # "stop": ["\nHuman:"],
                }
            },
		content_handler=content_handler,
		verbose=True,
)
headers = {
  'Accepts': 'application/json',
  'X-CMC_PRO_API_KEY': os.getenv("CMC_API_KEY"),
}
# cmc_currency_map_api=APIChain.from_llm_and_api_docs(llm=llm,api_docs=all_templates.cmc_currency_map_api_doc,headers=headers,verbose=True)
# cmc_quotes_api=APIChain.from_llm_and_api_docs(llm=llm,api_docs=all_templates.cmc_quote_lastest_api_doc,headers=headers,verbose=True)
chain = cmc_quotes_chain.CMCQuotesChain.from_llm(llm=llm,headers=headers,verbose=True)
input=input("Send:") 
print(f"Output:{chain(inputs=input)}")