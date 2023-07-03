import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from chain.taapi_dmi_chain import TaapiDMIChain 



# if getattr(sys, 'frozen', False):
#     script_location = pathlib.Path(sys.executable).parent.resolve()
# else:
#     script_location = pathlib.Path(__file__).parent.resolve()
load_dotenv(dotenv_path= '.env')

llm = ChatOpenAI(
	model="gpt-4",
    temperature=0.9,
    verbose=True,
    )
chain=TaapiDMIChain.from_llm(llm=llm,taapi_secret=os.getenv("TAAPI_KEY"),verbose=True)
input=input("Send:\n") 
print(f"Output:{chain(inputs=input)}")