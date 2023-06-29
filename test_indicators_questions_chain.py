import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from chain.indicators_quetions_chain import IndicatorsQuestionsChain 



# if getattr(sys, 'frozen', False):
#     script_location = pathlib.Path(sys.executable).parent.resolve()
# else:
#     script_location = pathlib.Path(__file__).parent.resolve()
load_dotenv(dotenv_path= '.env')

chain=IndicatorsQuestionsChain.from_indicators(indicators="RSI, CCI",verbose=True)
input=input("Send:\n") 
print(f"Output:{chain(inputs=input)}")