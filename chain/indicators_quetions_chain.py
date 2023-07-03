from __future__ import annotations

from typing import Any, Dict, List, Optional
from langchain import LLMChain

from pydantic import Extra

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.chains import APIChain
from langchain.chains.api.base import API_RESPONSE_PROMPT
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import SequentialChain
from langchain.requests import TextRequestsWrapper

from chain import all_templates

prompt=PromptTemplate(template=all_templates.quotes_chain_template,input_variables=["user_input"])

class IndicatorsQuestionsChain(Chain):
    """
    An example of a custom chain.
    """

    prompt: BasePromptTemplate=prompt
    """Prompt object to use."""
    llm: BaseLanguageModel
    output_key: str = "text"  #: :meta private:

    # consider_chain:LLMChain
    
    # cmc_quotes_api:APIChain 

    seq_chain:LLMChain

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        prompt_value = self.prompt.format_prompt(**inputs)
        
        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        response = self.llm.generate_prompt(
            [prompt_value],
            callbacks=run_manager.get_child() if run_manager else None
        )

        # If you want to log something about this run, you can do so by calling
        # methods on the `run_manager`, as shown below. This will trigger any
        # callbacks that are registered for that event.
        if run_manager:
            run_manager.on_text(response.generations[0][0].text, color="green", end="\n", verbose=self.verbose)
        original_question=response.generations[0][0].text
        try:
            res= self.seq_chain.run(original_question) 
            if run_manager:
                run_manager.on_text(res, color="green", end="\n", verbose=self.verbose)
            return {self.output_key: res}
        except Exception as err:
            # answer=await self.answer_chain.arun(question=inputs['user_input'],context=err.args)
            return {self.output_key: err.args}
        # answer=self.answer_chain.run(question=inputs['user_input'],context=res)
        # if run_manager:
        #     run_manager.on_text(answer, color="yellow", end="\n", verbose=self.verbose) 
        # return {self.output_key: answer}
        

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        prompt_value = self.prompt.format_prompt(**inputs)
        
        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        response = await self.llm.agenerate_prompt(
            [prompt_value],
            callbacks=run_manager.get_child() if run_manager else None
        )

        # If you want to log something about this run, you can do so by calling
        # methods on the `run_manager`, as shown below. This will trigger any
        # callbacks that are registered for that event.
        if run_manager:
            await run_manager.on_text(response.generations[0][0].text, color="green", end="\n", verbose=self.verbose)
        original_question=response.generations[0][0].text
        try:
            res=await self.seq_chain.arun(original_question) 
            if run_manager:
                await run_manager.on_text(res, color="green", end="\n", verbose=self.verbose)
            return {self.output_key: res}
        except Exception as err:
            # answer=await self.answer_chain.arun(question=inputs['user_input'],context=err.args)
            return {self.output_key: err.args}
        # answer=await self.answer_chain.arun(question=inputs['user_input'],context=res)
        # if run_manager:
        #     await run_manager.on_text(answer, color="yellow", end="\n", verbose=self.verbose) 
        # return {self.output_key: answer}

    @property
    def _chain_type(self) -> str:
        return "indicators_questions_chain"
    
    @classmethod
    def from_indicators(cls,indicators:str,**kwargs: Any,)->IndicatorsQuestionsChain:
        PROMPT_TEMPLATE = """The user's question may be to ask the latest market trend of a certain cryptocurrency. 
        The following index tools can help users analyze market trend.
        Index tools:
        """+indicators
        PROMPT_TEMPLATE=PROMPT_TEMPLATE+"""\nPlease generats questions, to ask the latest index above of the cryptocurrency with its symbol in user's question.
        And you should generate the questions in JSON Array format as following:
        ["What is the btc's recent cci?","xxxxxxxx"]
        User's Question: {question}
        You generations:"""

        prompt=PromptTemplate(input_variables=["question"],template=PROMPT_TEMPLATE)
        llm=ChatOpenAI(
            model_name="gpt-4",
            temperature=0.9,
            request_timeout=60,
            **kwargs
        )
        chain=LLMChain(llm=llm,prompt=prompt,**kwargs)
        return cls(llm=llm,seq_chain=chain,**kwargs)