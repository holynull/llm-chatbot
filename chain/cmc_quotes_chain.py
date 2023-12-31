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
from chain.indicators_quetions_chain import IndicatorsQuestionsChain
from chain.taapi_cci_chain import TaapiCCIChain
from chain.taapi_rsi_chain import TaapiRSIChain
from chain.taapi_stochrsi_chain import TaapiSTOCHRSIChain
from chain.taapi_dmi_chain import TaapiDMIChain
from chain.taapi_macd_chain import TaapiMACDChain
from chain.taapi_psar_chain import TaapiPSARChain
from chain.taapi_cmf_chain import TaapiCMFChain
import os
import json

from chain import all_templates

prompt=PromptTemplate(template=all_templates.quotes_chain_template,input_variables=["user_input"])

class CMCQuotesChain(Chain):
    """
    An example of a custom chain.
    """

    prompt: BasePromptTemplate=prompt
    """Prompt object to use."""
    llm: BaseLanguageModel
    output_key: str = "text"  #: :meta private:

    # consider_chain:LLMChain
    
    # cmc_quotes_api:APIChain 

    seq_chain:SequentialChain
    indicator_questions_chain:IndicatorsQuestionsChain
    cciChain:TaapiCCIChain
    rsiChain:TaapiRSIChain
    stochrsiChain:TaapiSTOCHRSIChain
    dmiChain:TaapiDMIChain
    macdChain:TaapiMACDChain
    psarChain:TaapiPSARChain
    cmfChain:TaapiCMFChain
    summaryChain:LLMChain

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
            quote= self.seq_chain.run(original_question=original_question) 
            index_questions_str=self.indicator_questions_chain.run(original_question)
            index_questions=json.loads(index_questions_str)
            rsi_str=self.rsiChain.run(index_questions["rsi"])
            cci_str=self.cciChain.run(index_questions["cci"])
            dmi_str=self.dmiChain.arun(index_questions["dmi"])
            # macd_str=self.macdChain.arun(index_questions[3])
            psar_str=self.psarChain.arun(index_questions["psar"])
            stochrsi_str=self.stochrsiChain.arun(index_questions["stochrsi"])
            cmf_str=self.cmfChain.arun(index_questions["cmf"])
            res=self.summaryChain.run(
                data0=quote,
                data1=rsi_str,
                data2=cci_str,
                data3=dmi_str,
                data4=psar_str,
                data5=stochrsi_str,
                data6=cmf_str,
				)
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
            quote=await self.seq_chain.arun(original_question=original_question) 
            index_questions_str=await self.indicator_questions_chain.arun(original_question)
            index_questions=json.loads(index_questions_str)
            rsi_str=await self.rsiChain.arun(index_questions["rsi"])
            cci_str=await self.cciChain.arun(index_questions["cci"])
            dmi_str=await self.dmiChain.arun(index_questions["dmi"])
            # macd_str=await self.macdChain.arun(index_questions[3])
            psar_str=await self.psarChain.arun(index_questions["psar"])
            stochrsi_str=await self.stochrsiChain.arun(index_questions["stochrsi"])
            cmf_str=await self.cmfChain.arun(index_questions["cmf"])
            res=await self.summaryChain.arun(
                data0=quote,
                data1=rsi_str,
                data2=cci_str,
                data3=dmi_str,
                data4=psar_str,
                data5=stochrsi_str,
                data6=cmf_str,
				)
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
        return "cmc_quotes_chain"
    
    @classmethod
    def from_llm(cls,llm:BaseLanguageModel,headers:dict,**kwargs: Any,)->CMCQuotesChain:
        API_URL_PROMPT_TEMPLATE = """You are given the below API Documentation:
        {api_docs}
        Using this documentation, generate the full API url to call for answering the user question.
        You should build the API url in order to get a response that is as short as possible. Pay attention to deliberately exclude any unnecessary pieces of data in the API call.
        You should not build API url with the word "aux".
        Question:{question}
        API url:"""

        API_URL_PROMPT = PromptTemplate(
            input_variables=[
                "api_docs",
                "question",
            ],
            template=API_URL_PROMPT_TEMPLATE,
        )
        api_req_llm=ChatOpenAI(
            # model_name="gpt-4",
            temperature=0,
            request_timeout=60,
            **kwargs
        )
        api_res_llm=ChatOpenAI(
            model_name="gpt-4",
            temperature=0.9,
            request_timeout=60,
            **kwargs
        )
        # api=APIChain.from_llm_and_api_docs(llm=api_llm,api_docs=all_templates.cmc_quote_lastest_api_doc,api_url_prompt=API_URL_PROMPT,headers=headers,**kwargs)
        api=APIChain(
            api_request_chain=LLMChain(llm=api_req_llm,prompt=API_URL_PROMPT,**kwargs),
            api_answer_chain=LLMChain(llm=api_res_llm,prompt=API_RESPONSE_PROMPT,**kwargs),
            api_docs=all_templates.cmc_quote_lastest_api_doc,
            requests_wrapper = TextRequestsWrapper(headers=headers),
            **kwargs,
            )
        # api=APIChain.from_llm_and_api_docs(llm=llm,api_docs=all_templates.cmc_quote_lastest_api_doc,headers=headers,**kwargs)
        product_prompt=PromptTemplate(
            input_variables=["original_question"],
            template=all_templates.consider_what_is_the_product
        )
        product_llm=ChatOpenAI(
            # model_name="gpt-4",
            temperature=0,
            request_timeout=60,
            **kwargs
        )
        product_chain=LLMChain(llm=product_llm,prompt=product_prompt,output_key="product",**kwargs)
        question_template=PromptTemplate(
            input_variables=["product"],
            template=all_templates.api_question_template,
        )
        question_chain=LLMChain(llm=product_llm,prompt=question_template,output_key="question",**kwargs)
        seq_chain=SequentialChain(chains=[product_chain,question_chain,api],input_variables=["original_question"],**kwargs)
        indicator_question_chain=IndicatorsQuestionsChain.from_indicators(indicators="RSI,CCI,DMI,PSAR,STOCHRSI,CMF",**kwargs)
        rsi_chain=TaapiRSIChain.from_llm(llm=api_res_llm,taapi_secret=os.getenv("TAAPI_KEY"),**kwargs)
        cci_chain=TaapiCCIChain.from_llm(llm=api_res_llm,taapi_secret=os.getenv("TAAPI_KEY"),**kwargs)
        dmi_chain=TaapiDMIChain.from_llm(llm=api_res_llm,taapi_secret=os.getenv("TAAPI_KEY"),**kwargs)
        macd_chain=TaapiMACDChain.from_llm(llm=api_res_llm,taapi_secret=os.getenv("TAAPI_KEY"),**kwargs)
        psar_chain=TaapiPSARChain.from_llm(llm=api_res_llm,taapi_secret=os.getenv("TAAPI_KEY"),**kwargs)
        stochrsi_chain=TaapiSTOCHRSIChain.from_llm(llm=api_res_llm,taapi_secret=os.getenv("TAAPI_KEY"),**kwargs)
        cmf_chain=TaapiCMFChain.from_llm(llm=api_res_llm,taapi_secret=os.getenv("TAAPI_KEY"),**kwargs)
        summary_template="""Quotes: {data0}
        {data1}
        
        {data2}

        {data3}

		{data4}

		{data5}

		{data6}

		The above are the latest quote of some Cryptocurrency and some index tool data. Please rewite the quote, provide the analysis results by analyzing the above index data, and provide the market trend."""
        # The above are the latest market trends of some Cryptocurrency and some index tool data. Please summarize the market trend and provide investment advice on this Cryptocurrency. I can use your advice to discuss with my financial advisor."""
        summary_prompt=PromptTemplate(input_variables=["data0","data1","data2","data3","data4","data5","data6"],template=summary_template)
        summaryChain=LLMChain(llm=api_res_llm,prompt=summary_prompt,**kwargs)
        return cls(llm=llm,
                   seq_chain=seq_chain,
                   indicator_questions_chain=indicator_question_chain,
                   rsiChain=rsi_chain,
                   cciChain=cci_chain,
                   summaryChain=summaryChain,
                   dmiChain=dmi_chain,
				   macdChain=macd_chain,
				   psarChain=psar_chain,
				   stochrsiChain=stochrsi_chain,
				   cmfChain=cmf_chain,
                   **kwargs)