from typing import Any, Dict, List, Optional
from langchain.prompts import PromptTemplate
from langchain.chains.base import Chain
from langchain import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema import LLMResult
from langchain.chains import LLMChain
from pydantic import Extra, Field
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory

import json
from datetime import datetime
from DateConvert import Lunar

class StarLoomChain(LLMChain):
    """
    An example of a custom chain.
    """

    prompt: BasePromptTemplate
    """Prompt object to use."""
    llm: BaseLanguageModel
    output_key: str = "text"  #: :meta private:

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

    def generate(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> LLMResult:
        prompts, stop = self.prep_prompts(input_list, run_manager=run_manager)
        return self.llm.generate_prompt(
            prompts=prompts,
            stop=stop,
            callbacks=run_manager.get_child() if run_manager else None,
        )

    async def agenerate(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> LLMResult:
        prompts, stop = await self.aprep_prompts(input_list, run_manager=run_manager)
        return await self.llm.agenerate_prompt(
            prompts=prompts,
            stop=stop,
            callbacks=run_manager.get_child() if run_manager else None,
        )

    @property
    def _chain_type(self) -> str:
        return "starloom_chain"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        **kwargs: Any,
    ) -> Chain:
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    """说明
-------------------
- 你是一个占星术的玄学大师。正在与人类进行对话。
- 你精通各种玄学和命理知识，能够帮助人类提供命理方面的解读和建议。
- 你本身具有很神秘的性格，外表看起来是一个睿智的老者。
- 你一直用占卜师带有神秘感的口吻与人类交谈。

"""
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{question}"),
            ]
        )
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        return cls(
            llm=llm,
            prompt=prompt,
            memory=memory,
            **kwargs,
        )