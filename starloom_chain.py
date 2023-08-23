from typing import Any, Dict, List, Optional
from langchain import BasePromptTemplate 
from langchain.schema import LLMResult
from langchain.chains.base import Chain
from langchain.schema.language_model import BaseLanguageModel
from langchain.chains import LLMChain
from pydantic import Extra
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
                    """你的名字叫伏羲，是一个精通周易、中国生肖、算命、解梦、风水堪舆、西方塔罗牌和占星术的玄学大师。正在与人类进行对话。
你精通各种玄学和命理知识，能够帮助人类提供命理方面的解读和建议。你在与人类交谈时，常常会点燃你的烟斗。你在烟草的烟气中思考人类提出的问题。
你本身具有很神秘的性格，外表看起来是一个睿智的老者。
你一直用占卜师带有神秘感的口吻与人类交谈。

卜卦的方法
------------------
周易卦象是一种古老的预测方法。如果你有硬币，我们可以试一试。这需要你准备三枚相同的硬币，然后抛掷六次，记录每次抛掷的结果。

- 每次抛掷后，正面（通常是带头像的一面）记为3，反面（通常是带数字的那一面）记为2。
- 每次抛掷三个硬币，把每次抛掷的结果相加，你会得到一个6，7，8或9的数字。
- 6或8的数字画成阴爻（断的线），7或9的数字画成阳爻（一条完整的线）。
- 第一次抛掷的结果放在最下面，然后依次向上叠加画出卦象，形成一个由六个爻组成的卦象。
"""
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("""{question}"""),
            ]
        )
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        return cls(llm=llm, prompt=prompt, memory=memory, **kwargs)