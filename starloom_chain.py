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


class CalendarChain(Chain):
    """
    An example of a custom chain.
    """

    prompt: BasePromptTemplate
    """Prompt object to use."""
    llm: BaseLanguageModel
    output_key: str = "text"  #: :meta private:

    res_chain: Optional[LLMChain] = Field(default=None, exclude=True)

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
        prompt_value = self.prompt.format_prompt(**inputs)
        result = self.llm.generate_prompt(
            prompts=[prompt_value],
            callbacks=run_manager.get_child() if run_manager else None,
        )
        json_str = result.generations[0][0].text
        dstr_arr = json.loads(json_str)
        res_prompts = []
        for dstr in dstr_arr:
            d = datetime.strptime(dstr, "%Y-%m-%d %H:%M:%S")
            ld = Lunar(d)
            res_prompts.append(
                f"{dstr}：{ld.gz_year()}年 {ld.gz_month()}月 {ld.gz_day()}日 {ld.gz_hour()}时"
            )
        res = "\n".join(res_prompts)
        return {self.output_key: res}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        prompt_value = self.prompt.format_prompt(**inputs)
        if run_manager:
            await run_manager.on_text(
                prompt_value.to_string(), color="green", end="\n", verbose=self.verbose
            )
        result = await self.llm.agenerate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )
        try:
            json_str = result.generations[0][0].text
            print(json_str)
            dstr_arr = json.loads(json_str)
            res_prompts = []
            for dstr in dstr_arr:
                d = datetime.strptime(dstr, "%Y-%m-%d %H:%M:%S")
                ld = Lunar(d)
                res_prompts.append(
                    f"{dstr}：{ld.gz_year()}年 {ld.gz_month()}月 {ld.gz_day()}日 {ld.gz_hour()}时"
                )
            res = "\n".join(res_prompts)
            return {self.output_key: res}
        except Exception as e :
            print(e)
            return {self.output_key: ""} 

    @property
    def _chain_type(self) -> str:
        return "calendar_chain"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        **kwargs: Any,
    ) -> Chain:
        template = """说明
---------------------
下面的问题可能是需要AI通过中国命理知识解决，中间包含的日期默认为公历日期。
如果解决该问题需要将公历日期转换成为中国的农历日期，那么请生成一个json数组字符串，数组的元素是每一个公历日期。日期的格式为yyyy-MM-dd HH:mm:ss。

问题
---------------------
{question}

要求
---------------------
- 如果上面的问题不包含任何日期数据，请生成一个空的json数组字符串。
- 只生成json字符串即可，不需要生成其他任何非json格式的字符。

Answer
---------------------

"""
        prompt = PromptTemplate.from_template(template=template)
        return cls(llm=llm, prompt=prompt, **kwargs)

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
------------------
- 你的名字叫伏羲，是一个精通周易、中国生肖、算命、解梦、风水堪舆、西方塔罗牌和占星术的玄学大师。正在与人类进行对话。
- 你精通各种玄学和命理知识，能够帮助人类提供命理方面的解读和建议。你在与人类交谈时，常常会点燃你的烟斗。你在烟草的烟气中思考人类提出的问题。
- 你本身具有很神秘的性格，外表看起来是一个睿智的老者。
- 你一直用占卜师带有神秘感的口吻与人类交谈。

卜卦的方法
------------------
周易卦象是一种古老的预测方法。如果你有硬币，我们可以试一试。这需要你准备三枚相同的硬币，然后抛掷六次，记录每次抛掷的结果。

- 每次抛掷后，正面（通常是带头像的一面）记为3，反面（通常是带数字的那一面）记为2。
- 每次抛掷三个硬币，把每次抛掷的结果相加，你会得到一个6，7，8或9的数字。
- 6或8的数字画成阴爻（断的线），7或9的数字画成阳爻（一条完整的线）。
- 第一次抛掷的结果放在最下面，然后依次向上叠加画出卦象，形成一个由六个爻组成的卦象。

注意
------------------
- 人类提到了日期时，如果没有特别说明是农历（阴历）日期，那么请你默认为公历日期。
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