"""
GOALS: watch -> LM (few shot or feature generator) -> "watch target" t_w

TODO:
extra inputs (e.g. stroke count, distance)

"""
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent, create_pandas_dataframe_agent
import os
from getpass import getpass
from params_proto import PrefixProto
from params_proto.partial import proto_partial
import pandas as pd
from dataclasses import dataclass
from typing import Literal


class ModelArgs(PrefixProto):
    model_name: Literal["gpt-4-1106-preview"] = "gpt-4-1106-preview"


@dataclass
class TextGenerator:
    dataset: str
    model_name: str = "gpt-4-1106-preview"
    conversation_history: list = None

    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []

        self.df = pd.read_csv(self.dataset)
        self.agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0, model=self.model_name), self.df,
                                                   prefix="Remove any ` from the Action Input",
                                                   # agent_type=AgentType.OPENAI_FUNCTIONS,
                                                   agent_executor_kwargs={"handle_parsing_errors": True},
                                                   verbose=True)
        # self.agent = create_csv_agent(ChatOpenAI(temperature=0, model=self.model_name), self.dataset, verbose=True)

    def __call__(self, prompt, verbose=True):
        self.conversation_history.append(prompt)

        full_prompt = "\n".join(self.conversation_history)

        tool_input = {
            "input": {
                "name": "python",
                "arguments": full_prompt,
            }
        }

        out = self.agent.run(tool_input)

        # Append the response to the conversation history
        self.conversation_history.append(out)

        return out

    def reset_conversation(self):
        # Clear the conversation history
        self.conversation_history = []


if __name__ == '__main__':
    prompt = "This was a swimming workout. What genre of music would be good to listen to? Use all the data to come to a conclusion. Summarize it as a playlist description, and only return the description."
    gen = TextGenerator("../example_data/swim.csv")
    lm_output = gen(prompt)
    print(lm_output)
