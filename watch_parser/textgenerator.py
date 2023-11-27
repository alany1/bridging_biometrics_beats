"""
GOALS: watch -> LM (few shot or feature generator) -> "watch target" t_w

TODO:
extra inputs (e.g. stroke count, distance)

"""
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
import os
from getpass import getpass
from params_proto import PrefixProto
from params_proto.partial import proto_partial
import pandas as pd
from dataclasses import dataclass


@dataclass
class TextGenerator:
    dataset: str

    def __call__(self, prompt):
        agent = create_csv_agent(
            ChatOpenAI(temperature=0, model="gpt-4-1106-preview"),
            self.dataset,
            verbose=True
        )
        df = pd.read_csv(self.dataset)
        print(df)

        tool_input = {"input":
                          {"name": "python",
                           "arguments": prompt,
                           }
                      }

        out = agent.run(tool_input)
        return out


if __name__ == '__main__':
    prompt="This was a swimming workout. What genre of music would be good to listen to? Use all the data to come to a conclusion. Summarize it as a playlist description."
    gen = TextGenerator("../example_data/swim.csv")
    gen(prompt)
