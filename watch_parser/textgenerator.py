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
    conversation_history: list = None

    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []

    def __call__(self, prompt, verbose=True):
        self.conversation_history.append(prompt)

        full_prompt = "\n".join(self.conversation_history)

        # Your existing logic to generate the response
        agent = create_csv_agent(
            ChatOpenAI(temperature=0, model="gpt-4-1106-preview"),
            self.dataset,
            verbose=verbose,
            agent_type=AgentType.OPENAI_FUNCTIONS
        )

        tool_input = {
            "input": {
                "name": "python",
                "arguments": full_prompt,
            }
        }

        out = agent.run(tool_input)

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
