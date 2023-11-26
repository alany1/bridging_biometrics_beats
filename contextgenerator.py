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

class Args(PrefixProto):
    dataset = "example_data/swim.csv"

print(os.environ["OPENAI_API_KEY"])

@proto_partial(Args)
def entrypoint(*, dataset):
    agent = create_csv_agent(
        ChatOpenAI(temperature=0, model="gpt-4-1106-preview"),
        dataset,
        verbose=True
    )
    df = pd.read_csv(dataset)
    print(df)

    tool_input = {"input": 
        {"name": "python", 
        "arguments": "This was a swimming workout. What genre of music would be good to listen to? Use all the data to come to a conclusion."
        }
    }

    out = agent.run(tool_input)
    print(out)
    print("i'm done")

if __name__ == '__main__':
    entrypoint()

