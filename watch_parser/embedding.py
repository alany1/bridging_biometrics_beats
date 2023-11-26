'''
Spotify Context Vector Example:
    "danceability": 0.173,
    "energy": 0.609,
    "key": 3,
    "loudness": -7.724,
    "mode": 1,
    "speechiness": 0.0482,
    "acousticness": 0.359,
    "instrumentalness": 0,
    "liveness": 0.1,
    "valence": 0.0988,
    "tempo": 197.819,
    "type": "audio_features",
    "id": "7IfLxGBrCCd6PQUFRZ1Ub4",
    "uri": "spotify:track:7IfLxGBrCCd6PQUFRZ1Ub4",
    "track_href": "https://api.spotify.com/v1/tracks/7IfLxGBrCCd6PQUFRZ1Ub4",
    "analysis_url": "https://api.spotify.com/v1/audio-analysis/7IfLxGBrCCd6PQUFRZ1Ub4",
    "duration_ms": 244307,
    "time_signature": 3
'''
from watch_parser.textgenerator import TextGenerator
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
import os
from getpass import getpass
from params_proto import PrefixProto
from params_proto.partial import proto_partial
import pandas as pd



dataset = "example_data/swim.csv"


def embed_text(args):
    """
    Uses HuggingFace sentence-transformers/all-MiniLM-L6-v2 model to map sentences & paragraphs 
    to a 384 dimensional dense vector space for use as input to playlist generation model. 
    Returns vector space.
    """
    # Open saved pickle file 
    with open('MiniLMTransformer.pkl', 'rb') as f:
        embedder = pickle.load(f)

    # Embed input text
    input_text = args.text
    input_to_model = embedder.encode(input_text)
    return input_to_model

# textgenerator = TextGenerator(dataset)
output = "Considering the heart rate data from the swimming workout, which ranges from 95 to 99 beats per minute, a genre of music with a tempo in that range or slightly higher could be suitable. Genres like dance, electronic, or upbeat pop music often have songs within the 100-130 BPM range, which could be a good match for maintaining the energy level during the workout."
print(type(output))
print(output)




