
# shows acoustic features for tracks for the given artist

from __future__ import print_function    # (at top of module)
from spotipy.oauth2 import SpotifyClientCredentials
import json
import spotipy
import time
import sys


client_credentials_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
sp.trace = False

if len(sys.argv) > 1:
    artist_name = ' '.join(sys.argv[1:])
else:
    artist_name = 'weezer'

results = sp.search(q=artist_name, limit=50)
tids = []
for i, t in enumerate(results['tracks']['items']):
    print(' ', i, t['name'])
    tids.append(t['uri'])

start = time.time()
features = sp.audio_features(tids)
delta = time.time() - start
for feature in features:
    print(json.dumps(feature, indent=4))
    # print()
    # analysis = sp._get(feature['analysis_url'])
    # print(json.dumps(analysis, indent=4))
    # print()
print("features retrieved in %.2f seconds" % (delta,))


def generate_params(model_input, args):
    """
    Takes vector space of dimension 384 and outputs a prediction for each of 12 audio parameters
    using 12 separate pre-trained XGBoostRegressor models.
    Returns a dictionary of predicted parameters, including desired genre and playlist length.
    """
    # Parameters and genre

    parameters = ['target_acousticness', 'danceability', 'target_energy', 'target_instrumentalness',
                  'key', 'target_liveness', 'target_loudness', 'mode', 'target_speechiness',
                  'target_tempo', 'time_signature', 'target_valence']

    # Create dictionary to be added to (with popularity if argument has been passed)
    if args.popularity:
        input_to_spotify_transformer = {'target_popularity': args.popularity}
    else:
        input_to_spotify_transformer = {}

    # Find the XGB files
    xgboost_files = os.path.join("model_xgb_384/*_model_xgb_384")
    xgboost_models = glob.glob(xgboost_files)

    # Use each XGB model to predict on corresponding audio parameter
    for parameter, model in zip(parameters, xgboost_models):
        with open(model, 'rb') as f:
            xgb_model = pickle.load(f)
        preds = xgb_model.predict(model_input.reshape(1, -1))
        input_to_spotify_transformer[parameter] = preds[0]

    if args.verbose > 0:
        print(input_to_spotify_transformer)
    return input_to_spotify_transformer
