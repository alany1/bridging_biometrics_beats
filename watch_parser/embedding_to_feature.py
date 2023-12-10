import os
import glob
import pickle


# we need to pass in our our dataset, textgenerator to get text, embedding to get embedding, and then run each model on the embedding
def generate_params(embedding, popularity=None, verbose=False):
    """
    Takes vector space of dimension 384 and outputs a prediction for each of 12 audio parameters
    using 12 separate pre-trained XGBoostRegressor models.
    Returns a dictionary of predicted parameters, including desired genre and playlist length.
    """
    parameters = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                  'key', 'liveness', 'target_loudness', 'mode', 'speechiness',
                  'tempo', 'signature', 'valence']

    # Create dictionary to be added to (with popularity if argument has been passed)
    if popularity:
        spotify_features = {'target_popularity': popularity}
    else:
        spotify_features = {}

    # Find the XGB files
    xgboost_files = os.path.join("../model_xgb_384/*_model_xgb_384")
    xgboost_models = glob.glob(xgboost_files)

    # Use each XGB model to predict on corresponding audio parameter
    for parameter, model in zip(parameters, xgboost_models):
        with open(model, 'rb') as f:
            xgb_model = pickle.load(f)
        preds = xgb_model.predict(embedding.reshape(1, -1))
        spotify_features[parameter] = preds[0]

    if verbose:
        print(spotify_features)
    return spotify_features
