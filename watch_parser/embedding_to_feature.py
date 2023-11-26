
def generate_params(model_input, args):
    """
    Takes vector space of dimension 384 and outputs a prediction for each of 12 audio parameters
    using 12 separate pre-trained XGBoostRegressor models.
    Returns a dictionary of predicted parameters, including desired genre and playlist length.
    """
    # Parameters and genre

    parameters = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                  'key', 'liveness', 'target_loudness', 'mode', 'speechiness',
                  'tempo', 'signature', 'valence']

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
