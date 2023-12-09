from watch_parser.alan_featurizer import Featurizer
from hf.feedback import FeedbackModule

class Runner:
    def __init__(self):
        self.featurizer = Featurizer("medium_term", verbose=True, recommendation_sample_count=3, num_genres=3,
                            use_cached_features=True)
        self.feedback = FeedbackModule(featurizer.gen)



if __name__ == '__main__':
    from killport import kill_ports

    kill_ports(ports=[9090])

    desc = "Concentrated Study Beats: Enhance your focus with this playlist featuring instrumental and ambient tracks. Ideal for deep concentration and productivity, these soothing, lyric-free melodies are perfect for any study session."
    # desc = "Energize Your Swim: A high-tempo mix of EDM, pop, and upbeat hits to keep your heart pumping and your strokes powerful. Perfect for moderate to high-intensity pool sessions."
    # desc = "Embrace the tranquility of the night with 'Midnight Stroll Serenade,' a playlist blending ambient sounds and soft, reflective melodies. Perfect for your nocturnal neighborhood walks, these tracks create a serene, contemplative atmosphere under the moonlit sky."
    # desc = "Step into the calm of the night with 'Lunar Whisper: Midnight Walks,' a playlist featuring soothing Chinese songs. Ideal for a peaceful midnight stroll, these melodic tunes blend traditional instruments with modern sensibilities, creating a serene and reflective ambiance."
    # desc = "'Morning Rise and Shine' - A vibrant playlist crafted for those crisp, early morning walks to the school bus stop. It's a blend of upbeat and energizing tracks to kick-start your day with positivity and motivation. Expect a mix of light-hearted pop, indie vibes, and inspiring tunes that mirror the freshness of a new day, perfect for a brisk walk under the morning sky."
    # desc = "'Volley Vibes' - This dynamic playlist is designed to pump you up for your volleyball game warm-ups. It's a high-energy mix of motivating beats and powerful anthems that will get your adrenaline flowing. Expect a blend of intense electronic, upbeat pop, and driving rock tunes that are perfect for getting into the competitive spirit and preparing your body and mind for the game. The rhythm and energy of these tracks will keep you focused and ready to dominate the court."
    featurizer = Featurizer("medium_term", verbose=True, recommendation_sample_count=3, num_genres=3,
                            use_cached_features=True)
    # featurizer.sample_genres(ALL_GENRES)
    features, genres = featurizer(desc)
    featurizer.generate_playlist(features, genres, "pre_feedback")
    
    feedback = FeedbackModule(featurizer.gen)    
    
    example_feedback = "that was a bit slow, I usually study with faster songs"
    
    featurizer.gen.reset_conversation()

    feedback_prompt = (
        "Having already generated an initial feature vector {} based on the playlist description '{}', "
        "you now need to adjust this vector in light of the user's feedback: {}. "
        "Crucially, use the dataframe provided earlier, which contains example features and their corresponding genres, as a key reference in this process. "
        "Examine the typical values and trends within the genres in the dataframe to inform your adjustments. "
        "Your task is to interpret the user's feedback to modify the feature vector accurately, ensuring each updated value is consistent with the data patterns observed in the dataframe. "
        "Refine the vector so it more closely aligns with both the user's preferences and the musical characteristics of similar genres. "
        "Format your adjusted feature vector as a Python dictionary: "
        "{{'feature1': value1, 'feature2': value2, 'feature3': value3, ...}}. "
        "This adjustment should be data-driven, grounded in the specifics of the user's feedback and the genre characteristics in the dataframe. "
        "Present only the revised feature vector dictionary, without additional commentary. "
        "This refinement is akin to customizing the song suggestions to better fit the user's musical tastes, using the dataframe as a guide for realistic and appropriate feature values."
    ).format(features, desc, example_feedback)
    
    x = featurizer.gen(feedback_prompt, verbose=True)

    featurizer.gen.reset_conversation()
    
    allowed_genres = featurizer.sp.recommendation_genre_seeds()["genres"]
    
    prompt = (
        "Based on the user's feedback regarding their previous song recommendations, which is as follows: '{}', "
        "determine the most suitable music genre(s) for their next listening experience. "
        "You have access to a range of genres: {}. Your task is to carefully analyze the user's feedback, "
        "then select and recommend genres from this list that closely match the user's music preferences as described in their feedback. "
        "Critically assess the defining characteristics of each genre in the context of the feedback. "
        "Present your genre recommendations in a Python list format. For example, if 'ambient', 'classical', and 'jazz' are the best matches, "
        "format your response as: ['ambient', 'classical', 'jazz']. "
        "Ensure that your selections are directly informed by the user's feedback, focusing on the specific aspects of the music they enjoyed or did not enjoy. "
        "Your genre recommendations should accurately reflect the musical preferences and style indicated in the user's feedback."
    ).format(example_feedback, ', '.join(allowed_genres))

    y = featurizer.gen(prompt, verbose=True)

    featurizer.generate_playlist(eval(x), eval(y), "post_feedback")
