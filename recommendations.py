from params_proto import PrefixProto
import logging

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


logger = logging.getLogger('examples.artist_recommendations')
logging.basicConfig(level='INFO')

client_credentials_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


class Args(PrefixProto):
    artist = "NOTD"


def get_artist(name):
    results = sp.search(q='artist:' + name, type='artist')
    items = results['artists']['items']
    if len(items) > 0:
        return items[0]
    else:
        return None


def show_recommendations_for_artist(artist):
    results = sp.recommendations(seed_artists=[artist['id']])
    for track in results['tracks']:
        logger.info('Recommendation: %s - %s', track['name'],
                    track['artists'][0]['name'])


def main():
    artist = get_artist(Args.artist)
    if artist:
        show_recommendations_for_artist(artist)
    else:
        logger.error("Can't find that artist", Args.artist)


if __name__ == '__main__':
    main()
