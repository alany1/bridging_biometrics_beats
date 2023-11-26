import os

print(os.environ["SPOTIPY_CLIENT_ID"])
print(os.environ["SPOTIPY_CLIENT_SECRET"])

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

if __name__ == '__main__':
    birdy_uri = 'spotify:artist:2WX2uTcsvV5OnS0inACecP'
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
    
    results = spotify.artist_albums(birdy_uri, album_type='album')
    albums = results['items']
    while results['next']:
        results = spotify.next(results)
        albums.extend(results['items'])
    
    for album in albums:
        print(album['name'])
