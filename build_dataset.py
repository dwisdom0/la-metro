import json
import os
# stop tensorflow from prining novels to stdout
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import sys

import pandas as pd
import requests
import tensorflow as tf
import tensorflow_hub as hub

from time import sleep
from tqdm import tqdm


def query_api(q: str) -> pd.DataFrame:
  """
  Queries the LA Metro's API. This is accomplished by appending 
  `q` to the URL `https://api.metro.net/agencies/lametro/`

  :param q: The API query to send to thte LA Metro's API

  :returns: The data from the query as a Pandas DataFrame
  """
  base = 'https://api.metro.net/agencies/lametro/'
  try:
    res = requests.get(base+q)
  except requests.exceptions.RequestException as e:
    raise e

  okay_codes = [200]
  if res.status_code not in okay_codes:
    raise ValueError(f'Got {res.status_code} response, which is not one of the acceptable responses ({okay_codes})')

  j = json.loads(res.text)
  try:
    return pd.DataFrame.from_dict(j['items'])
  except KeyError as e:
    raise e


def build_dataset() -> pd.DataFrame:
  """
  Goes through every route of the LA Metro and records the all of the stops
  on that route to build a dataset of all of the stops in the LA Metro

  :returns: The ID, latitude, longitude, and name of every stop in the LA Metro
            as a Pandas DataFrame
  """
  # First, get a list of all the routes
  try:
    routes = query_api('routes')
  except (requests.exceptions.RequestException, ValueError, KeyError):
    raise RuntimeError("Couldn't get the routes")

  # Get the list of stops for every route
  # The API doesn't provide a list of the stops,
  # only the list of stops for a specific route
  stops = pd.DataFrame(columns=['id', 'latitude', 'longitude', 'display_name'])
  for route_id in tqdm(routes['id'], ascii=True):
    stop_q = f'routes/{route_id}/stops'
    try:
      new_stops = query_api(stop_q)
    except (requests.exceptions.RequestException, ValueError, KeyError):
      continue
    stops = stops.append(new_stops)
    # Be polite to the API
    sleep(1)

  # Drop stops with duplicate IDs
  # This leaves us with 12_258 stops
  # I found a list of all the stops on another page and it has 12_268 stops
  # That makes me confident that this approach is good enough for
  # this quick excercise
  return stops.drop_duplicates('id')


def encode_sentences(sentences: pd.Series) -> tf.Tensor:
   """
   Uses the Universal Sentence Encoder v4 available on tfhub to encode sentences

   :param sentences: A Pandas Series where each element is a sentence to encode

   :returns: A Tensor of the encodings with shape (number of sentences, 512)
   """
   encode = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
   return encode(sentences)


def main(out_path: str, enc_out_path: str):
  """
  Builds a dataset of every stop in the LA Metro and writes it out as a .csv file

  Encodes the names of the stops and saves the encodings in a .pkl file

  :param out_path: Where to write the resulting .csv file. This should include the name of the file.
                   For example: `data/stops.csv`
  :param enc_out_path: Where to write the pickled encodings. This should include te name of the file.
                       For example: `data/enocdings.pkl`
  """

  try:
    df = build_dataset()
  except RuntimeError as e:
    sys.exit(e)

  dir_path, f_path = os.path.split(out_path)
  if dir_path != '':
    os.makedirs(dir_path, exist_ok=True)
  # There are only 13k rows so .csv should be fine
  df.to_csv(out_path, index=False)

  # Encode the display names and pickle the encodings
  # This ends up being about 25 megabytes
  encodings = encode_sentences(df['display_name'])
  with open(enc_out_path, 'wb') as f:
    pickle.dump(encodings, f)


if __name__ == '__main__':
  import argparse

  p = argparse.ArgumentParser()
  p.add_argument('--out_path',
                 nargs='?',
                 default='data/stops.csv', 
                 help="Path where you would like to save the .csv file. Defaults to 'data/stops.csv'")
  p.add_argument('--enc_out_path',
                 nargs='?',
                 default='data/encodings.pkl',
                 help="Where to save the pickled encodings. Defaults to 'data/encodings.pkl'")

  args = p.parse_args()
  main(**vars(args))
