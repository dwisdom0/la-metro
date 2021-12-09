import os
# stop tensorflow from printing novels to stdout
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import tensorflow as tf

from sklearn.cluster import DBSCAN


def read_stops(p: str):
  """
  Read in the .csv file of metro stops
  
  :param p: The path to the .csv file of metro stops
  """
  return pd.read_csv(p)


def read_encodings(p: str) -> tf.Tensor:
  """
  Unpickle the Universal Sentence Encoder v4 encodings
  and return them
  
  This function doesn't make any attempt to patch the security holes in `pickle`.
  
  :param p: Path to the encodings
 
  :returns: A Tensor of the encodings with shape (number of sentences, 512)
  """
  with open(p, 'rb') as f:
    encodings = pickle.load(f)
  return encodings


def cluster_encodings(encodings: tf.Tensor) -> np.ndarray:
  """
  Cluster the sentence encodings using DBSCAN.

  :param encodings: A Tensor of sentence encodings with shape
                    (number of sentences, 512) 

  :returns: a NumPy array of the cluster labels
  """
  # I know the hyperparams I want from the EDA I did in the notebook
  clusterer = DBSCAN(eps=0.7, min_samples=100).fit(encodings)
  return clusterer.labels_


def cluster_lat_lon(df: pd.DataFrame) -> np.ndarray:
  """
  Cluster the metro stops by their latitude and longitude using DBSCAN.

  :param df: A Pandas DataFrame of stops that has 'latitude` and 'longitude` columns

  :returns: a NumPy array of the cluster labels
  """
  # I know the hyperparams I want from the EDA I did in the notebook
  clusterer = DBSCAN(eps=0.025, min_samples=100).fit(df[['latitude', 'longitude']])
  return clusterer.labels_


def plot_example(df: pd.DataFrame, labels: np.ndarray):
  """
  Plot the geographic clustering

  :param df: A Pandas DataFrame of stops that has 'latitude` and 'longitude` columns
  :param labels: a NumPy array of the cluster labels
  """ 
  with open('.mapbox_token', 'r') as f:
    token = f.read().strip()

  px.set_mapbox_access_token(token)
  labels = labels.astype('str')

  fig = px.scatter_mapbox(df, lon='longitude', lat='latitude',
                          hover_name='display_name',
                          color=labels,
                          zoom=8,
                          color_discrete_sequence=px.colors.qualitative.Dark24)
  fig.show()
  

def plot_venice_blvd(df: pd.DataFrame, labels: np.ndarray):
  """
  Plot the metro stops and color them based on their names

  :param df: A Pandas DataFrame of stops that has 'latitude` and 'longitude` columns
  :param labels: a NumPy array of the cluster labels
  """
  with open('.mapbox_token', 'r') as f:
    token = f.read().strip()

  px.set_mapbox_access_token(token)
  venice_blvd = {'lat': 34.008350,
                 'lon': -118.425362}
  labels = labels.astype('str')

  fig = px.scatter_mapbox(df, lat='latitude', lon='longitude',
                          color=labels,
                          hover_name='display_name', 
                          center=venice_blvd,
                          zoom=12,
                          color_discrete_sequence=px.colors.qualitative.Dark24)

  fig.show()
  
  
def main(data_path: str, enc_path: str):
  df = read_stops(data_path)

  # Cluster based on lat/lon
  example_labels = cluster_lat_lon(df)
  plot_example(df, example_labels)

  # Cluster based on the name of the stop
  encodings = read_encodings(enc_path)
  encoding_labels = cluster_encodings(encodings)
  plot_venice_blvd(df, encoding_labels)
  

if __name__ == '__main__':
  import argparse

  p = argparse.ArgumentParser()
  p.add_argument('--data_path',
                 nargs='?',
                 default='data/stops.csv',
                 help="Path to the dataset of LA Metro stops. Defaults to 'data/stops.csv'")
  p.add_argument('--enc_path',
                 nargs='?',
                 default='data/encodings.pkl',
                 help="Path to the pickled encodings. Defaults to 'data/encodings.pkl'")
  args = p.parse_args()

  main(**vars(args))

