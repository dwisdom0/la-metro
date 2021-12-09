# Quickstart

## Set up the environment
```shell
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Build the dataset
```shell
$ python build_dataset.py
```

## Plot the two clusters
```shell
$ python cluster.py
```

# Description
First, I clustered the
stops by their geographic location.
The DBSCAN algorithm finds three clusters.
Points labeled `-1` aren't part of any cluster.
Clicking on `-1` in the legend will turn off those points.


In the next plot, I encoded the names of all the stops using the Universal Sentence Encoder v4.
I then clustered those encodings so that I could group the stops based on their names
instead of their geographic position.
As I expected, stops on the same road end up close enough to each other that DBSCAN can cluster them together.


Sometimes, however, a stop has a name that means something to the encoder.
When that happens, the encoding ends up too far away from the rest of the stops on that road.
For example, the stops on Venice Blvd get clustered together,
but the stop "Venice / Lincoln" ends up somewhere else.
I assume it ends up somewhere else because the encoder recognizes "Lincoln"
and that meaning overpowers the "Venice" meaning enough that the encoding
is too far away from the rest of the "Venice" stops.
A few other examples on Venice Blvd are "Saint Andrews," "Harvard," and "Beethoven."
There are also a few that I don't ascribe much meaning to, such as "Girard" and "Robertson."


There's a lot more to dig into here but I'll leave it there for now.
My mind first jumps to adversarial prompts that use famous names to move the encoding
around in the encoding space.

