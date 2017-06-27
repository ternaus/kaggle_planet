"""
map train labels to a more clean format
"""

from __future__ import division

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
labels = pd.read_csv('../data/train_v2.csv')

vectorizer = CountVectorizer()
one_hot_labels = vectorizer.fit_transform(labels['tags'].values).toarray()
values, keys = zip(*vectorizer.vocabulary_.items())

ids = labels['image_name']
labels = pd.DataFrame(one_hot_labels, columns=np.array(values)[np.argsort(keys)])
labels['image_name'] = ids

labels.to_csv('../data/train_labels.csv', index=False)
