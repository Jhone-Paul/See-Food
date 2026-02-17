#this is the ml classifier.

import tensorflow_datasets as tfds

# using the food101 data set for training. 
(ds_train, ds_test), ds_info = tfds.load(
    'food101',
    split=['train', 'validation'],
    as_supervised=True,
    with_info=True
)

