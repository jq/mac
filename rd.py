import tensorflow as tf

# tfra does some patching on tensorflow so MUST be imported after importing tf
import tensorflow_recommenders as tfrs
import tensorflow_recommenders_addons as tfra
import tensorflow_recommenders_addons.dynamic_embedding as de
import tensorflow_datasets as tfds

import functools
from typing import Dict
import dataclasses
import matplotlib.pyplot as plt

from bento.training.data.embedding_types import EmbeddingSpec

# plt.plot([1, 2, 3, 4])
# plt.ylabel('some numbers')
# # 使用 plt.show() 来显示图形
# plt.show()

EmbeddingSpec("categorical_feature", vocabulary_type="dynamic", dimension=3, initializer=tf.keras.initializers.Constant(3)),

# Interactions dataset
raw_ratings_dataset = tfds.load("movielens/1m-ratings", split="train")
# Candidates dataset
raw_movies_dataset = tfds.load("movielens/1m-movies", split="train")

def check_dataset(raw_ratings_dataset):
    df = tfds.as_dataframe(raw_ratings_dataset.take(100))
    df.head()
    for item in raw_ratings_dataset.take(1):
        print(item)


max_token_length = 6
pad_token = "[PAD]"
punctuation_regex = "[\!\"#\$%&\(\)\*\+,-\.\/\:;\<\=\>\?@\[\]\\\^_`\{\|\}~\\t\\n]"

def process_text(x: tf.Tensor, max_token_length: int, punctuation_regex: str) -> tf.Tensor:

    return tf.strings.split(
        tf.strings.regex_replace(
            tf.strings.lower(x["movie_title"]), punctuation_regex, ""
        )
    )[:max_token_length]


def process_ratings_dataset(ratings_dataset: tf.data.Dataset) -> tf.data.Dataset:

    partial_process_text = functools.partial(
        process_text, max_token_length=max_token_length, punctuation_regex=punctuation_regex
    )

    preprocessed_movie_title_dataset = ratings_dataset.map(
        lambda x: partial_process_text(x)
    )

    processed_dataset = tf.data.Dataset.zip(
        (ratings_dataset, preprocessed_movie_title_dataset)
    ).map(
        lambda x,y: {"user_id": x["user_id"]} | {"movie_title": y}
    )

    return processed_dataset


def process_movies_dataset(movies_dataset: tf.data.Dataset) -> tf.data.Dataset:

    partial_process_text = functools.partial(
        process_text, max_token_length=max_token_length, punctuation_regex=punctuation_regex
    )

    processed_dataset = raw_movies_dataset.map(
        lambda x: partial_process_text(x)
    )

    return processed_dataset

processed_ratings_dataset = process_ratings_dataset(raw_ratings_dataset)
for item in processed_ratings_dataset.take(3):
    print(item)

batch_size=4096
seed=2023
train_size = int(len(processed_ratings_dataset) * 0.9)
validation_size = len(processed_ratings_dataset) - train_size
print(f"Train size: {train_size}")
print(f"Validation size: {validation_size}")

# Things we're not considering
# out of sample validation
# out of time validation

@dataclasses.dataclass(frozen=True)
class TrainingDatasets:
    train_ds: tf.data.Dataset
    validation_ds: tf.data.Dataset

@dataclasses.dataclass(frozen=True)
class RetrievalDatasets:
    training_datasets: TrainingDatasets
    candidate_dataset: tf.data.Dataset

def pad_and_batch_ratings_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:

    return dataset.padded_batch(
        batch_size,
        padded_shapes={
            "user_id": tf.TensorShape([]),
            "movie_title": tf.TensorShape([max_token_length,])
        }, padding_values={
            "user_id": pad_token,
            "movie_title": pad_token
        }
    )


def split_train_validation_datasets(ratings_dataset: tf.data.Dataset) -> TrainingDatasets:

    shuffled_dataset = ratings_dataset.shuffle(buffer_size=5*batch_size, seed=seed)
    train_ds = shuffled_dataset.skip(validation_size).shuffle(buffer_size=10*batch_size).apply(pad_and_batch_ratings_dataset)
    validation_ds = shuffled_dataset.take(validation_size).apply(pad_and_batch_ratings_dataset)

    return TrainingDatasets(train_ds=train_ds, validation_ds=validation_ds)

def pad_and_batch_candidate_dataset(movies_dataset: tf.data.Dataset) -> tf.data.Dataset:
    return movies_dataset.padded_batch(
        batch_size,
        padded_shapes=tf.TensorShape([max_token_length,]),
        padding_values=pad_token
    )

def create_datasets() -> RetrievalDatasets:

    raw_ratings_dataset = tfds.load("movielens/1m-ratings", split="train")
    raw_movies_dataset = tfds.load("movielens/1m-movies", split="train")

    processed_ratings_dataset = process_ratings_dataset(raw_ratings_dataset)
    processed_movies_dataset = process_movies_dataset(raw_movies_dataset)

    training_datasets = split_train_validation_datasets(processed_ratings_dataset)
    candidate_dataset = pad_and_batch_candidate_dataset(processed_movies_dataset)

    return RetrievalDatasets(training_datasets=training_datasets, candidate_dataset=candidate_dataset)

datasets = create_datasets()
print(f"Train dataset size (after batching): {len(datasets.training_datasets.train_ds)}")
print(f"Validation dataset size (after batching): {len(datasets.training_datasets.validation_ds)}")

def get_user_id_lookup_layer(dataset: tf.data.Dataset) -> tf.keras.layers.Layer:
    user_lookup_layer = tf.keras.layers.StringLookup(mask_token=None)
    user_lookup_layer.adapt(dataset.map(lambda x: x["user_id"]))
    return user_lookup_layer

import tensorflow_recommenders_addons.dynamic_embedding as de

def build_de_user_model(user_id_lookup_layer: tf.keras.layers.StringLookup) -> tf.keras.layers.Layer:
    vocab_size = user_id_lookup_layer.vocabulary_size()
    return tf.keras.Sequential([
        # Fix from https://github.com/keras-team/keras/issues/16101
        tf.keras.layers.InputLayer(input_shape=(), dtype=tf.string),
        user_id_lookup_layer,
        de.keras.layers.Embedding(
            embedding_size=64,
            initializer=tf.random_uniform_initializer(),
            init_capacity=int(vocab_size*0.8),
            restrict_policy=de.FrequencyRestrictPolicy,
            name="UserDynamicEmbeddingLayer"
        ),
        tf.keras.layers.Dense(64, activation="gelu"),
        tf.keras.layers.Dense(32),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))

    ], name='user_model')

def build_de_item_model(movie_title_lookup_layer: tf.keras.layers.StringLookup) -> tf.keras.layers.Layer:
    vocab_size = movie_title_lookup_layer.vocabulary_size()
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(max_token_length), dtype=tf.string),
        movie_title_lookup_layer,
        de.keras.layers.SquashedEmbedding(
            embedding_size=64,
            initializer=tf.random_uniform_initializer(),
            init_capacity=int(vocab_size*0.8),
            restrict_policy=de.FrequencyRestrictPolicy,
            combiner="mean",
            name="ItemDynamicEmbeddingLayer"
        ),
        tf.keras.layers.Dense(64, activation="gelu"),
        tf.keras.layers.Dense(32),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ])
def build_user_model(user_id_lookup_layer: tf.keras.layers.StringLookup):
    vocab_size = user_id_lookup_layer.vocabulary_size()
    return tf.keras.Sequential([
        # Fix from https://github.com/keras-team/keras/issues/16101
        tf.keras.layers.InputLayer(input_shape=(), dtype=tf.string),
        user_id_lookup_layer,
        tf.keras.layers.Embedding(vocab_size, 64),
        tf.keras.layers.Dense(64, activation="gelu"),
        tf.keras.layers.Dense(32),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))

    ], name='user_model')
def get_movie_title_lookup_layer(dataset: tf.data.Dataset) -> tf.keras.layers.Layer:
    movie_title_lookup_layer = tf.keras.layers.StringLookup(mask_token=pad_token)
    movie_title_lookup_layer.adapt(dataset.map(lambda x: x["movie_title"]))
    return movie_title_lookup_layer

def build_item_model(movie_title_lookup_layer: tf.keras.layers.StringLookup):
    vocab_size = movie_title_lookup_layer.vocabulary_size()
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(max_token_length), dtype=tf.string),
        movie_title_lookup_layer,
        tf.keras.layers.Embedding(vocab_size, 64),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation="gelu"),
        tf.keras.layers.Dense(32),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ])

# TODO: example models
user_id_lookup_layer = get_user_id_lookup_layer(datasets.training_datasets.train_ds)
print(f"Vocabulary size (user_id): {len(user_id_lookup_layer.get_vocabulary())}")

movie_title_lookup_layer = get_movie_title_lookup_layer(datasets.training_datasets.train_ds)
print(f"Vocabulary size (movie_title): {len(movie_title_lookup_layer.get_vocabulary())}")

class TwoTowerModel(tfrs.Model):
    # We derive from a custom base class to help reduce boilerplate. Under the hood,
    # these are still plain Keras Models.

    def __init__(self,user_model: tf.keras.Model,item_model: tf.keras.Model,task: tfrs.tasks.Retrieval):
        super().__init__()

        # Set up user and movie representations.
        self.user_model = user_model
        self.item_model = item_model
        # Set up a retrieval task.
        self.task = task

    def compute_loss(self, features: Dict[str, tf.Tensor], training=False) -> tf.Tensor:
        # Define how the loss is computed.
        user_embeddings = self.user_model(features["user_id"])
        movie_embeddings = self.item_model(features["movie_title"])
        return self.task(user_embeddings, movie_embeddings)

def create_two_tower_model(dataset: tf.data.Dataset, candidate_dataset: tf.data.Dataset) -> tf.keras.Model:
    user_id_lookup_layer = get_user_id_lookup_layer(dataset)
    movie_title_lookup_layer = get_movie_title_lookup_layer(dataset)
    user_model = build_user_model(user_id_lookup_layer)
    item_model = build_item_model(movie_title_lookup_layer)
    task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidate_dataset.map(item_model)
        ),
    )

    model = TwoTowerModel(user_model, item_model, task)
    model.compile(optimizer=tf.keras.optimizers.Adam())

    return model

datasets = create_datasets()
model = create_two_tower_model(datasets.training_datasets.train_ds, datasets.candidate_dataset)

history = model.fit(datasets.training_datasets.train_ds, epochs=3, validation_data=datasets.training_datasets.validation_ds)

