from arguments import from_config

from bento.training.data.embedding_types import EmbeddingSpec, EmbeddingType


def test_parse_embedding_spec():
    d = {
        "name": "test",
        "vocabulary_type": "static",
        "dimension": None,
        "shared_names": [],
        "type": "static",
    }
    m = from_config(EmbeddingSpec, d)
    assert m.type == EmbeddingType.STATIC
