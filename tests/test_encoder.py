"""Tests for encoder.py — problem statement encoding.

These tests require sentence_transformers to be installed.
They use the smallest available model for speed.
"""

import numpy as np
import pytest

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

pytestmark = pytest.mark.skipif(not HAS_ST, reason="sentence_transformers not installed")


@pytest.fixture(scope="module")
def encoder():
    from encoder import ProblemEncoder
    return ProblemEncoder(model_name="all-MiniLM-L6-v2")


class TestProblemEncoder:
    def test_encode_returns_normalized_vector(self, encoder):
        vec = encoder.encode("Sort an array of integers.")
        assert vec.dtype == np.float32
        assert vec.ndim == 1
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5

    def test_encode_deterministic(self, encoder):
        v1 = encoder.encode("Find the longest path in a tree.")
        v2 = encoder.encode("Find the longest path in a tree.")
        np.testing.assert_array_almost_equal(v1, v2)

    def test_batch_encode_matches_single(self, encoder):
        statements = [
            "Find the sum of two numbers.",
            "Sort an array of integers.",
        ]
        batch = encoder.batch_encode(statements)
        for i, s in enumerate(statements):
            single = encoder.encode(s)
            np.testing.assert_array_almost_equal(batch[i], single, decimal=5)

    def test_embedding_dim(self, encoder):
        assert encoder.embedding_dim == 384


class TestCreateEncoder:
    def test_fallback_to_base(self, tmp_path):
        from encoder import create_encoder
        fake_path = str(tmp_path / "no_such_model")
        enc = create_encoder(prefer_finetuned=True, finetuned_path=fake_path)
        assert enc is not None
        assert enc.embedding_dim > 0
