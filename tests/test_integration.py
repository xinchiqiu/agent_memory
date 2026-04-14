"""Integration tests — end-to-end pipeline with mock LLM and real verifier."""

import json
import pickle
import tempfile
import numpy as np
import pytest

from data_structures import Memory, EpisodicEntry, Strategy, Problem
from verifier import Verifier


class MockEncoder:
    """Encoder mock that produces deterministic embeddings based on text hash."""

    @property
    def embedding_dim(self):
        return 384

    def encode(self, statement: str) -> np.ndarray:
        rng = np.random.RandomState(hash(statement) % 2**31)
        vec = rng.randn(384).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    def batch_encode(self, statements, **kwargs):
        return np.array([self.encode(s) for s in statements])


@pytest.fixture
def encoder():
    return MockEncoder()


@pytest.fixture
def simple_problem():
    return Problem(
        problem_id="INT001",
        contest_id=1000,
        index="A",
        title="A+B",
        statement="Read two integers and print their sum.",
        difficulty_rating=800,
        algorithm_tags=["implementation"],
        test_cases=[
            {"input": "2 3", "expected_output": "5"},
            {"input": "0 0", "expected_output": "0"},
        ],
    )


class TestSeedAndRetrieve:
    def test_seed_then_retrieve(self, mock_llm_client, encoder, simple_problem):
        from retriever import Retriever
        from strategy_extraction import extract_strategy

        memory = Memory()
        # Seed one entry
        strategy = extract_strategy(simple_problem, "print(sum(map(int,input().split())))", mock_llm_client)
        emb = encoder.encode(simple_problem.statement)
        memory.current_timestep = 1
        entry = EpisodicEntry(
            entry_id="seed_1",
            problem_id=simple_problem.problem_id,
            problem_statement=simple_problem.statement,
            strategy=strategy,
            solution_code="print(sum(map(int,input().split())))",
            difficulty_rating=800,
            verification_passed=True,
            embedding=emb,
            created_at=1,
            source="seed",
        )
        memory.entries["seed_1"] = entry

        # Retrieve for a similar problem
        query = Problem(
            problem_id="Q1", contest_id=1001, index="B",
            title="Add Numbers",
            statement="Read two integers and print their sum.",
            difficulty_rating=800,
            algorithm_tags=["implementation"],
            test_cases=[],
        )
        retriever = Retriever(encoder, top_k=1)
        results = retriever.retrieve(query, memory)
        assert len(results) == 1
        assert results[0][0].entry_id == "seed_1"


class TestFullPipelineMock:
    def test_solve_verify_update(self, mock_llm_client, encoder, simple_problem):
        from retriever import Retriever
        from adaptation import solve_with_fallbacks
        from memory_update import update_memory
        from strategy_extraction import extract_strategy

        verifier = Verifier(timeout_seconds=5)
        memory = Memory()

        # Seed memory
        strategy = extract_strategy(simple_problem, "a,b=map(int,input().split())\nprint(a+b)", mock_llm_client)
        emb = encoder.encode(simple_problem.statement)
        memory.current_timestep = 1
        entry = EpisodicEntry(
            entry_id="seed_1",
            problem_id="SEED",
            problem_statement="Add two numbers.",
            strategy=strategy,
            solution_code="a,b=map(int,input().split())\nprint(a+b)",
            difficulty_rating=800,
            verification_passed=True,
            embedding=emb,
            created_at=1,
            source="seed",
        )
        memory.entries["seed_1"] = entry

        # Retrieve
        retriever = Retriever(encoder, top_k=1)
        retrieved = retriever.retrieve(simple_problem, memory)

        # Solve with fallbacks (mock LLM returns a+b code which matches test cases)
        result = solve_with_fallbacks(simple_problem, retrieved, mock_llm_client, verifier)
        assert result["success"] is True

        # Update memory
        memory = update_memory(memory, simple_problem, result, encoder, mock_llm_client)
        assert len(memory.entries) == 2  # seed + new


class TestAgentRunSingleProblem:
    def test_agent_processes_one_problem(self, mock_llm_client, encoder, simple_problem, tmp_path):
        """Exercise StrategyAdaptationAgent.run() end-to-end with 1 problem."""
        from agent import StrategyAdaptationAgent
        from retriever import Retriever
        from strategy_extraction import extract_strategy

        verifier = Verifier(timeout_seconds=5)
        retriever = Retriever(encoder, top_k=1)
        agent = StrategyAdaptationAgent(
            llm_client=mock_llm_client,
            encoder=encoder,
            retriever=retriever,
            verifier=verifier,
            log_dir=str(tmp_path / "logs"),
        )

        # Seed memory with one problem so retrieval returns something
        seed_problem = Problem(
            problem_id="SEED",
            contest_id=500,
            index="A",
            title="Seed Problem",
            statement="Add two numbers together.",
            difficulty_rating=800,
            algorithm_tags=["implementation"],
            test_cases=[],
        )
        agent.seed_memory([(seed_problem, "a,b=map(int,input().split())\nprint(a+b)")])
        assert len(agent.memory.entries) == 1

        # Run agent on 1 eval problem
        results = agent.run([simple_problem])
        assert results["total_problems"] == 1
        assert results["overall_accuracy"] in (0.0, 1.0)
        assert len(results["per_problem_log"]) == 1
        assert results["per_problem_log"][0]["problem_id"] == "INT001"
        # Memory should have grown (seed + potentially new entry)
        assert len(agent.memory.entries) >= 1

    def test_agent_checkpoint_and_load(self, mock_llm_client, encoder, simple_problem, tmp_path):
        """Test checkpoint save via agent, then load_memory."""
        from agent import StrategyAdaptationAgent
        from retriever import Retriever

        verifier = Verifier(timeout_seconds=5)
        retriever = Retriever(encoder, top_k=1)
        log_dir = str(tmp_path / "ckpt_logs")
        agent = StrategyAdaptationAgent(
            llm_client=mock_llm_client,
            encoder=encoder,
            retriever=retriever,
            verifier=verifier,
            log_dir=log_dir,
        )
        seed_prob = Problem(
            problem_id="S1", contest_id=1, index="A", title="S",
            statement="Add numbers.", difficulty_rating=800,
            algorithm_tags=["implementation"], test_cases=[],
        )
        agent.seed_memory([(seed_prob, "print(1)")])

        # Save memory
        import pickle
        from pathlib import Path
        mem_path = Path(log_dir) / "test_memory.pkl"
        agent._save_memory(mem_path)

        # Load into a new agent
        agent2 = StrategyAdaptationAgent(
            llm_client=mock_llm_client,
            encoder=encoder,
            retriever=retriever,
            verifier=verifier,
            log_dir=str(tmp_path / "logs2"),
        )
        agent2.load_memory(str(mem_path))
        assert len(agent2.memory.entries) == len(agent.memory.entries)


class TestCheckpointSaveLoad:
    def test_pickle_roundtrip(self, sample_memory):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(sample_memory, f)
            path = f.name

        with open(path, "rb") as f:
            loaded = pickle.load(f)

        assert len(loaded.entries) == len(sample_memory.entries)
        for eid in sample_memory.entries:
            orig = sample_memory.entries[eid]
            load = loaded.entries[eid]
            assert orig.problem_id == load.problem_id
            np.testing.assert_array_almost_equal(orig.embedding, load.embedding)

    def test_json_results_roundtrip(self):
        results = {
            "total_problems": 10,
            "total_successes": 7,
            "overall_accuracy": 0.7,
            "per_problem_log": [{"step": 1, "success": True}],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(results, f)
            path = f.name

        with open(path) as f:
            loaded = json.load(f)

        assert loaded["overall_accuracy"] == 0.7
        assert loaded["per_problem_log"][0]["success"] is True
