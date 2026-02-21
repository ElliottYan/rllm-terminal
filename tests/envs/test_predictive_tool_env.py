"""Tests for PredictiveToolEnvironment with similarity reward."""

from unittest.mock import Mock, patch

import pytest

from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.rewards.reward_fn import RewardFunction, RewardOutput
from rllm_ext.environments.predictive_tool_env import PredictiveToolEnvironment
from rllm_ext.rewards.prediction_similarity import SimilarityConfig


class MockRewardFunction(RewardFunction):
    """Mock reward function for testing."""

    def __call__(self, task_info, action, **kwargs):
        return RewardOutput(reward=0.0, metadata={"mock": True})


class TestPredictiveToolEnvironment:
    """Test PredictiveToolEnvironment with similarity reward."""

    def test_init_without_similarity_config(self):
        """Test initialization without similarity config (default disabled)."""
        env = PredictiveToolEnvironment(
            task={"question": "test"},
            reward_fn=MockRewardFunction(),
        )
        assert env.similarity_config.enabled is False
        assert isinstance(env, ToolEnvironment)

    def test_init_with_similarity_config_dict(self):
        """Test initialization with similarity config as dict."""
        env = PredictiveToolEnvironment(
            task={"question": "test"},
            reward_fn=MockRewardFunction(),
            similarity_config={
                "enabled": True,
                "weight": 0.2,
                "n": 2,
                "min_length": 5,
            },
        )
        assert env.similarity_config.enabled is True
        assert env.similarity_config.weight == 0.2
        assert env.similarity_config.n == 2
        assert env.similarity_config.min_length == 5

    def test_init_with_similarity_config_object(self):
        """Test initialization with SimilarityConfig object."""
        config = SimilarityConfig(enabled=True, weight=0.15, n=3)
        env = PredictiveToolEnvironment(
            task={"question": "test"},
            reward_fn=MockRewardFunction(),
            similarity_config=config,
        )
        assert env.similarity_config == config

    def test_reset(self):
        """Test reset method."""
        env = PredictiveToolEnvironment(
            task={"question": "test"},
            similarity_config={"enabled": True},
        )
        obs, info = env.reset()

        assert obs == {"question": "test"}
        assert isinstance(info, dict)

    def test_reset_with_new_task(self):
        """Test reset with new task parameter."""
        env = PredictiveToolEnvironment(
            task={"question": "old"},
            similarity_config={"enabled": True},
        )

        new_task = {"question": "new"}
        obs, info = env.reset(task=new_task)

        assert obs == new_task
        assert env.task == new_task

    def test_step_with_standard_action_list(self):
        """Test step with standard list action (backward compatibility)."""
        env = PredictiveToolEnvironment(
            task={"question": "test"},
            similarity_config={"enabled": True, "min_length": 1},
        )
        env.reset()

        with patch.object(env, "_execute_tool_calls") as mock_execute:
            mock_execute.return_value = {"call_1": "Tool output result"}

            action = [
                {
                    "id": "call_1",
                    "function": {"name": "python", "arguments": '{"code": "1+1"}'},
                }
            ]

            obs, reward, done, info = env.step(action)

            assert obs == {"tool_outputs": {"call_1": "Tool output result"}}
            assert reward == 0.0
            assert done is False
            # No prediction similarity info for standard action
            assert "prediction_similarity" not in info

    def test_step_with_standard_action_dict(self):
        """Test step with standard dict action (backward compatibility)."""
        env = PredictiveToolEnvironment(
            task={"question": "test"},
            similarity_config={"enabled": True},
        )
        env.reset()

        with patch.object(env, "_execute_tool_calls") as mock_execute:
            mock_execute.return_value = {"call_1": "Output"}

            action = {
                "id": "call_1",
                "function": {"name": "python", "arguments": '{"code": "print(1)"}'},
            }

            obs, reward, done, info = env.step(action)

            assert obs == {"tool_outputs": {"call_1": "Output"}}
            assert "prediction_similarity" not in info

    def test_step_with_standard_action_string(self):
        """Test step with string action (final answer)."""
        env = PredictiveToolEnvironment(
            task={"question": "test"},
            reward_fn=MockRewardFunction(),
            similarity_config={"enabled": True},
        )
        env.reset()

        action = "This is my final answer"
        obs, reward, done, info = env.step(action)

        assert obs == {}
        assert done is True
        assert "prediction_similarity" not in info

    def test_step_with_enhanced_action_no_prediction(self):
        """Test enhanced action format without prediction."""
        env = PredictiveToolEnvironment(
            task={"question": "test"},
            similarity_config={"enabled": True},
        )
        env.reset()

        with patch.object(env, "_execute_tool_calls") as mock_execute:
            mock_execute.return_value = {"call_1": "Tool output"}

            enhanced_action = {
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "python", "arguments": "{}"}}
                ],
                "prediction": None,
            }

            obs, reward, done, info = env.step(enhanced_action)

            assert obs == {"tool_outputs": {"call_1": "Tool output"}}
            assert "prediction_similarity" not in info

    def test_step_with_enhanced_action_with_prediction_perfect_match(self):
        """Test enhanced action with perfect prediction match."""
        env = PredictiveToolEnvironment(
            task={"question": "test"},
            similarity_config={"enabled": True, "weight": 0.1, "min_length": 1},
        )
        env.reset()

        with patch.object(env, "_execute_tool_calls") as mock_execute:
            mock_execute.return_value = {"call_1": "the result is 42"}

            enhanced_action = {
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "python", "arguments": "{}"}}
                ],
                "prediction": {
                    "text": "the result is 42",
                    "raw_text": "the result is 42",
                    "prompt": "predict",
                },
            }

            obs, reward, done, info = env.step(enhanced_action)

            # Should have similarity reward
            assert "prediction_similarity" in info
            assert info["prediction_similarity"]["raw_reward"] == pytest.approx(0.1, abs=0.01)
            assert info["prediction_similarity"]["score"] == pytest.approx(1.0, abs=0.01)
            # Total reward should include similarity
            assert reward == pytest.approx(0.1, abs=0.01)

    def test_step_with_enhanced_action_with_prediction_no_match(self):
        """Test enhanced action with no prediction match."""
        env = PredictiveToolEnvironment(
            task={"question": "test"},
            similarity_config={"enabled": True, "weight": 0.1, "min_length": 1},
        )
        env.reset()

        with patch.object(env, "_execute_tool_calls") as mock_execute:
            mock_execute.return_value = {"call_1": "calculator result"}

            enhanced_action = {
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "python", "arguments": "{}"}}
                ],
                "prediction": {
                    "text": "weather forecast",
                    "raw_text": "weather forecast",
                    "prompt": "predict",
                },
            }

            obs, reward, done, info = env.step(enhanced_action)

            # Should have similarity info but zero reward
            assert "prediction_similarity" in info
            assert info["prediction_similarity"]["raw_reward"] == 0.0
            assert info["prediction_similarity"]["score"] == 0.0
            assert reward == 0.0

    def test_step_with_enhanced_action_short_output_skipped(self):
        """Test that short outputs are skipped based on min_length."""
        env = PredictiveToolEnvironment(
            task={"question": "test"},
            similarity_config={"enabled": True, "weight": 0.1, "min_length": 5},
        )
        env.reset()

        with patch.object(env, "_execute_tool_calls") as mock_execute:
            mock_execute.return_value = {"call_1": "42"}  # Too short

            enhanced_action = {
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "python", "arguments": "{}"}}
                ],
                "prediction": {
                    "text": "42",
                    "raw_text": "42",
                    "prompt": "predict",
                },
            }

            obs, reward, done, info = env.step(enhanced_action)

            # Should not compute similarity for short outputs
            assert "prediction_similarity" not in info
            assert reward == 0.0

    def test_step_with_enhanced_action_multiple_tool_outputs(self):
        """Test enhanced action with multiple tool outputs."""
        env = PredictiveToolEnvironment(
            task={"question": "test"},
            similarity_config={"enabled": True, "weight": 0.1, "min_length": 1},
        )
        env.reset()

        with patch.object(env, "_execute_tool_calls") as mock_execute:
            # Multiple tool outputs
            mock_execute.return_value = {
                "call_1": "first output",
                "call_2": "second output",
            }

            enhanced_action = {
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "python", "arguments": "{}"}},
                    {"id": "call_2", "function": {"name": "python", "arguments": "{}"}},
                ],
                "prediction": {
                    "text": "first output second output",
                    "raw_text": "first output second output",
                    "prompt": "predict",
                },
            }

            obs, reward, done, info = env.step(enhanced_action)

            # Should compute similarity against concatenated outputs
            assert "prediction_similarity" in info
            # Concatenated actual: "first output second output"
            # Prediction: "first output second output" (perfect match)
            assert info["prediction_similarity"]["score"] == pytest.approx(1.0, abs=0.01)

    def test_step_similarity_disabled(self):
        """Test that similarity is not computed when disabled."""
        env = PredictiveToolEnvironment(
            task={"question": "test"},
            similarity_config={"enabled": False},
        )
        env.reset()

        with patch.object(env, "_execute_tool_calls") as mock_execute:
            mock_execute.return_value = {"call_1": "output"}

            enhanced_action = {
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "python", "arguments": "{}"}}
                ],
                "prediction": {"text": "output", "raw_text": "output", "prompt": "predict"},
            }

            obs, reward, done, info = env.step(enhanced_action)

            # Should not compute similarity
            assert "prediction_similarity" not in info
            assert reward == 0.0

    def test_similarity_info_structure(self):
        """Test that similarity info has correct structure."""
        env = PredictiveToolEnvironment(
            task={"question": "test"},
            similarity_config={"enabled": True, "weight": 0.1, "min_length": 1},
        )
        env.reset()

        with patch.object(env, "_execute_tool_calls") as mock_execute:
            mock_execute.return_value = {"call_1": "the tool output"}

            enhanced_action = {
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "python", "arguments": "{}"}}
                ],
                "prediction": {
                    "text": "the tool output",
                    "raw_text": "the tool output",
                    "prompt": "predict",
                },
            }

            obs, reward, done, info = env.step(enhanced_action)

            assert "prediction_similarity" in info
            sim_info = info["prediction_similarity"]
            assert "score" in sim_info
            assert "raw_reward" in sim_info
            assert "prediction_length" in sim_info
            assert "actual_length" in sim_info
            assert isinstance(sim_info["score"], float)
            assert isinstance(sim_info["raw_reward"], float)
            assert isinstance(sim_info["prediction_length"], int)
            assert isinstance(sim_info["actual_length"], int)

    def test_from_dict_without_similarity_config(self):
        """Test from_dict without similarity config."""
        env_args = {
            "question": "test",
            "tools": ["python"],
            "max_steps": 10,
        }

        with patch("rllm.environments.tools.tool_env.MultiTool"):
            env = PredictiveToolEnvironment.from_dict(env_args)

            assert isinstance(env, PredictiveToolEnvironment)
            assert env.similarity_config.enabled is False

    def test_from_dict_with_similarity_config(self):
        """Test from_dict with similarity config."""
        env_args = {
            "question": "test",
            "tools": ["python"],
            "max_steps": 10,
            "similarity_config": {
                "enabled": True,
                "weight": 0.2,
                "n": 2,
                "min_length": 5,
            },
        }

        with patch("rllm.environments.tools.tool_env.MultiTool"):
            env = PredictiveToolEnvironment.from_dict(env_args)

            assert env.similarity_config.enabled is True
            assert env.similarity_config.weight == 0.2
            assert env.similarity_config.n == 2
            assert env.similarity_config.min_length == 5

    def test_is_multithread_safe(self):
        """Test that PredictiveToolEnvironment is multithread-safe."""
        assert PredictiveToolEnvironment.is_multithread_safe() is True

    def test_full_flow_with_prediction(self):
        """Test complete flow with prediction step."""
        env = PredictiveToolEnvironment(
            task={"question": "What is 2+2?"},
            similarity_config={"enabled": True, "weight": 0.1, "min_length": 1},
        )

        # Reset
        obs, info = env.reset()
        assert obs == {"question": "What is 2+2?"}

        # Step with prediction
        with patch.object(env, "_execute_tool_calls") as mock_execute:
            mock_execute.return_value = {"call_1": "the result is 4"}

            enhanced_action = {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "python", "arguments": '{"code": "2+2"}'},
                    }
                ],
                "prediction": {
                    "text": "the result is 4",
                    "raw_text": "I predict the result is 4",
                    "prompt": "Predict the output",
                },
            }

            obs, reward, done, info = env.step(enhanced_action)

            assert not done
            assert "tool_outputs" in obs
            assert "prediction_similarity" in info
            assert reward > 0  # Should have similarity reward

    def test_code_output_similarity(self):
        """Test similarity with realistic code output."""
        env = PredictiveToolEnvironment(
            task={"question": "test"},
            similarity_config={"enabled": True, "weight": 0.1, "min_length": 4},
        )
        env.reset()

        with patch.object(env, "_execute_tool_calls") as mock_execute:
            # Realistic code output
            mock_execute.return_value = {
                "call_1": "x = 5\ny = 10\nresult = x + y\nprint(result)"
            }

            enhanced_action = {
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "python", "arguments": "{}"}}
                ],
                "prediction": {
                    "text": "x = 5\ny = 10\nresult = 15\nprint(result)",
                    "raw_text": "x = 5\ny = 10\nresult = 15\nprint(result)",
                    "prompt": "predict",
                },
            }

            obs, reward, done, info = env.step(enhanced_action)

            # Should have partial similarity (x=5, y=10 match)
            assert "prediction_similarity" in info
            assert info["prediction_similarity"]["score"] > 0
            assert info["prediction_similarity"]["score"] < 1.0

    def test_math_output_similarity(self):
        """Test similarity with math tool output."""
        env = PredictiveToolEnvironment(
            task={"question": "test"},
            similarity_config={"enabled": True, "weight": 0.1, "min_length": 4},
        )
        env.reset()

        with patch.object(env, "_execute_tool_calls") as mock_execute:
            mock_execute.return_value = {
                "call_1": "Calculating: 2 + 2 = 4\nThe result is 4"
            }

            enhanced_action = {
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "python", "arguments": "{}"}}
                ],
                "prediction": {
                    "text": "Calculating: 2 + 2 = 4\nThe result is 4.0",
                    "raw_text": "Calculating: 2 + 2 = 4\nThe result is 4.0",
                    "prompt": "predict",
                },
            }

            obs, reward, done, info = env.step(enhanced_action)

            # Should have high similarity
            assert "prediction_similarity" in info
            assert info["prediction_similarity"]["score"] > 0.8

    def test_empty_prediction_text(self):
        """Test with empty prediction text."""
        env = PredictiveToolEnvironment(
            task={"question": "test"},
            similarity_config={"enabled": True, "weight": 0.1, "min_length": 1},
        )
        env.reset()

        with patch.object(env, "_execute_tool_calls") as mock_execute:
            mock_execute.return_value = {"call_1": "some output"}

            enhanced_action = {
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "python", "arguments": "{}"}}
                ],
                "prediction": {
                    "text": "",
                    "raw_text": "",
                    "prompt": "predict",
                },
            }

            obs, reward, done, info = env.step(enhanced_action)

            # Empty prediction should not compute similarity
            assert "prediction_similarity" not in info
            assert reward == 0.0

    def test_no_tool_outputs_in_observation(self):
        """Test when observation has no tool_outputs."""
        env = PredictiveToolEnvironment(
            task={"question": "test"},
            reward_fn=MockRewardFunction(),
            similarity_config={"enabled": True, "weight": 0.1, "min_length": 1},
        )
        env.reset()

        # String action (final answer) - no tool_outputs
        action = "Final answer"

        obs, reward, done, info = env.step(action)

        assert obs == {}
        assert "prediction_similarity" not in info

    def test_different_n_values(self):
        """Test with different n-gram sizes."""
        for n in [1, 2, 3, 4]:
            env = PredictiveToolEnvironment(
                task={"question": "test"},
                similarity_config={"enabled": True, "weight": 0.1, "n": n, "min_length": 1},
            )
            env.reset()

            with patch.object(env, "_execute_tool_calls") as mock_execute:
                mock_execute.return_value = {"call_1": "the cat sat on the mat"}

                enhanced_action = {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "python", "arguments": "{}"},
                        }
                    ],
                    "prediction": {
                        "text": "the cat sat on the mat",
                        "raw_text": "the cat sat on the mat",
                        "prompt": "predict",
                    },
                }

                obs, reward, done, info = env.step(enhanced_action)

                # Perfect match should give max reward regardless of n
                assert info["prediction_similarity"]["score"] == pytest.approx(1.0, abs=0.01)

    def test_weight_scaling(self):
        """Test that weight scales the reward correctly."""
        for weight in [0.05, 0.1, 0.2]:
            env = PredictiveToolEnvironment(
                task={"question": "test"},
                similarity_config={"enabled": True, "weight": weight, "min_length": 1},
            )
            env.reset()

            with patch.object(env, "_execute_tool_calls") as mock_execute:
                mock_execute.return_value = {"call_1": "same output"}

                enhanced_action = {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "python", "arguments": "{}"},
                        }
                    ],
                    "prediction": {
                        "text": "same output",
                        "raw_text": "same output",
                        "prompt": "predict",
                    },
                }

                obs, reward, done, info = env.step(enhanced_action)

                # Perfect match should give full weight as reward
                assert info["prediction_similarity"]["raw_reward"] == pytest.approx(
                    weight, abs=0.01
                )
