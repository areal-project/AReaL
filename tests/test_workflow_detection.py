"""Unit tests for workflow detection logic in PPOTrainer."""

import warnings

import pytest

from areal.api import AgentWorkflow, RolloutWorkflow


class DummyRolloutWorkflow(RolloutWorkflow):
    """Test RolloutWorkflow implementation."""

    async def arun_episode(self, engine, data):
        return {}


class DummyAgentNonInheriting:
    """No inheritance - duck typing only."""

    async def run(self, data, **extra_kwargs):
        return 1.0


class TestWorkflowDetection:
    """Test the _requires_proxy_workflow detection logic."""

    @pytest.fixture
    def trainer_with_detection(self):
        """Create a minimal object with the detection method."""
        # We can't easily instantiate a full PPOTrainer in unit tests,
        # so we'll create a simple object with just the method we need
        from areal.trainer import PPOTrainer

        class MinimalTrainer:
            def _requires_proxy_workflow(self, workflow):
                # Copy the implementation from PPOTrainer
                return PPOTrainer._requires_proxy_workflow(self, workflow)

        return MinimalTrainer()

    @pytest.fixture
    def dummy_agent_workflow_class(self):
        """Create a DummyAgentWorkflow class inside fixture to avoid module-level warning."""

        class DummyAgentWorkflow(AgentWorkflow):
            """Test AgentWorkflow implementation (deprecated)."""

            async def run(self, data, **extra_kwargs):
                return 1.0

        return DummyAgentWorkflow

    def test_rollout_workflow_instance_no_proxy(self, trainer_with_detection):
        """RolloutWorkflow instances should NOT require proxy."""
        workflow = DummyRolloutWorkflow()
        assert not trainer_with_detection._requires_proxy_workflow(workflow)

    def test_rollout_workflow_class_no_proxy(self, trainer_with_detection):
        """RolloutWorkflow classes should NOT require proxy."""
        assert not trainer_with_detection._requires_proxy_workflow(DummyRolloutWorkflow)

    def test_rollout_workflow_string_no_proxy(self, trainer_with_detection):
        """String paths to RolloutWorkflow should NOT require proxy."""
        workflow = "areal.workflow.rlvr.RLVRWorkflow"
        assert not trainer_with_detection._requires_proxy_workflow(workflow)

    def test_agent_workflow_instance_requires_proxy(
        self, trainer_with_detection, dummy_agent_workflow_class
    ):
        """AgentWorkflow instances should require proxy and emit deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            workflow = dummy_agent_workflow_class()
            assert trainer_with_detection._requires_proxy_workflow(workflow)
            # Verify deprecation warning was emitted
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "inherits from deprecated AgentWorkflow" in str(w[0].message)

    def test_agent_workflow_class_requires_proxy(
        self, trainer_with_detection, dummy_agent_workflow_class
    ):
        """AgentWorkflow classes should require proxy."""
        assert trainer_with_detection._requires_proxy_workflow(
            dummy_agent_workflow_class
        )

    def test_non_inheriting_agent_instance_requires_proxy(self, trainer_with_detection):
        """Non-inheriting agent instances should require proxy."""
        workflow = DummyAgentNonInheriting()
        assert trainer_with_detection._requires_proxy_workflow(workflow)

    def test_non_inheriting_agent_class_requires_proxy(self, trainer_with_detection):
        """Non-inheriting agent classes should require proxy."""
        assert trainer_with_detection._requires_proxy_workflow(DummyAgentNonInheriting)

    def test_invalid_string_requires_proxy(self, trainer_with_detection):
        """Invalid string paths should require proxy (fail-safe)."""
        workflow = "nonexistent.module.Workflow"
        assert trainer_with_detection._requires_proxy_workflow(workflow)

    def test_string_to_agent_workflow_requires_proxy(self, trainer_with_detection):
        """String paths to agent workflows should require proxy.

        Note: SimpleAgent is a duck-typed agent (no AgentWorkflow inheritance),
        demonstrating that any class with a compatible run() method works.
        """
        workflow = "tests.experimental.openai.utils.SimpleAgent"
        assert trainer_with_detection._requires_proxy_workflow(workflow)


class TestAgentWorkflowDeprecation:
    """Test that AgentWorkflow deprecation warning works correctly."""

    def test_deprecation_warning_on_instantiation(self):
        """Verify deprecation warning is triggered when instantiating AgentWorkflow subclass."""

        class MyAgent(AgentWorkflow):
            async def run(self, data, **extra_kwargs):
                return 1.0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _agent = MyAgent()  # noqa: F841
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "MyAgent inherits from deprecated AgentWorkflow" in str(w[0].message)

    def test_deprecation_warning_without_super_init(self):
        """Verify warning triggers even if subclass doesn't call super().__init__()."""

        class MyAgentNoSuper(AgentWorkflow):
            def __init__(self):
                # Intentionally not calling super().__init__()
                self.custom_attr = "test"

            async def run(self, data, **extra_kwargs):
                return 1.0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            agent = MyAgentNoSuper()
            # Warning should still be triggered via metaclass
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "MyAgentNoSuper inherits from deprecated AgentWorkflow" in str(
                w[0].message
            )
            # Verify the instance was created correctly
            assert agent.custom_attr == "test"
