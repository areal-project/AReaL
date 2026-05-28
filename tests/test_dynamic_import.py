from unittest.mock import patch

import pytest

from areal.utils.dynamic_import import import_from_string


class TestImportFromString:
    def test_import_class(self):
        WorkflowClass = import_from_string("areal.api.workflow_api.RolloutWorkflow")
        assert WorkflowClass.__name__ == "RolloutWorkflow"
        assert isinstance(WorkflowClass, type)

    def test_import_function(self):
        func = import_from_string("areal.utils.data.concat_padded_tensors")
        assert callable(func)
        assert func.__name__ == "concat_padded_tensors"

    def test_invalid_module(self):
        with pytest.raises(ImportError, match="Failed to import module"):
            import_from_string("nonexistent.module.SomeClass")

    def test_invalid_attribute(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            import_from_string("areal.utils.data.nonexistent_function")

    def test_invalid_format_no_dot(self):
        with pytest.raises(ValueError, match="Invalid module path"):
            import_from_string("NoDotsHere")

    def test_empty_string(self):
        with pytest.raises(ValueError, match="Invalid module path"):
            import_from_string("")

    def test_none_input(self):
        with pytest.raises(ValueError, match="Invalid module path"):
            import_from_string(None)

    def test_non_string_input(self):
        """Test that non-string inputs raise ValueError"""
        with pytest.raises(ValueError, match="Invalid module path"):
            import_from_string(123)

        with pytest.raises(ValueError, match="Invalid module path"):
            import_from_string([])

        with pytest.raises(ValueError, match="Invalid module path"):
            import_from_string({})

    def test_multiple_dots_in_path(self):
        """Test module paths with multiple dots"""
        # Valid path with multiple dots
        obj = import_from_string("areal.utils.data.concat_padded_tensors")
        assert callable(obj)

        # Another valid nested path
        workflow_class = import_from_string(
            "areal.workflow.multi_turn.MultiTurnWorkflow"
        )
        assert isinstance(workflow_class, type)

    def test_import_with_circular_import(self):
        """Test handling of circular imports (should raise ImportError)"""
        # This test simulates a circular import scenario
        with patch("importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("circular import detected")

            with pytest.raises(ImportError, match="Failed to import module"):
                import_from_string("some.module.CircularClass")

    def test_import_module_with_side_effects(self):
        """Test that module import works even with side effects"""
        # Test importing a module that might have initialization code
        obj = import_from_string("areal.utils.logging.getLogger")
        assert callable(obj)
        assert obj.__name__ == "getLogger"

    def test_error_messages_are_descriptive(self):
        """Test that error messages provide helpful information"""
        # Test module not found error
        with pytest.raises(ImportError) as exc_info:
            import_from_string("nonexistent_module.SomeClass")

        error_msg = str(exc_info.value)
        assert "nonexistent_module" in error_msg
        assert "SomeClass" in error_msg

        # Test attribute not found error
        with pytest.raises(AttributeError) as exc_info:
            import_from_string("areal.utils.data.NonExistentFunction")

        error_msg = str(exc_info.value)
        assert "areal.utils.data" in error_msg
        assert "NonExistentFunction" in error_msg

    def test_import_builtin_function(self):
        """Test importing built-in functions from standard library modules"""
        # This should work for any valid Python module path
        func = import_from_string("os.path.join")
        assert callable(func)
        # Verify it's the actual function by calling with correct string arguments
        result = func("a", "b", "c")
        assert result == "a/b/c" or result == "a\\b\\c"
