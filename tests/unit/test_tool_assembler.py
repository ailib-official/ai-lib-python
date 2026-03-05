"""Tests for ToolCallAssembler."""


from ai_lib_python.utils import (
    MultiToolCallAssembler,
    ToolCallAssembler,
    ToolCallFragment,
)


class TestToolCallFragment:
    """Tests for ToolCallFragment."""

    def test_basic_creation(self) -> None:
        """Test basic fragment creation."""
        fragment = ToolCallFragment(
            id="call_123",
            name="get_weather",
        )
        assert fragment.id == "call_123"
        assert fragment.name == "get_weather"
        assert fragment.arguments_buffer == ""


class TestToolCallAssembler:
    """Tests for ToolCallAssembler."""

    def test_on_started(self) -> None:
        """Test handling tool call started."""
        assembler = ToolCallAssembler()
        assembler.on_started("call_123", "get_weather")

        assert assembler.has_tool_calls()
        fragment = assembler.get_fragment("call_123")
        assert fragment is not None
        assert fragment.name == "get_weather"

    def test_on_partial(self) -> None:
        """Test handling partial arguments."""
        assembler = ToolCallAssembler()
        assembler.on_started("call_123", "get_weather")
        assembler.on_partial("call_123", '{"loc')
        assembler.on_partial("call_123", 'ation": "NYC"}')

        fragment = assembler.get_fragment("call_123")
        assert fragment is not None
        assert fragment.arguments_buffer == '{"location": "NYC"}'

    def test_on_partial_creates_fragment(self) -> None:
        """Test on_partial creates fragment if not exists."""
        assembler = ToolCallAssembler()
        assembler.on_partial("call_123", '{"key": "value"}')

        assert assembler.has_tool_calls()
        fragment = assembler.get_fragment("call_123")
        assert fragment is not None

    def test_on_name(self) -> None:
        """Test handling partial name."""
        assembler = ToolCallAssembler()
        assembler.on_started("call_123", "")
        assembler.on_name("call_123", "get_")
        assembler.on_name("call_123", "weather")

        fragment = assembler.get_fragment("call_123")
        assert fragment is not None
        assert fragment.name == "get_weather"

    def test_finalize_with_json(self) -> None:
        """Test finalize with valid JSON arguments."""
        assembler = ToolCallAssembler()
        assembler.on_started("call_123", "get_weather")
        assembler.on_partial("call_123", '{"location": "NYC", "unit": "celsius"}')

        tool_calls = assembler.finalize()

        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_123"
        assert tool_calls[0].function_name == "get_weather"
        assert tool_calls[0].arguments == {"location": "NYC", "unit": "celsius"}

    def test_finalize_with_invalid_json(self) -> None:
        """Test finalize with invalid JSON stores as raw."""
        assembler = ToolCallAssembler()
        assembler.on_started("call_123", "test_func")
        assembler.on_partial("call_123", "not valid json")

        tool_calls = assembler.finalize()

        assert len(tool_calls) == 1
        assert tool_calls[0].arguments == {}  # Empty dict for invalid JSON
        assert tool_calls[0].arguments_raw == "not valid json"

    def test_finalize_empty_arguments(self) -> None:
        """Test finalize with empty arguments."""
        assembler = ToolCallAssembler()
        assembler.on_started("call_123", "no_args_func")

        tool_calls = assembler.finalize()

        assert len(tool_calls) == 1
        assert tool_calls[0].arguments == {}

    def test_multiple_tool_calls(self) -> None:
        """Test multiple tool calls in order."""
        assembler = ToolCallAssembler()
        assembler.on_started("call_1", "func_a")
        assembler.on_started("call_2", "func_b")
        assembler.on_partial("call_1", '{"a": 1}')
        assembler.on_partial("call_2", '{"b": 2}')

        tool_calls = assembler.finalize()

        assert len(tool_calls) == 2
        assert tool_calls[0].function_name == "func_a"
        assert tool_calls[1].function_name == "func_b"

    def test_finalize_and_reset(self) -> None:
        """Test finalize_and_reset."""
        assembler = ToolCallAssembler()
        assembler.on_started("call_123", "test")

        tool_calls = assembler.finalize_and_reset()

        assert len(tool_calls) == 1
        assert not assembler.has_tool_calls()

    def test_reset(self) -> None:
        """Test reset."""
        assembler = ToolCallAssembler()
        assembler.on_started("call_123", "test")

        assembler.reset()

        assert not assembler.has_tool_calls()
        assert len(assembler) == 0

    def test_count(self) -> None:
        """Test count property."""
        assembler = ToolCallAssembler()
        assert assembler.count == 0

        assembler.on_started("call_1", "func")
        assembler.on_started("call_2", "func")

        assert assembler.count == 2

    def test_duplicate_started(self) -> None:
        """Test duplicate on_started updates name."""
        assembler = ToolCallAssembler()
        assembler.on_started("call_123", "")
        assembler.on_started("call_123", "new_name")

        fragment = assembler.get_fragment("call_123")
        assert fragment is not None
        assert fragment.name == "new_name"


class TestMultiToolCallAssembler:
    """Tests for MultiToolCallAssembler."""

    def test_multiple_turns(self) -> None:
        """Test assembling across multiple turns."""
        assembler = MultiToolCallAssembler()

        # Turn 1
        assembler.on_started("turn1", "call_1", "search")
        assembler.on_partial("turn1", "call_1", '{"query": "weather"}')

        # Turn 2
        assembler.on_started("turn2", "call_2", "calculate")
        assembler.on_partial("turn2", "call_2", '{"expr": "2+2"}')

        turn1_calls = assembler.finalize_turn("turn1")
        turn2_calls = assembler.finalize_turn("turn2")

        assert len(turn1_calls) == 1
        assert turn1_calls[0].function_name == "search"

        assert len(turn2_calls) == 1
        assert turn2_calls[0].function_name == "calculate"

    def test_finalize_all(self) -> None:
        """Test finalize_all."""
        assembler = MultiToolCallAssembler()

        assembler.on_started("t1", "c1", "func1")
        assembler.on_started("t2", "c2", "func2")

        all_calls = assembler.finalize_all()

        assert "t1" in all_calls
        assert "t2" in all_calls

    def test_reset_turn(self) -> None:
        """Test resetting a specific turn."""
        assembler = MultiToolCallAssembler()

        assembler.on_started("t1", "c1", "func")
        assembler.on_started("t2", "c2", "func")

        assembler.reset_turn("t1")

        assert "t1" not in assembler.turns
        assert "t2" in assembler.turns

    def test_reset_all(self) -> None:
        """Test resetting all turns."""
        assembler = MultiToolCallAssembler()

        assembler.on_started("t1", "c1", "func")
        assembler.on_started("t2", "c2", "func")

        assembler.reset()

        assert len(assembler) == 0

    def test_turns_property(self) -> None:
        """Test turns property."""
        assembler = MultiToolCallAssembler()

        assembler.on_started("turn_a", "c1", "func")
        assembler.on_started("turn_b", "c2", "func")

        turns = assembler.turns
        assert "turn_a" in turns
        assert "turn_b" in turns

    def test_finalize_empty_turn(self) -> None:
        """Test finalize on non-existent turn."""
        assembler = MultiToolCallAssembler()
        result = assembler.finalize_turn("nonexistent")
        assert result == []
