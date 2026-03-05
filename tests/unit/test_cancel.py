"""Tests for cancel module."""

import asyncio

import pytest

from ai_lib_python.client import (
    CancelHandle,
    CancellableStream,
    CancelReason,
    CancelToken,
    create_cancel_pair,
)


class TestCancelToken:
    """Tests for CancelToken."""

    def test_initial_state(self) -> None:
        """Test initial token state."""
        token = CancelToken()
        assert token.is_cancelled is False
        assert token.reason is None

    def test_cancel(self) -> None:
        """Test cancellation."""
        token = CancelToken()
        result = token.cancel(CancelReason.USER_REQUEST)

        assert result is True
        assert token.is_cancelled is True
        assert token.reason == CancelReason.USER_REQUEST

    def test_cancel_twice(self) -> None:
        """Test cancelling twice returns False."""
        token = CancelToken()
        first = token.cancel()
        second = token.cancel()

        assert first is True
        assert second is False

    def test_cancel_with_metadata(self) -> None:
        """Test cancellation with metadata."""
        token = CancelToken()
        token.cancel(CancelReason.TIMEOUT, custom_key="custom_value")

        assert token.state.metadata["custom_key"] == "custom_value"

    @pytest.mark.asyncio
    async def test_wait(self) -> None:
        """Test waiting for cancellation."""
        token = CancelToken()

        async def cancel_later():
            await asyncio.sleep(0.01)
            token.cancel(CancelReason.USER_REQUEST)

        task = asyncio.create_task(cancel_later())
        reason = await token.wait()
        await task

        assert reason == CancelReason.USER_REQUEST

    @pytest.mark.asyncio
    async def test_wait_with_timeout(self) -> None:
        """Test wait with timeout."""
        token = CancelToken()

        # Should timeout
        result = await token.wait_with_timeout(0.01)
        assert result is False

        # Now cancel
        token.cancel()
        result = await token.wait_with_timeout(0.1)
        assert result is True

    def test_on_cancel_callback(self) -> None:
        """Test callback on cancel."""
        token = CancelToken()
        called = []

        def callback(reason):
            called.append(reason)

        token.on_cancel(callback)
        token.cancel(CancelReason.ERROR)

        assert len(called) == 1
        assert called[0] == CancelReason.ERROR

    def test_callback_called_immediately_if_cancelled(self) -> None:
        """Test callback called immediately if already cancelled."""
        token = CancelToken()
        token.cancel(CancelReason.SHUTDOWN)

        called = []
        token.on_cancel(lambda r: called.append(r))

        assert len(called) == 1

    def test_raise_if_cancelled(self) -> None:
        """Test raise_if_cancelled."""
        token = CancelToken()

        # Should not raise
        token.raise_if_cancelled()

        token.cancel()
        with pytest.raises(asyncio.CancelledError):
            token.raise_if_cancelled()

    def test_reset(self) -> None:
        """Test resetting token."""
        token = CancelToken()
        token.cancel()

        token.reset()

        assert token.is_cancelled is False
        assert token.reason is None


class TestCancelHandle:
    """Tests for CancelHandle."""

    def test_cancel_through_handle(self) -> None:
        """Test cancelling through handle."""
        token = CancelToken()
        handle = CancelHandle(token)

        handle.cancel(CancelReason.USER_REQUEST)

        assert token.is_cancelled is True
        assert handle.is_cancelled is True
        assert handle.reason == CancelReason.USER_REQUEST


class TestCreateCancelPair:
    """Tests for create_cancel_pair."""

    def test_create_pair(self) -> None:
        """Test creating cancel pair."""
        handle, token = create_cancel_pair()

        assert not handle.is_cancelled
        assert not token.is_cancelled

        handle.cancel()

        assert handle.is_cancelled
        assert token.is_cancelled

    def test_create_with_timeout(self) -> None:
        """Test creating pair with timeout."""
        # Just test creation doesn't raise
        handle, _token = create_cancel_pair(timeout=60.0)
        assert not handle.is_cancelled


class TestCancellableStream:
    """Tests for CancellableStream."""

    @pytest.mark.asyncio
    async def test_normal_iteration(self) -> None:
        """Test normal stream iteration."""
        token = CancelToken()

        async def source():
            for i in range(5):
                yield i

        stream = CancellableStream(source(), token)
        result = []
        async for item in stream:
            result.append(item)

        assert result == [0, 1, 2, 3, 4]
        assert stream.finished is True

    @pytest.mark.asyncio
    async def test_cancel_mid_stream(self) -> None:
        """Test cancellation mid-stream."""
        token = CancelToken()

        async def source():
            for i in range(10):
                yield i
                if i == 2:
                    token.cancel()

        stream = CancellableStream(source(), token)
        result = []
        async for item in stream:
            result.append(item)

        # Should stop after cancellation
        assert len(result) <= 4  # At most 0, 1, 2, 3

    @pytest.mark.asyncio
    async def test_on_cancel_callback(self) -> None:
        """Test on_cancel callback."""
        token = CancelToken()
        cleanup_called = []

        async def source():
            yield 1
            token.cancel()
            yield 2

        def cleanup():
            cleanup_called.append(True)

        stream = CancellableStream(source(), token, on_cancel=cleanup)
        async for _ in stream:
            pass

        assert len(cleanup_called) == 1

    @pytest.mark.asyncio
    async def test_started_property(self) -> None:
        """Test started property."""
        token = CancelToken()

        async def source():
            yield 1

        stream = CancellableStream(source(), token)
        assert stream.started is False

        async for _ in stream:
            break

        assert stream.started is True

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        """Test closing stream."""
        token = CancelToken()

        async def source():
            yield 1
            yield 2

        stream = CancellableStream(source(), token)
        await stream.close()

        assert stream.finished is True
