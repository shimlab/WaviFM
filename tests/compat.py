"""
Compatibility layer for running tests when pytest is not installed.
"""

try:
    import pytest
except ImportError:
    class _RaisesContext:
        def __init__(self, expected_exc, match=None):
            self.expected_exc = expected_exc
            self.match = match

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                raise AssertionError(f"Expected exception {self.expected_exc} was not raised.")
            if not issubclass(exc_type, self.expected_exc):
                return False
            if self.match is not None and self.match not in str(exc_val):
                raise AssertionError(f"Pattern '{self.match}' not found in '{str(exc_val)}'")
            return True

    class _PytestFallback:
        def raises(self, expected_exc, *args, **kwargs):
            match = kwargs.get("match", None)
            return _RaisesContext(expected_exc, match=match)

    pytest = _PytestFallback()

__all__ = ["pytest"]
