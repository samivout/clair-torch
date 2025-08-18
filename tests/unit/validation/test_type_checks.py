import pytest

import torch

from clair_torch.validation import type_checks as tc


class TestIsBroadcastable:

    @pytest.mark.parametrize("shape_1, shape_2, expected", [
        ((3, 1), (1, 4), True),
        ((1, 3, 1), (4, 1, 5), True),
        ((5,), (1,), True),
        ((2, 3, 4), (3, 4), True),
        (torch.Size([2, 1, 4]), torch.Size([1, 3, 1]), True),

        ((3, 2), (3,), False),
        ((3, 2), (2, 2, 2), False),
        ((4,), (5,), False),
        ((2, 3), (3, 3), False),
    ])
    def test_is_broadcastable(self, shape_1, shape_2, expected):
        if expected:
            assert tc.is_broadcastable(shape_1, shape_2) is True
        else:
            assert tc.is_broadcastable(shape_1, shape_2, raise_error=False) is False
            with pytest.raises(ValueError, match="not broadcastable"):
                tc.is_broadcastable(shape_1, shape_2, raise_error=True)

    def test_empty_shape_raises(self):
        with pytest.raises(ValueError, match="Shapes cannot be empty"):
            tc.is_broadcastable((), (3, 2))

        with pytest.raises(ValueError, match="Shapes cannot be empty"):
            tc.is_broadcastable((3, 2), ())


class TestValidateAll:

    @pytest.mark.parametrize("items, cls, allow_none, expected", [
        ([1, 2, 3], int, True, True),
        (["a", "b"], str, True, True),
        ([], int, True, True),  # Empty iterable should pass
        (None, int, True, True),  # Should pass when None allowed
        (None, int, False, False),  # Should not pass when None not allowed
    ])
    def test_validate_all_pass(self, items, cls, allow_none, expected):
        assert tc.validate_all(items, cls, raise_error=False, allow_none_iterable=allow_none) == expected

    @pytest.mark.parametrize("items, cls", [
        ([1, "a", 3], int),
        (["a", None], str),
        ([object(), 3], str),
    ])
    def test_validate_all_invalid_elements_raises(self, items, cls):
        with pytest.raises(TypeError, match="Invalid item"):
            tc.validate_all(items, cls, raise_error=True, allow_none_elements=False)

    def test_none_with_raise(self):
        with pytest.raises(TypeError, match="cannot be None"):
            tc.validate_all(None, int, raise_error=True, allow_none_iterable=False)

    def test_non_iterable_input(self):
        with pytest.raises(TypeError, match="must be iterable"):
            tc.validate_all(42, int)

    def test_invalid_elements_returns_false(self):
        result = tc.validate_all([1, "a", 3], int, raise_error=False)
        assert result is False
