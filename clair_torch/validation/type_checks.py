import torch


def validate_all(iterable, cls, *, raise_error=True, allow_none_iterable=True, allow_none_elements=True, name="item") -> bool:
    """
    Generic utility function to validate that all items in a container are of the given type. Allows selecting whether
    to raise an error on failed checks or to return False.
    Args:
        iterable: The container to check.
        cls: the expected type to check for.
        raise_error: whether to raise an error on failure or return False on failure.
        allow_none_iterable: whether to allow iterable=None to pass the check. This does not check if the elements in
            it are None.
        allow_none_elements: whether to allow elements inside the iterable to be None.
        name: name to use for the error message.

    Returns:

    """
    if iterable is None:
        if allow_none_iterable:
            return True
        elif not allow_none_iterable and raise_error:
            raise TypeError(f"{name} cannot be None")
        else:
            return False

    if not hasattr(iterable, '__iter__'):
        raise TypeError(f"{name} must be iterable")

    invalid = []
    for item in iterable:
        if item is None:
            if not allow_none_elements:
                invalid.append(item)
        elif not isinstance(item, cls):
            invalid.append(item)

    if invalid:
        if raise_error:
            raise TypeError(f"Invalid {name}(s): {invalid}")
        return False

    return True


def is_broadcastable(shape_1: tuple[int, ...] | torch.Size, shape_2: tuple[int, ...] | torch.Size, *, raise_error=True)\
        -> bool:
    """
    Check that the shapes of two NumPy arrays or Torch tensors are broadcastable.
    Args:
        shape_1: shape of the first item.
        shape_2: shape of the second item.
        raise_error: whether to raise error on failed check, or return False.

    Returns:
        True for passed check, False for failed.
    """
    if not shape_1 or not shape_2:
        raise ValueError('Shapes cannot be empty')

    reversed_shape1 = list(shape_1)[::-1]
    reversed_shape2 = list(shape_2)[::-1]

    for a, b in zip(reversed_shape1, reversed_shape2):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            if raise_error:
                raise ValueError(f"Shapes {shape_1} and {shape_2} are not broadcastable.")
            return False
    return True


def validate_multiple_dimensions(list_of_tensors: list[torch.Tensor],
                                 list_of_expected_ndim: list[int | tuple[int, ...]], *,
                                 raise_error: bool = True) -> list[bool]:
    """
    Wrapper for validating the number of dimensions for multiple tensors in a single call. Relies on the
    validate_dimensions function, returning a list of bools depending on the results of the checks for each element.
    Allows raising an error on failed check.
    Args:
        list_of_tensors: list of tensors to check.
        list_of_expected_ndim: list of expected dimensions. Either length of one or equal length with list_of_tensors.
            With length of one, all tensors are expected to have that same dimensionality. Each element should be either
            a single int for allowing only a specific dimensionality, or tuple of ints for allowing multiple different
            dimensionalities.
        raise_error: Whether to return list of bools for the checks or raise error on a failed check.

    Returns:
        List of bools corresponding to the results of each tensor's dimensionality check.
    """
    evals = []

    if len(list_of_expected_ndim) == 1:
        list_of_expected_ndim = list_of_expected_ndim * len(list_of_tensors)

    for tensor, expected_ndim in zip(list_of_tensors, list_of_expected_ndim):

        evals.append(validate_dimensions(tensor, expected_ndim, raise_error=raise_error))

    return evals


def validate_dimensions(tensor: torch.Tensor, expected_ndim: int | tuple[int, ...], *, raise_error: bool = True) -> bool:
    """
    Validation function to check the number of dimensions in a tensor. Allows either raising an error or returning False
    on a failed check.
    Args:
        tensor: the tensor to check.
        expected_ndim: single int or tuple of ints if multiple dimensions are allowed.
        raise_error: whether to raise error or return False on failed check.

    Returns:
        True on passing check, False on failed check if errors are not raised.
    """
    if isinstance(expected_ndim, int):
        expected_ndim = (expected_ndim, )

    if tensor.ndim not in expected_ndim:
        if raise_error:
            raise ValueError(f"Expected tensor of {expected_ndim} dimensions, got {tensor.ndim}")
        else:
            return False

    return True
