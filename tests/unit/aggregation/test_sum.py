import pytest

from torchjd.aggregation import Sum

from .utils import ExpectedShapeProperty, PermutationInvarianceProperty


@pytest.mark.parametrize("aggregator", [Sum()])
class TestSum(ExpectedShapeProperty, PermutationInvarianceProperty):
    pass


def test_representations():
    A = Sum()
    assert repr(A) == "Sum()"
    assert str(A) == "Sum"
