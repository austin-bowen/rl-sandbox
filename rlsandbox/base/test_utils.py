import unittest

from rlsandbox.base.utils import zip_require_same_len


class TestZipRequireSameLen(unittest.TestCase):
    iterable0 = [1, 2, 3]
    iterable1 = [4, 5, 6, 7]

    def test_returns_same_as_zip_when_iterables_are_same_len(self):
        self.assertSequenceEqual(
            list(zip(self.iterable0, self.iterable0)),
            list(zip_require_same_len(self.iterable0, self.iterable0))
        )

    def test_raises_value_error_when_iterables_are_not_same_len(self):
        with self.assertRaises(ValueError):
            list(zip_require_same_len(self.iterable0, self.iterable1))
