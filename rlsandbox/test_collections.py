import unittest
import random

from rlsandbox.collections import ReservoirSampledList


class ReservoirSampledListTest(unittest.TestCase):
    def test_avg_item_age_in_rsl_is_close_to_true_avg_item_age(self):
        rsl = ReservoirSampledList(max_len=100, rng=random.Random(42))
        rsl.extend(range(1000))

        avg_rsl_age = sum(rsl) / len(rsl)
        close_to_500 = 507.03
        self.assertAlmostEqual(avg_rsl_age, close_to_500)
