import unittest

from rlsandbox.base.collections import CircularList


class TestCircularList(unittest.TestCase):
    def setUp(self):
        self.list = CircularList(max_len=3)

    def test_init(self):
        self.assertListEqual([], CircularList())
        self.assertListEqual([], CircularList(max_len=3))
        self.assertListEqual([0, 1, 2], CircularList(range(3), max_len=3))
        self.assertListEqual([6, 4, 5], CircularList(range(7), max_len=3))

    def test_append(self):
        l = self.list

        self.assertEqual(0, len(l))
        self.assertFalse(l.is_max_len)

        l.append(0)
        self.assertListEqual([0], l)
        self.assertEqual(1, len(l))
        self.assertFalse(l.is_max_len)

        l.append(1)
        self.assertListEqual([0, 1], l)
        self.assertEqual(2, len(l))
        self.assertFalse(l.is_max_len)

        l.append(2)
        self.assertListEqual([0, 1, 2], l)
        self.assertEqual(3, len(l))
        self.assertTrue(l.is_max_len)

        l.append(3)
        self.assertListEqual([3, 1, 2], l)
        self.assertEqual(3, len(l))
        self.assertTrue(l.is_max_len)

        l.append(4)
        self.assertListEqual([3, 4, 2], l)
        self.assertEqual(3, len(l))
        self.assertTrue(l.is_max_len)

        l.append(5)
        self.assertListEqual([3, 4, 5], l)
        self.assertEqual(3, len(l))
        self.assertTrue(l.is_max_len)

        l.append(6)
        self.assertListEqual([6, 4, 5], l)
        self.assertEqual(3, len(l))
        self.assertTrue(l.is_max_len)

    def test_extend(self):
        self.list.extend([0, 1, 2, 3, 4])
        self.assertListEqual([3, 4, 2], self.list)

    def test_clear(self):
        self.list.extend([0, 1, 2])
        self.assertListEqual([0, 1, 2], self.list)

        self.list.clear()
        self.assertListEqual([], self.list)

        self.list.extend([0, 1, 2, 3])
        self.assertListEqual([3, 1, 2], self.list)

    def test_copy(self):
        self.list.extend([0, 1, 2])

        list_copy = self.list.copy()

        self.assertListEqual(self.list, list_copy)
        self.assertEqual(self.list.max_len, list_copy.max_len)
        self.assertIsNot(list_copy, self.list)

        list_copy.extend([3, 4, 5, 6])
        self.assertListEqual([6, 4, 5], list_copy)
        self.assertListEqual([0, 1, 2], self.list)
