# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for sample error handling helpers."""

import unittest

from megatron.energon.errors import handle_restore_errors


class TestHandleRestoreErrors(unittest.TestCase):
    def test_restore_iterator_allows_consumer_close(self) -> None:
        """Closing a partially consumed restore iterator is not a restore error."""
        handled_errors = []

        def handle_error(error, sample, sources):
            handled_errors.append((error, sample, sources))

        def restore_iterator():
            with handle_restore_errors(handle_error, "sample"):
                yield 1

        iterator = restore_iterator()
        self.assertEqual(next(iterator), 1)
        iterator.close()

        self.assertEqual(handled_errors, [])


if __name__ == "__main__":
    unittest.main()
