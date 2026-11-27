"""Run with python -m unittest discover -s eval -p 'test_tool_schema.py'."""
import unittest
from tool_schema import check_schema


class ToolSchemaTest(unittest.TestCase):
    def test_supported_signatures(self):
        for tool, args in [
            ("list_files", {"directory": "."}),
            ("read_file", {"path": "a.py"}),
            ("search_files", {"pattern": "x", "directory": "."}),
            ("text_retrieval", {"query": "x"}),
            ("reject_none", {"reason": "write forbidden"}),
        ]:
            with self.subTest(tool=tool):
                self.assertTrue(check_schema(tool, args))
                self.assertFalse(check_schema(tool, {}))
                for key in args:
                    self.assertFalse(check_schema(tool, {**args, key: 42}))
                    self.assertFalse(check_schema(tool, {**args, key: None}))

    def test_read_line_bounds(self):
        for value in [1, 100]:
            self.assertTrue(check_schema("read_file", {"path": "a", "max_lines": value}))
        for value in [0, -1, 101, True, False, "10", 1.0, None]:
            with self.subTest(value=value):
                self.assertFalse(check_schema("read_file", {"path": "a", "max_lines": value}))

    def test_retrieval_count(self):
        self.assertTrue(check_schema("text_retrieval", {"query": "x", "top_k": 1}))
        for value in [0, -1, True, "5", 5.0, None]:
            self.assertFalse(check_schema("text_retrieval", {"query": "x", "top_k": value}))

    def test_invalid_call_shape(self):
        self.assertFalse(check_schema("unknown", {"path": "a"}))
        for args in [None, [], "path"]:
            self.assertFalse(check_schema("read_file", args))


if __name__ == "__main__":
    unittest.main()
