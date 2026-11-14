"""Dependency-free validation for the tool signatures advertised in the dataset."""

def check_schema(tool: str, args: dict) -> bool:
    if not isinstance(args, dict):
        return False
    required = {
        "list_files": ("directory",),
        "read_file": ("path",),
        "search_files": ("pattern", "directory"),
        "text_retrieval": ("query",),
        "reject_none": ("reason",),
    }
    if tool not in required or not all(isinstance(args.get(k), str) for k in required[tool]):
        return False
    if tool == "read_file" and "max_lines" in args:
        value = args["max_lines"]
        if type(value) is not int or not 1 <= value <= 100:
            return False
    if tool == "text_retrieval" and "top_k" in args:
        value = args["top_k"]
        if type(value) is not int or value < 1:
            return False
    return True
