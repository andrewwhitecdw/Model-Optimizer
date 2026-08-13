import ast
import inspect
from pathlib import Path

from modelopt.torch.export import unified_export_hf
from modelopt.torch.export.model_config import QUANTIZATION_NONE


def test_quantization_none_uses_equality_comparison():
    """String constants like QUANTIZATION_NONE must be compared with ==/!=.

    Identity comparison (is / is not) relies on CPython interning and can fail
    for non-interned string values. This regression test scans the export
    module for identity comparisons against QUANTIZATION_NONE.
    """
    source_path = Path(inspect.getfile(unified_export_hf))
    tree = ast.parse(source_path.read_text())

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and any(
            isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops
        ):
            for comparator in node.comparators:
                if isinstance(comparator, ast.Name) and comparator.id == "QUANTIZATION_NONE":
                    raise AssertionError(
                        f"{source_path}: QUANTIZATION_NONE compared with identity operator; "
