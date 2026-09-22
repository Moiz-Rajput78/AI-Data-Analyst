"""Restricted Python execution tool for the AI Data Analyst."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pandas as pd

from tools.dataset import (
    DEFAULT_DATASET_PATH,
    DatasetError,
    load_dataset,
)


class PythonExecutionError(RuntimeError):
    """Raised when restricted Python analysis cannot be executed."""


MAX_CODE_LENGTH = 5000


BLOCKED_NAMES = {
    "__import__",
    "open",
    "exec",
    "eval",
    "compile",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "input",
    "breakpoint",
    "help",
    "memoryview",
    "exit",
    "quit",
}


BLOCKED_ATTRIBUTES = {
    # Filesystem output
    "to_csv",
    "to_excel",
    "to_json",
    "to_pickle",
    "to_parquet",
    "to_feather",
    "to_hdf",
    "to_sql",
    "to_clipboard",

    # File/data loading
    "read_csv",
    "read_excel",
    "read_json",
    "read_pickle",
    "read_parquet",
    "read_feather",
    "read_hdf",
    "read_sql",

    # Serialization
    "dump",
    "dumps",
    "load",
    "loads",

    # Process / environment access
    "system",
    "popen",
    "spawn",
    "fork",
    "kill",
    "remove",
    "unlink",
    "rmdir",
    "mkdir",
    "makedirs",
    "rename",
    "replace",
    "chdir",
    "getcwd",
    "listdir",
    "walk",
}


BLOCKED_AST_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Raise,
    ast.Global,
    ast.Nonlocal,
    ast.Delete,
    ast.Yield,
    ast.YieldFrom,
    ast.Await,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def _validate_code(
    code: str,
) -> ast.Module:
    """
    Parse and validate model-generated analysis code.

    Only a restricted analytical subset of Python is permitted.
    """

    if not isinstance(
        code,
        str,
    ):
        raise PythonExecutionError(
            "Python code must be provided as a string."
        )

    code = code.strip()

    if not code:
        raise PythonExecutionError(
            "Python code cannot be empty."
        )

    if len(code) > MAX_CODE_LENGTH:
        raise PythonExecutionError(
            f"Python code cannot exceed {MAX_CODE_LENGTH} characters."
        )

    try:
        tree = ast.parse(
            code,
            mode="exec",
        )

    except SyntaxError as exc:
        raise PythonExecutionError(
            f"Invalid Python syntax: {exc}"
        ) from exc

    for node in ast.walk(
        tree
    ):
        if isinstance(
            node,
            BLOCKED_AST_NODES,
        ):
            raise PythonExecutionError(
                "The requested Python code contains "
                f"a blocked operation: {type(node).__name__}."
            )

        if isinstance(
            node,
            ast.Name,
        ):
            if node.id in BLOCKED_NAMES:
                raise PythonExecutionError(
                    f"Use of '{node.id}' is not allowed."
                )

            if (
                node.id.startswith("__")
                and node.id != "__result__"
            ):
                raise PythonExecutionError(
                    "Dunder names are not allowed."
                )

        if isinstance(
            node,
            ast.Attribute,
        ):
            attribute = node.attr

            if attribute.startswith("_"):
                raise PythonExecutionError(
                    f"Access to attribute '{attribute}' is not allowed."
                )

            if attribute in BLOCKED_ATTRIBUTES:
                raise PythonExecutionError(
                    f"Use of attribute '{attribute}' is not allowed."
                )

    return tree


def _capture_final_expression(
    tree: ast.Module,
) -> ast.Module:
    """
    Capture the final expression as __result__.

    Example:

        summary = df.groupby(...)
        summary

    becomes conceptually:

        summary = df.groupby(...)
        __result__ = summary
    """

    if not tree.body:
        return tree

    last_statement = tree.body[
        -1
    ]

    if isinstance(
        last_statement,
        ast.Expr,
    ):
        assignment = ast.Assign(
            targets=[
                ast.Name(
                    id="__result__",
                    ctx=ast.Store(),
                )
            ],
            value=last_statement.value,
        )

        ast.copy_location(
            assignment,
            last_statement,
        )

        tree.body[
            -1
        ] = assignment

    ast.fix_missing_locations(
        tree
    )

    return tree


def _safe_scalar(
    value: Any,
) -> Any:
    """Convert dataframe values into serializable Python values."""

    if value is None:
        return None

    try:
        if pd.isna(
            value
        ):
            return None

    except (
        TypeError,
        ValueError,
    ):
        pass

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    if isinstance(
        value,
        pd.Period,
    ):
        return str(
            value
        )

    if hasattr(
        value,
        "item",
    ):
        try:
            return value.item()

        except (
            AttributeError,
            ValueError,
        ):
            pass

    return value


def _serialize_dataframe(
    dataframe: pd.DataFrame,
    max_rows: int,
) -> dict[str, Any]:
    """Serialize a DataFrame into a compact tool result."""

    preview = dataframe.head(
        max_rows
    )

    rows: list[
        dict[str, Any]
    ] = []

    for index, row in (
        preview.iterrows()
    ):
        serialized_row: dict[
            str,
            Any,
        ] = {
            str(column): _safe_scalar(
                row[column]
            )
            for column in preview.columns
        }

        serialized_row[
            "__index__"
        ] = _safe_scalar(
            index
        )

        rows.append(
            serialized_row
        )

    return {
        "type": "dataframe",
        "rows_total": int(
            len(dataframe)
        ),
        "rows_returned": int(
            len(preview)
        ),
        "columns": [
            str(column)
            for column in dataframe.columns
        ],
        "data": rows,
    }


def _serialize_series(
    series: pd.Series,
    max_rows: int,
) -> dict[str, Any]:
    """Serialize a Pandas Series."""

    preview = series.head(
        max_rows
    )

    data: list[
        dict[str, Any]
    ] = []

    for index, value in (
        preview.items()
    ):
        data.append(
            {
                "index": _safe_scalar(
                    index
                ),
                "value": _safe_scalar(
                    value
                ),
            }
        )

    return {
        "type": "series",
        "name": (
            str(
                series.name
            )
            if series.name is not None
            else None
        ),
        "rows_total": int(
            len(series)
        ),
        "rows_returned": int(
            len(preview)
        ),
        "data": data,
    }


def _serialize_result(
    result: Any,
    max_rows: int,
) -> Any:
    """Convert analytical results into JSON-friendly structures."""

    if isinstance(
        result,
        pd.DataFrame,
    ):
        return _serialize_dataframe(
            result,
            max_rows,
        )

    if isinstance(
        result,
        pd.Series,
    ):
        return _serialize_series(
            result,
            max_rows,
        )

    if isinstance(
        result,
        pd.Index,
    ):
        values = [
            _safe_scalar(
                value
            )
            for value in result[
                :max_rows
            ]
        ]

        return {
            "type": "index",
            "values_total": int(
                len(result)
            ),
            "values": values,
        }

    if isinstance(
        result,
        dict,
    ):
        return {
            str(key): _serialize_result(
                value,
                max_rows,
            )
            for key, value
            in result.items()
        }

    if isinstance(
        result,
        (
            list,
            tuple,
            set,
        ),
    ):
        values = list(
            result
        )

        return [
            _serialize_result(
                value,
                max_rows,
            )
            for value
            in values[
                :max_rows
            ]
        ]

    return _safe_scalar(
        result
    )


def run_python(
    code: str,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    max_output_rows: int = 50,
) -> dict[str, Any]:
    """
    Execute restricted Python/Pandas analysis.

    Available objects:

        df
            A copy of the currently loaded dataset.

        pd
            Pandas, available for analytical operations.

    Imports are not required and are blocked.

    The environment does not expose unrestricted:

    - operating-system access
    - file access
    - imports
    - subprocesses
    - network calls
    - eval / exec
    - dataframe file writing
    """

    if (
        max_output_rows < 1
        or max_output_rows > 100
    ):
        raise PythonExecutionError(
            "max_output_rows must be between 1 and 100."
        )

    try:
        dataframe = load_dataset(
            dataset_path
        )

    except DatasetError as exc:
        raise PythonExecutionError(
            str(exc)
        ) from exc

    tree = _validate_code(
        code
    )

    tree = _capture_final_expression(
        tree
    )

    try:
        compiled_code = compile(
            tree,
            filename="<restricted-analysis>",
            mode="exec",
        )

    except Exception as exc:
        raise PythonExecutionError(
            f"Unable to compile analysis code: {exc}"
        ) from exc

    safe_globals: dict[
        str,
        Any,
    ] = {
        "__builtins__": SAFE_BUILTINS,
    }

    safe_locals: dict[
        str,
        Any,
    ] = {
        "df": dataframe.copy(),
        "pd": pd,
    }

    try:
        exec(
            compiled_code,
            safe_globals,
            safe_locals,
        )

    except Exception as exc:
        raise PythonExecutionError(
            "Python analysis failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if "__result__" not in safe_locals:
        return {
            "status": "success",
            "code": code,
            "result": None,
            "message": (
                "Code executed successfully but did not "
                "produce a final expression."
            ),
        }

    result = safe_locals[
        "__result__"
    ]

    return {
        "status": "success",
        "code": code,
        "result": _serialize_result(
            result,
            max_output_rows,
        ),
    }