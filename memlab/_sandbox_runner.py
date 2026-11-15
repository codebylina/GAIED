"""Child process of memlab.sandbox: reads {code, rows} on stdin, prints the test result as JSON."""
import builtins
import json
import math
import statistics
import sys

ALLOWED_MODULES = {"math", "statistics", "numpy", "pandas", "scipy", "scipy.stats", "itertools",
                   "collections", "re", "functools"}
SAFE_BUILTINS = ["abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance",
                 "len", "list", "map", "max", "min", "range", "round", "set", "sorted", "str", "sum",
                 "tuple", "zip", "Exception", "ValueError", "KeyError", "TypeError", "print", "frozenset"]


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] not in {m.split(".")[0] for m in ALLOWED_MODULES}:
        raise ImportError(f"module '{name}' is not allowed in tests")
    return builtins.__import__(name, globals, locals, fromlist, level)


def main() -> None:
    data = json.loads(sys.stdin.read())
    env_builtins = {n: getattr(builtins, n) for n in SAFE_BUILTINS}
    env_builtins["__import__"] = _import
    env = {"__builtins__": env_builtins, "math": math, "statistics": statistics}
    try:
        import pandas as pd
        env["pd"] = pd
        df = pd.DataFrame(data["rows"])
        if "id" in df.columns:
            df["id"] = df["id"].astype(str)
    except ImportError:
        df = data["rows"]
    try:
        import numpy as np
        env["np"] = np
    except ImportError:
        pass
    try:
        exec(compile(data["code"], "<judge-test>", "exec"), env)
        fn = data.get("fn", "test")
        if fn not in env:
            raise ValueError(f"the code must define a function {fn}(df)")
        out = env[fn](df)
        if fn != "test":
            out = {"result": out}
        elif not isinstance(out, dict):
            raise ValueError("test(df) must return a dict")

        def plain(v):
            if hasattr(v, "tolist"):
                v = v.tolist()
            if isinstance(v, (list, tuple, set)):
                return [plain(x) for x in v]
            if isinstance(v, dict):
                return {str(k): plain(x) for k, x in v.items()}
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            if hasattr(v, "item"):
                return v.item()
            return v
        print(json.dumps(plain(out), default=str))
    except Exception as exc:  # reported to the judge, which may rewrite its test
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))


if __name__ == "__main__":
    main()
