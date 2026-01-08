import os
import sys
import traceback

# Ensure project root is on sys.path so importing top-level modules works
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flasgger.utils import parse_docstring  # noqa: E402

from app_v7 import app, swagger  # noqa: E402

for rule in app.url_map.iter_rules():
    endpoint = rule.endpoint
    view = app.view_functions[endpoint]
    methods = list(rule.methods - {"HEAD", "OPTIONS"})
    verb = methods[0] if methods else "GET"
    try:
        parse_docstring(view, swagger.sanitizer, endpoint=endpoint, verb=verb)
        print(endpoint, "OK")
    except Exception as e:
        print("FAILED", endpoint, type(e), e)
        traceback.print_exc()
        break
