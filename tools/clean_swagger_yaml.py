import pathlib

SWAGGER_DIR = pathlib.Path("docs") / "swagger"
files = list(SWAGGER_DIR.glob("*.yml"))
for p in files:
    s = p.read_text(encoding="utf-8")
    lines = s.splitlines()
    # find index where a YAML mapping likely starts
    # (first line that begins with 'tags:' or 'responses:' or 'parameters:')
    start = 0
    for i, L in enumerate(lines):
        if (
            L.lstrip().startswith("tags:")
            or L.lstrip().startswith("responses:")
            or L.lstrip().startswith("parameters:")
        ):
            start = i
            break
    # drop potential trailing code fence lines (```)
    end = len(lines)
    while end > start and lines[end - 1].strip().startswith("```"):
        end -= 1
    new_lines = lines[start:end]
    new_text = "\n".join(new_lines) + "\n"
    p.write_text(new_text, encoding="utf-8")
    print("Cleaned", p)
