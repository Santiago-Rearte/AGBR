import os

if os.environ.get("BR_OUT"):
    try:
        import brcore

        brcore.start_child()
    except Exception as error:  # pragma: no cover
        import sys

        sys.stderr.write(f"brtrace sitecustomize failed: {error}\n")
