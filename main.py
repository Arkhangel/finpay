import sys

from modes import REST_MODE
from modes.rest import run_server

if __name__ == "__main__":
    mode = (sys.argv[1] if len(sys.argv) > 1 else REST_MODE).lower()

    if mode == REST_MODE:
        run_server()
    else:
        print(f"Unknown mode: {mode!r}. Available: {REST_MODE}")
        sys.exit(1)
