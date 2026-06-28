import sys

from modes import BOT_MODE, REST_MODE

if __name__ == "__main__":
    mode = (sys.argv[1] if len(sys.argv) > 1 else REST_MODE).lower()

    if mode == REST_MODE:
        from modes.rest import run_server
        run_server()
    elif mode == BOT_MODE:
        from modes.bot import run_bot
        run_bot()
    else:
        print(f"Unknown mode: {mode!r}. Available: {REST_MODE}, {BOT_MODE}")
        sys.exit(1)
