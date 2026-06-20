"""
Throttling reverse proxy for garak testing.

Listens on :8001, forwards to :8000 with a minimum interval between requests
so we stay within Grok free-tier rate limits (default: 20 RPM = 3s per request).

Usage:
    python eval/security/throttle_proxy.py [--rpm 20] [--backend http://localhost:8000]

Then point garak at http://localhost:8001/chat instead of :8000.
"""
from __future__ import annotations

import argparse
import asyncio
import time

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response

parser = argparse.ArgumentParser()
parser.add_argument("--rpm", type=int, default=20, help="Max requests per minute")
parser.add_argument("--backend", default="http://localhost:8000", help="Backend URL")
args, _ = parser.parse_known_args()

MIN_INTERVAL: float = 60.0 / args.rpm  # seconds between requests
BACKEND: str = args.backend.rstrip("/")

_lock = asyncio.Lock()
_last_sent: float = 0.0

app = FastAPI()


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy(request: Request, path: str) -> Response:
    global _last_sent

    async with _lock:
        now = time.monotonic()
        wait = MIN_INTERVAL - (now - _last_sent)
        if wait > 0:
            print(f"  [throttle] sleeping {wait:.1f}s …")
            await asyncio.sleep(wait)
        _last_sent = time.monotonic()

    body = await request.body()
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "transfer-encoding")
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.request(
            method=request.method,
            url=f"{BACKEND}/{path}",
            headers=headers,
            content=body,
            params=dict(request.query_params),
        )

    # 400 = blocked by security layer, no LLM tokens spent — reset timer so next request goes immediately
    if resp.status_code == 400:
        async with _lock:
            _last_sent = 0.0
        print(f"  [throttle] 400 blocked — timer reset")

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
        media_type=resp.headers.get("content-type"),
    )


if __name__ == "__main__":
    print(f"Throttle proxy  :8001 → {BACKEND}  ({args.rpm} RPM, {MIN_INTERVAL:.1f}s/req)")
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")
