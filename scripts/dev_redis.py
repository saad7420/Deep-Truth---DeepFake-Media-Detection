"""Development-only Redis on 127.0.0.1:6379, for machines with no real Redis.

Pure Python (fakeredis). Keeps everything in memory, so the queue and the
result cache are lost when it stops; case records live in SQLite and survive.
Use a real Redis (Memurai, WSL, Docker) for anything beyond development.

    pip install "fakeredis[lua]"
    python scripts\\dev_redis.py
"""
from fakeredis import TcpFakeServer

server = TcpFakeServer(("127.0.0.1", 6379), server_type="redis")
print("dev redis listening on 127.0.0.1:6379 (Ctrl+C to stop)", flush=True)
server.serve_forever()
