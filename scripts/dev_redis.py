from fakeredis import TcpFakeServer

server = TcpFakeServer(("127.0.0.1", 6379), server_type="redis")
print("dev redis listening on 127.0.0.1:6379 (Ctrl+C to stop)", flush=True)
server.serve_forever()
