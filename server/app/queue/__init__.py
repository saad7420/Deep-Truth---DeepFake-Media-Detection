"""Asynchronous task orchestration (Module 5).

Four pieces, deliberately kept separate so each can be reasoned about — and
failed — on its own:

    redis_client.py   one connection pool, plus the reachability check the
                      API uses to decide whether it can accept work at all
    state.py          the job lifecycle record: queued -> running -> done,
                      queue position, attempt count, and the pub/sub channel
                      the API streams to browsers
    cache.py          the hot-result cache: content-hash and source-URL keyed,
                      so a file (or page media) analysed once is never
                      analysed again
    tasks.py          the Celery task itself, with the retry policy

The API process never runs inference. It hashes the upload, asks the cache,
and either answers immediately or hands the job to Celery. Everything heavy
happens in a worker process, which is what lets several uploads progress at
once instead of serialising behind one FastAPI event loop.
"""
