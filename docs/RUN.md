# Running DeepTruth (Windows)

Four processes, each in its own PowerShell window, started **in this order**.
Project root: `S:\deeptruth_pipeline`.

| # | Process | Folder | Command |
|---|---|---|---|
| 1 | Redis | project root | `python scripts\dev_redis.py` |
| 2 | API (port 8000) | `server` | `python -m uvicorn main:app --host 127.0.0.1 --port 8000` |
| 3 | Worker | `server` | `python -m celery -A app.queue.celery_app worker --loglevel=info --pool=solo --queues=analysis --without-gossip --without-mingle --hostname w1@%h` |
| 4 | Web console (port 3000) | `client` | `npm run dev` |

Redis must come first. The API and worker both fail to queue anything without it.
If you have a real Redis (Memurai, WSL, Docker) running on port 6379, skip step 1.

## Check it is up

- <http://127.0.0.1:8000/api/health> must show `"status":"ok"` and `"workers":1`.
  Allow about 10 seconds after the worker prints `ready`.
- <http://localhost:3000> should show **Engine online** in the top bar.

| Health says | Meaning |
|---|---|
| `degraded`, `redis: unavailable` | Redis (step 1) is not running |
| `idle`, `workers: 0` | The worker (step 3) is not running |
| connection refused | The API (step 2) is not running |

## After an interruption, reboot or crash

1. Close every old PowerShell window so no stale process holds a port.
2. If a start fails with **WinError 10013** or "address already in use", something
   still owns the port. Find it and stop it (replace `<PID>`):

   ```
   netstat -ano | findstr :8000
   taskkill /PID <PID> /F
   ```

   Do the same for `:6379` (Redis) and `:3000` (console).
3. Start the four processes again in order.

A job that was running when the worker died is **not** recovered on the dev Redis
(its queue lived in memory). Resubmit the file. With a real Redis the job is redelivered.

## Behaviour to expect

- A first analysis downloads nothing, but loading the models takes a while: an
  image takes about 2-3 minutes and a video about 8 minutes on a CPU-only machine.
- Uploading the same file again is answered instantly from the cache, unless the
  dev Redis was restarted (the cache is in memory).
- Run only **one** worker per machine unless you have the RAM: each loads its own
  copy of the models.

## Browser extension

1. Chrome -> `chrome://extensions` -> turn on **Developer mode**.
2. **Load unpacked** -> choose `S:\deeptruth_pipeline\extension`
   (click the reload icon after any rebuild).
3. Open the extension popup and set the server to `http://localhost:8000`.

After editing `extension\source\src`, rebuild with
`cd extension\source; npm run build`, then reload the extension.

## Stopping

Press **Ctrl+C** in each window. Stop the worker last if it is mid-job, or wait for the job to finish.

`scripts\start.ps1` can start everything in one go (`-Stop` stops it, `-Check` only
reports). It is not tested on this machine, so prefer the four commands above.
