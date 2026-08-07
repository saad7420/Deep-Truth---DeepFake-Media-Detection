# Deep Truth — browser extension

Adds a **Verify** badge to images, video and audio on any page. Clicking it
uploads that media to your Deep Truth analysis server, opens a case, and shows
the verdict in place.

## What it talks to

The extension is a client of the FastAPI service in `server/`. Nothing else.

| Step | Call |
|------|------|
| Connection check | `GET  {apiUrl}/api/health` |
| Start a scan | `POST {apiUrl}/api/cases` — multipart: `title`, `media_type`, `file` |
| Await the verdict | `GET  {apiUrl}/api/cases/{caseId}` until `status` leaves `processing` |

`apiUrl` defaults to `http://localhost:8000` and is editable from the popup.
Give it the origin only — no `/api` suffix (the popup strips one if you paste
it anyway).

> **There is no hosted Deep Truth API.** Earlier builds defaulted to
> `https://api.deep-truth.ai`, which does not resolve, and shipped a
> "simulated mode" that was **on by default** and generated verdicts with
> `Math.random()`. Both are gone. If the server is unreachable the popup says
> so and no result is recorded — the extension never estimates.

## Install

```bash
cd source
npm install
npm run build      # emits popup.js, content.js, background.js, popup.css into ../
```

Then in Chrome: **chrome://extensions** → enable *Developer mode* → **Load
unpacked** → select the `extension/` directory (the parent of `source/`).

`npm run watch` rebuilds on save; reload the extension from
`chrome://extensions` to pick up changes to the service worker.

## Using it

1. Start the backend — `python main.py` from `server/`.
2. Open the popup. The status strip should read **Analysis server connected**.
   If not, fix that before scanning; the address is editable right there.
3. Browse to any page. Media larger than 120×120 gets a `⚡ Verify with Deep
   Truth` badge in its top-left corner.
4. Click it. The badge tracks the case: *Uploading* → *Analysing* → the
   verdict, coloured to match the console (green authentic, red manipulated,
   amber inconclusive).
5. Recent scans collect in the popup, each linking to its full case in the
   console.

## What it will not scan

Some media genuinely cannot be captured from the page, and the badge says which
rather than starting a scan that is bound to fail:

- **Streamed players** (YouTube and most video platforms). These use Media
  Source Extensions and expose only a `blob:` URL that is meaningless outside
  the page. Download the clip and upload it in the console instead.
- **Inline `data:` URLs.**
- **Formats the server rejects.** The accepted list mirrors
  `server/app/services/analyser.py::ALLOWED_TYPES` exactly, and the size cap
  mirrors `MAX_FILE_SIZE_MB` (500 MB by default).

## Permissions, and why each is needed

| Permission | Reason |
|---|---|
| `storage` | Settings and the recent-scan list |
| `alarms` | An MV3 service worker is suspended when idle; the alarm resumes polling for a case that was still running |
| `activeTab`, `scripting` | Content script injection |
| `<all_urls>` | The service worker downloads media from whatever CDN the page uses; without it, cross-origin fetches fail |
| `http://localhost/*`, `http://127.0.0.1/*` | Reaching the local backend |

If you run the backend somewhere other than localhost, add that origin to
`host_permissions` in `manifest.json`.

## Server-side requirement

The backend must allow the extension's origin. `server/app/core.py` matches
`chrome-extension://*` by pattern, because the ID changes on every unpacked
reload and cannot be listed in `.env`. Nothing to configure — but if you
replace the CORS block, keep that regex.

## Layout

```
extension/
├── manifest.json            MV3 manifest
├── popup.html
├── popup.js / content.js / background.js / popup.css   ← build output, do not edit
└── source/
    ├── esbuild.config.js
    └── src/
        ├── lib/api.ts       every network call to the backend
        ├── types/index.ts   mirrors server/app/models.py
        ├── background/      service worker: upload, poll, broadcast
        ├── content/         DOM scanning and the in-page badge
        └── popup/           settings, connection status, scan history
```
