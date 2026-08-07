# Deep Truth — Console (Frontend)

Next.js 16 console for the Deep Truth active deepfake defense system. Talks to the
FastAPI service in `server/`.

---

## Running it

```bash
# 1. Backend — from server/server/
python main.py                     # http://localhost:8000

# 2. Frontend — from client/
cp .env.local.example .env.local
npm install
npm run dev                        # http://localhost:3000
```

The server's `ALLOWED_ORIGINS` must include `http://localhost:3000` (it does by
default in `server/.env`).

---

## Routes

| Route | Purpose |
|---|---|
| `/` | Public landing page |
| `/login` | Operator profile (attribution, not authentication — see below) |
| `/dashboard` | Command Center — live vitals, verdict distribution, recent cases |
| `/forensics` | Upload evidence, watch the pipeline, read the verdict |
| `/cases` | Case History — search, filter by verdict/modality, paginate, delete |
| `/cases/[case_id]` | Forensic report — evidence preview, risk dial, modality breakdown, notes |
| `/cases/[case_id]/report` | Print-first document for PDF export |
| `/analytics` | Risk distribution, modality mix, volume over time |
| `/vault` | Stored evidence files with download and delete |
| `/admin` | Live system configuration (read-only — see below) |
| `/settings` | Operator profile and console preferences |
| `/docs` | Pipeline reference, API surface, stated limits |

---

## Bugs fixed from the previous build

1. **The case detail page never loaded.** It read `params.caseId` while the route
   segment is `[case_id]`, so the ID was always `undefined` and the fetch never ran.

2. **The data layer described a different backend.** `shared/routes.ts`, `lib/api.ts`
   and `hooks/use-cases.ts` assumed numeric IDs, a bare array from `GET /cases`, a
   JSON upload body, and a `POST /cases/{id}/process` endpoint. None of those exist.
   They also pointed at in-memory mock route handlers under `app/api/cases/` that
   shadowed the real service, so the UI was reading fixture data. Those handlers are
   deleted; everything now goes through `lib/api-client.ts`.

3. **The design system was never defined.** `font-display`, `grid-bg` and `neon-text`
   were used throughout but absent from `globals.css`, which was 8 lines with no
   tokens. Headings silently fell back to the body face.

4. **Case links were wrong.** They pointed at `/case/${c.id}` — wrong path, and the
   internal UUID rather than the public `CASE-XXXXXXXX` ID the server looks up on.

5. **Mobile navigation 404'd.** The desktop rail used `/Dashboard`, `/Forensics`,
   `/Vault`; the mobile drawer used lowercase. Next.js routes are case-sensitive.
   One nav definition now drives both.

6. **The sidebar rendered twice.** Pages rendered their own `<Sidebar />` *and* were
   wrapped in `AppShell`, which also rendered one.

7. **Seven unused shadcn components broke `tsc`** — importing uninstalled packages
   (`react-hook-form`, `input-otp`, `@radix-ui/react-menubar`, …) and using
   `@/lib/utils` instead of `@/app/lib/utils`. Removed; the rest of the library is intact.

---

## Architecture

```
app/
  lib/api-client.ts     Every backend call. Typed, validated, one place to change.
  shared/schema.ts      Zod schemas mirroring server/app/models.py + verdict tokens.
  hooks/use-cases.ts    React Query bindings; polls only while a case is processing.
  components/           AppShell, Navigation, CaseTable, MediaPreview, Primitives…
  globals.css           Design tokens, utilities, print styles, a11y floor.
```

**Verdict colours are defined once** in `shared/schema.ts` as `VERDICT`, so a status
can't render emerald on one screen and green on another. `RISK_BANDS` mirrors
`fusion_engine._verdict` (authentic ≤35, manipulated ≥65) — if you change the
thresholds server-side, change them there too.

**Polling stops when work stops.** `useCase` and `useCases` re-fetch every 2s only
while something is `processing`, then go quiet.

---

## Two deliberate constraints

**`/login` collects attribution, not credentials.** The FastAPI service mounts only
the cases and health routers — there is no auth layer, no token to exchange, nothing
to verify against. Rather than staging a password field that validates nothing, the
page says so and collects a name/email to stamp on uploads via the `user_id` form
field. When the backend grows real auth, replace the body of `signIn`/`signOut` in
`hooks/use-auth.ts` and every consumer keeps working.

**`/admin` reports state rather than changing it.** Fusion weights live in
`fusion_engine.DEFAULT_WEIGHTS`, thresholds in `_verdict()`, file limits in the
server's environment — none are exposed as endpoints. So the page shows live values
labelled with the file they're set in, instead of sliders that would silently do
nothing. It also reports which engines actually contributed weight to recent
fusions, read from the `weights_used` field the backend already writes.

The same principle removed the Secure Vault's old hard-coded key-rotation audit log
and its "immunize" button, which called nothing.

---

## Report export

"Export report" opens a print-first document and hands off to the browser's own
print-to-PDF. That keeps selectable text, working links and correct pagination — all
of which a client-side canvas capture loses — and needs no server-side PDF service.

The footer carries a SHA-256 digest over the fields shown. The report states plainly
that this detects alteration of the document and is **not** a signature attesting to
the provenance of the media.

---

## Verified

- `tsc --noEmit` — 0 errors
- `next build` — all 12 routes compile
- Contract probed against the running FastAPI app: create → poll → patch → delete,
  including the 404 after delete
- All four zod schemas parse real server payloads, and the three-tier analysis-row
  classification (checkpoints → modality → `Fusion (M9)`) resolves correctly

---

## Notes

Evidence is served from the backend's `/uploads` mount, so `next/image` optimisation
is deliberately bypassed in `MediaPreview` — the origin is arbitrary and not in
`next.config.ts`.

Deep Truth · Final Year Project 2023–2027 · COMSATS University Islamabad
