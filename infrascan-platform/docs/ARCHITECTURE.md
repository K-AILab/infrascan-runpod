# InfraScan platform architecture

Current state + proposed end state. Companion to `USER_MODEL.md` and `ERD.md`.

---

## TL;DR

```
APEX  infrascan-ai.com         marketing landing page · NOT served from this dev box
                                (different Cloudflare backend; out of scope here)

NEW    app.infrascan-ai.com    end-user platform — login, my spaces, search, upload, viewers
NEW    admin.infrascan-ai.com  internal — gallery of all experiments, users, ownership, logs
TEMP   design.infrascan-ai.com mockup tree for design iteration; retires once tokens land
                                everywhere

OLD    gallery.infrascan-ai.com   → folds into admin/
OLD    upload.infrascan-ai.com    → folds into app/upload
OLD    tagging.infrascan-ai.com   → folds into app/spaces/ (currently uvicorn already here)
OLD    icc1-v2.infrascan-ai.com   ┐
OLD    w5-019.infrascan-ai.com    ├ each becomes app.infrascan-ai.com/spaces/<slug>/viewer/
OLD    … 30+ per-scan subdomains  ┘
```

End state: **3 active subdomains on this dev box** (`app`, `admin`, optionally `design`) plus the apex landing the user already owns.

---

## Why so many subdomains exist today

This dev box originated as a "demo box" — every experiment, scan, comparison and variant got its own port + its own Cloudflare ingress rule + its own DNS record. There are now **33 subdomains** all tunneling to local ports 8000–8060, each backed by a `python -m http.server` window in tmux.

This was fine while the audience was 1-3 people who knew which subdomain meant what. It does not scale to "give it to a customer."

## Current shape (as of today)

```
                Browser  https://*.infrascan-ai.com
                     │
                     ▼
            ┌──────────────────┐  CNAME *.infrascan-ai.com →
            │  Cloudflare      │  f60a1af3-...cfargotunnel.com
            └────────┬─────────┘
                     │ Argo Tunnel (single tunnel UUID)
                     ▼
            ┌──────────────────┐  ~/.cloudflared/infrascan-config.yml
            │  cloudflared     │  33 ingress rules → 33 local ports
            │  (this dev box)  │  in tmux:infrascan-viewers:cf-tunnel
            └────────┬─────────┘
                     │
   ┌──────────────────────────────────────────────────────────────┐
   │ Static viewers (24) ·     Custom apps (3) ·    Pure static  │
   │ python -m http.server ·   FastAPI / uvicorn ·  one-off      │
   │ one process per scan ·    state, GPU optional  index.html   │
   │                                                              │
   │  e.g. :8030 icc1-v2 ·     :8050 tagging        :8080 gallery│
   │       :8037 w5-017 ·      :8014 welcome-q      :8060 design │
   │       :8005 shinhan-r1 ·  :8020 upload                      │
   │       ... 21 others                                          │
   └──────────────────────────────────────────────────────────────┘
                     │
                     │ rsync (no tunnel — outbound only)
                     ▼
            ┌──────────────────┐
            │ DGX boxes        │  dgx-kail, dgx-kangsan
            │ batch compute    │  selected by /infrascan-dev/scripts/dgx/select.sh
            │ no public ingress│  pipeline only, never serves traffic
            └──────────────────┘
```

## Where the FAISS query API lives

**On this dev box**, inside the uvicorn process listening on `localhost:8050`. The same FastAPI app serves the three icc spaces at `/icc1/`, `/icc2/`, `/icc3/`. Three search indexes (≈ 11 GB total) get mmap'd at startup.

- CPU search by default (`IndexFlatIP`). Query latency ≈ 50–150 ms.
- The local GPU is **only** touched at query time, to dedupe top-K hits with a feature matcher. Lazy-loaded; the server is idle on GPU until a click.

The DGX boxes do **not** serve queries. They only run the offline pipeline that builds the index, then `rsync` the index back to this dev box.

---

## Target shape

```
                Browser  https://app.infrascan-ai.com
                     │
                     ▼
            ┌──────────────────┐
            │  Cloudflare      │
            └────────┬─────────┘
                     │ Argo Tunnel (same tunnel UUID — reuse the wiring)
                     ▼
            ┌──────────────────┐  ingress: 3 rules instead of 33
            │  cloudflared     │  ├ app    → localhost:8050
            │  (this dev box)  │  ├ admin  → localhost:8051
            └────────┬─────────┘  └ design → localhost:8060 (optional)
                     │
   ┌──────────────────────────────────────────────────────────────────┐
   │  app.infrascan-ai.com  (FastAPI, uvicorn :8050)                  │
   │  ─ /                          login or my-spaces                │
   │  ─ /login                     auth                              │
   │  ─ /signup                    invite acceptance                 │
   │  ─ /spaces/                   the user's spaces                 │
   │  ─ /spaces/<slug>/            space detail                      │
   │  ─ /spaces/<slug>/viewer/     3D walk-through (was *-v2 subdmn) │
   │  ─ /search                    cross-space visual search          │
   │  ─ /upload                    capturer upload flow               │
   │  ─ /api/spaces                JSON listing                       │
   │  ─ /api/spaces/<slug>/query   visual search                      │
   │  ─ /api/spaces/<slug>/save    save a named object                │
   │  ─ /api/spaces/<slug>/upload  resumable upload                   │
   │                                                                 │
   │  Auth: session cookie at `.infrascan-ai.com`. SQLite at         │
   │  `/var/infrascan/db.sqlite` (users, spaces, memberships,         │
   │  named_objects). Embeddings + search index stay on disk.        │
   ├──────────────────────────────────────────────────────────────────┤
   │  admin.infrascan-ai.com  (FastAPI, uvicorn :8051 — separate)    │
   │  ─ /                          today's gallery (every experiment)│
   │  ─ /experiments/<slug>/       direct viewer for deprecated      │
   │                                per-scan demos                   │
   │  ─ /users                     manage users / roles              │
   │  ─ /spaces                    cross-user space ownership        │
   │  ─ /logs                      uvicorn + pipeline logs           │
   │  ─ /ml-experiments/           the infrascan-dev/ml-experiments  │
   │                                tree (count-anything, fastsam…)  │
   │                                                                 │
   │  Auth: same cookie, role check `admin`.                         │
   ├──────────────────────────────────────────────────────────────────┤
   │  design.infrascan-ai.com  (python -m http.server :8060)         │
   │  ─ static mockup tree — design-system/preview/                  │
   │  ─ retires once tokens land everywhere on app + admin           │
   └──────────────────────────────────────────────────────────────────┘
```

### Old per-scan subdomains — what happens to them

Three options, increasing aggressiveness:

1. **Keep as Cloudflare-level redirects.**
   - `icc1-v2.infrascan-ai.com` → `app.infrascan-ai.com/spaces/icc1/viewer/`
   - Old links don't break.
   - Cloudflare Page Rules — no origin change, no http.server processes.
   - **Recommended.**

2. **Retire silently.** Drop the ingress, drop the DNS. Old links 404.
3. **Hard delete.** Same as (2) plus remove the static files. Frees disk on the dev box.

Default plan: (1) for the 6 icc + w5 viewers people might still link to; (2) for the rest (visionaries, scan13–16, parking-lot, etc. — these were ephemeral experiment dumps).

---

## Migration phases

```
PHASE 0 — what's done   (✓ today's session)
  • Design-system tokens shipped at design.infrascan-ai.com
  • USER_MODEL.md + ERD.md drafted
  • tagging.infrascan-ai.com picker swapped to use shared tokens

PHASE 1 — collapse per-scan subdomains  (next, 1 day of work)
  • Inside the tagging FastAPI on :8050, mount every per-scan dir as
    /spaces/<slug>/viewer/ via app.mount(...).
  • Already done for icc1/2/3. Repeat for w5_017/018/019, shinhan-r*,
    welcome-*, etc.
  • Add Cloudflare Page Rules for the old subdomains → new paths.
  • Drop the static http.server windows in tmux.

PHASE 2 — auth + SQLite  (2 days)
  • Implement users + sessions per ERD.md.
  • Cookie at .infrascan-ai.com — works across every subdomain.
  • Migrate spaces.json into the `spaces` table; owner = creator user.
  • Replace the SHA-256 shared-cookie gate with real sessions.

PHASE 3 — admin subdomain split  (1 day)
  • Stand up a second uvicorn on :8051 (the admin process).
  • Move gallery + users + spaces management + logs in.
  • admin.infrascan-ai.com cf-tunnel rule added; DNS CNAME.

PHASE 4 — retire design subdomain  (when ready)
  • Once tokens are everywhere, design.infrascan-ai.com retires.
  • Mockups archive into the infrascan-dev repo only.
```

Each phase is additive. No phase requires bringing the platform down — old subdomains keep working until their replacement is verified.

---

## Where each component lives

```
Project root                                 What's in it
───────────────────────────────────────────────────────────────────────────
/home/chan/Desktop/3d-object-tagging/        intern's FastAPI app — the
                                              source of the future `app`
                                              server. Pipeline scripts +
                                              server + ui + shared/ tokens.

/home/chan/Desktop/3d-tagging-project/        canonical per-scan data tree
                                              (views/, depth/, frames/,
                                              pointcloud.ply, viewers).
                                              icc1/2/3 + w5_017/018/019.

/home/chan/Desktop/infrascan-dev/             this repo — dev tooling, design
                                              system, scripts, ml-experiments,
                                              session logs.

  design-system/tokens/                       single source of CSS truth
  design-system/preview/                      mockup tree (served by `design`)
  design-system/docs/                         USER_MODEL, ERD, ARCHITECTURE
  scripts/dgx/                                DGX host selector + registry
  ml-experiments/                             dated model probes (Locate-,
                                              Count-Anything, etc.)
  PROGRESS_LOG.md                             append-only hourly log

/home/chan/Desktop/infrascan/                 the original "demo box" tree:
                                              gallery index.html, every
                                              per-scan viewer dir, the
                                              cloudflared config, SESSION_LOG.
                                              This is what folds into `admin`
                                              in Phase 3.

~/.cloudflared/infrascan-config.yml           tunnel ingress — 33 rules →
                                              edited per phase
```

## Non-goals (for the v1 platform)

- Multi-tenant isolation (one company per database). Single shared DB for now; tenant separation later.
- Real-time collaboration on a space. Read-only viewers; one capturer per scan.
- Mobile-native capture flow. Browser only; mobile capture goes through Insta360 Studio → upload.
- Auto-scaling the query path. Single uvicorn on this dev box is fine for ~6 spaces × ~1M proposals.
