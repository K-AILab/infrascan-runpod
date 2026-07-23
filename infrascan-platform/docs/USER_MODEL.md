# User & space model — sketch

Goal: a single capturer or space manager logs in, sees only the spaces they own,
captures + uploads new scans, runs inference (search / tagging) on them, and
optionally invites collaborators. Admins see and manage everything.

Kept concise on purpose — this is the v1 model, not the final one.

## Roles (one per user)

| role      | can do                                                              |
|-----------|---------------------------------------------------------------------|
| capturer  | upload scans, see own spaces, run search / tagging on own spaces    |
| manager   | everything a capturer does + invite collaborators to own spaces     |
| admin     | everything + see all spaces and users + reassign ownership          |

`manager` and `capturer` are intentionally close — most users will be both.
Splitting only matters at companies with one operator and one analyst.

## Entities

```
User            id, email, name, password_hash, role, created_at, last_seen_at
Space           id, slug, title, owner_id, created_at, updated_at, status
                  (status: uploading | processing | ready | failed)
Membership      space_id, user_id, role_in_space (viewer | editor)
                  (only used when sharing with collaborators)
Session         id, user_id, expires_at, last_seen_at  (cookie value)

NamedObject     id, space_id, name, slug, prompt_embedding (vector),
                anchor_proposal_id, anchor_view_id, anchor_bbox,
                created_by, created_at,
                cached_member_proposal_ids (recomputable from the index)
```

A NamedObject is a user-curated label for an embedding cluster. The
tagging pipeline only produces unlabeled embeddings — there are no class
names from the system. A user clicks an object in the viewer, gets the
top-K visually similar matches, and decides whether to save the query
under a name.

- `prompt_embedding` is the visual vector that defines the cluster.
- `anchor_*` fields point at the visual prompt the user clicked, so the
  thumbnail in search.html shows the user's original choice.
- `cached_member_proposal_ids` is a denormalized convenience; the
  authoritative member set is always "top-K nearest to prompt_embedding
  inside this space, filtered by confidence threshold."
- Names are scoped to one space by default; a future "lift to library"
  could promote them across all spaces an owner has.

The user never sees the specific model names behind these vectors. All
copy refers to "visual embeddings", "object proposals", "nearest match"
— never the specific architecture.

Notes:
- `slug` is the URL segment (e.g. `icc1`). Unique globally for now; we can
  scope to owner later if we need to.
- `owner_id` is the canonical permission anchor — owner can always edit,
  delete, reassign.
- `Membership` rows are only present when the owner has invited others.
  Empty by default. No "team" abstraction yet.

## URLs (clean and consistent)

```
/                        landing → recent spaces (logged-in) OR marketing (anon)
/login                   sign in
/spaces/                 list of spaces the user can see
/spaces/<slug>/          one space: 3D viewer + tagging + meta
/spaces/<slug>/search    text/click search inside the space
/upload/                 new scan capture / file drop
/admin/                  cross-user / cross-space console (admin only)
/account/                user settings
```

The current ad-hoc subdomains (`icc1-v2.infrascan-ai.com`, `w5-017...`, etc.)
fold into `/spaces/icc1`, `/spaces/w5_017` over time. The viewer per scan
stays the same code — only the URL prefix changes.

## Auth

- Cookie-based session at parent domain `.infrascan-ai.com` (works across
  every subdomain — gallery, tagging, viewers, upload).
- Initial: email + password, argon2 hash.
- Phase 2: Google OAuth (one-click for the capturers we already know).
- The existing `infrascan_auth = SHA-256("KAIL_infrascan")` gate stays as a
  fallback for old links during migration; we deprecate it once everyone has
  accounts.

## Inference per space

The tagging FastAPI server already loads a search index per space. Wiring this
up to permissions:

- All `/api/spaces` calls filter by what the session user can see.
- `/api/spaces/<slug>/query` requires the user be owner, member, or admin.
- The 360° walkthrough viewer (still served as static assets) gets a small
  permission check on its asset path — Cloudflare won't block it, the server
  will.

## Admin page (concise)

Two tables and four numbers across the top — `users`, `spaces`, `objects
indexed`, `in flight`. No dashboards, no graphs, no project management. Click
a user row → see/edit role + their spaces. Click a space row → see/edit
owner + memberships. That's it.

## What this v1 leaves out (on purpose)

- Teams / org accounts. Per-user only for now.
- Quotas, billing. None.
- Audit log. Out of scope; logs live in server stdout.
- Webhooks, API tokens for programmatic use. Add when asked.

If any of these become real needs they get a separate doc and a separate
phase. Right now: log in, see your spaces, search inside them.
