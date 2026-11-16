# DSH Web GUI — Production-Readiness Remediation Plan

**Target:** `D:\deepseek-harness_2` @ `0a15e36` (tag `dsh-v0.1.6-alpha.1`), served at `http://127.0.0.1:3080`
**Findings source:** `DSH-GUI-Review-Report.md` (independently re-verified 2026-10-11)
**Goal:** close every valid finding and make the GUI safe to call production-ready for (a) npm distribution, (b) LAN/network exposure, (c) a release claim.

---

## Revision 2 (2026-10-11, execution) — B1 is a FALSE POSITIVE

B1 claimed `@deepseek-ai/dsh-client-store` is unpublishable because `lib/index.js` imports `zustand`/`immer` bare while they sit in `devDependencies`. Tested against the repo's own gates and stated design, it does not hold:

- `packages/client/store/README.md:35` documents the model: *"Web-shell build input — the static ESM retains third-party imports for Vite; independent consumers supply its development dependencies."*
- `packages/client/tsdown.client.ts:290` (staticLinked preset) deliberately marks every bare specifier external, and `scripts/package-dependency-policy.ts:106` classifies all `packages/client/*` as browser-build inputs.
- Moving `zustand`/`immer` to `dependencies` makes `pnpm run verify-package-dependencies` fail with 2 violations ("must be devDependencies-only; found dependencies"). Reproduced empirically, then the manifest was restored.
- Repo-wide, `dependencies: {}` + devDependencies is the enforced convention for client packages, not a `store`-only slip.

**Conclusion:** the recommended fix would *break the dependency gate*, and the "fresh install then import" path it tests is explicitly unsupported. B1 is closed as an expectation mismatch, not code. The real distribution gate is `pnpm run release:verify-packed-install`.

**Scope correction for H1:** DSH's own code contains **zero** Monaco references (grep over `packages/**`), so H1 crashes a third-party theme plugin (e.g. `web-file-editor`), not the stock GUI. It is still the top *real* interop defect.

---

## 0. Current state before you start

### 0.1 Already fixed in the uncommitted working tree (land these first)
These were edited today and are **not** part of the remaining work — land and test them as their own commit:

| Was | Fix already on disk |
|---|---|
| M5 `--host 0.0.0.0` contradiction | `packages/bundle/web-app/src/startup.ts` no longer rejects; `tests/startup.spec.ts` now asserts acceptance; `docs/subsystems/web-server.md`/`.zh.md`, `packages/bundle/web-app/README.md`, `packages/client/connection/README.md` reconciled; LAN plaintext-cookie warning added |
| §4 false positive | `DSH_DISABLE_BROWSER_AUTH` kill switch actually implemented (`browser-auth.ts`, `connection/src/index.ts`, `tests/browser-auth-disable.host.spec.ts`) with non-loopback refusal |
| LOW: HMR EventSource blind | `packages/client/hmr/src/client/index.ts` gained `onerror`/`readyState` + reconnect ladder |
| LOW: title injection breaks silently | `apps/web/vite.config.ts` throws on missing placeholder and uses a replacer function |
| LOW: `serve:preview` binds 0.0.0.0 | `apps/web/package.json` now binds `127.0.0.1` |

### 0.2 Repo hygiene (do before any commit)
- **Remove the stray `%SystemDrive%/` directory** — a literal unexpanded-env-var folder; it indicates a script wrote to an unexpanded path. Find the culprit script, delete the folder.
- **Split the unrelated dirty LLM edits** (`packages/llm/llm/src/index.ts`, `packages/llm/llm-pi-ai/src/*`) into a separate commit; they are not GUI review work.
- Decide commit boundaries: one commit per phase below; never bundle the token fix with header work.

### 0.3 Corrections to the review itself (fix the report, do not act on these)
- `DSH_DISABLE_BROWSER_AUTH` did **not** exist at review time; it was added today. Remove it from the report's §4 "keep" list (or mark "introduced post-review").
- **M3 is largely a misread.** `apps/web/tests/smoke-real.e2e.ts:666` genuinely performs real model rounds ("empty-state first send completes a real model round", credentials inherited from the environment). The mocks/fake keys (`keyless-web-*`) are in the *earlier* suites. So do not "un-gate" it — add a **keyless** boot→auth→reconnect sibling instead.
- Counts: 20 workflows (not 21); 45 `ui-*` / 53 `packages/client` manifests (not 58); ~1,399 tracked test files (not 528); 81 golden/expected files, no `snapshots/web` (not 230); 51 `^18.2.0` react manifests (not 46). `rpc-host.ts:250` reflects `Error: msg`, not a stack. Node still applies default HTTP timeouts.

---

## 1. Definition of done / release gates

| Gate | Passes when |
|---|---|
| **G1 Distribution** | Revised: `pnpm run release:verify-packed-install` installs every packed tarball and drives the shipped entry point. (The original B1 form was a misdiagnosis — Revision 2.) |
| **G2 Crash-free** | Boot + theme toggle + open file editor produce zero console/page errors in a browser test; the token contract test is green. |
| **G3 Network-safe** | Every response carries the security-header set; release dist has no `.map`; `.map` requests 404; SSE/WS/uploads are bounded and authenticated. |
| **G4 CI proves what ships** | Web browser lane runs on Linux **and** Windows (+ macOS); desktop app is built/packaged in CI; a keyless boot→auth→reconnect e2e runs on every PR. |
| **G5 Usable** | Modal focus trap, tooltip aria, landmarks, keyboard drag; docs match behavior. |
| **Regression guard** | §4 "production-grade" items keep their existing tests; no test deleted to make a gate pass. |

---

## Phase 1 — Distribution blocker (B1) · CLOSED (false positive, Revision 2)

No code change. Actions:
- Record the finding as a misdiagnosis so a later pass does not re-open it.
- Optional hardening: a spec asserting the documented contract (non-workspace runtime imports stay in `devDependencies`; the README states the build-input limitation). `verify-package-dependencies` already enforces the first half.
- Keep `pnpm run release:verify-packed-install` as the distribution gate.
- **Acceptance:** `pnpm run verify-package-dependencies` stays green with `store/package.json` unchanged (verified).

---

## Phase 2 — Theme/token crash (H1, H2) · ~1–2 days

**H1 — 3/4-digit hex reaches Monaco.** Root cause: `lightningcss` `transform({ minify: true })` in `packages/client/tsdown.client.ts` (~lines 547, 562) shortens `rgb(255,255,255)` → `#fff`; the alias `--dsw-alias-bg-base` then resolves to `#fff`, and Monaco's `ColorMap` regex `/^#?([0-9A-Fa-f]{6})([0-9A-Fa-f]{2})?$/` throws `Illegal value for token color: #fff` because the plugin's `editor.background`/`editor.foreground` become a synthetic token rule.
- **DONE (2026-10-11):** `packages/client/css-color-normalization.ts` (`expandHexColors`) is wired into all three CSS transform sites in `packages/client/tsdown.client.ts`; contract spec `scripts/css-color-normalization.client.spec.ts` (5 tests, green). A lightningcss visitor cannot express hex literals and `exclude: Features.Colors` does not stop color shortening — text post-processing over declaration values is the only workable seam.
- Alternative/additional: keep a JS-consumable token source of truth (normalized hex) and expose it from `ui-theme` so plugins need not read CSS at all.
- **Contract test:** new spec that loads the built `ui-theme` CSS, resolves every `--dsw-alias-*` (and any alias a JS consumer may read), and asserts each is a Monaco-legal color. Wire it into the client lane.
- **Verify:** browser test that calls `monaco.editor.defineTheme` with the resolved tokens and asserts no throw.

**H2 — dark probe returns light values.**

- **DONE (2026-10-11):** confirmed by reading both sides — the host scopes its dark palette with `body[data-ds-dark-theme]` (`design-platform.css:80,249`, `shiki.css:21`, `gradient-shadow-text.css:37`) and custom properties inherit downward, so the plugin's attribute on a child `<div>` could never match and the dark theme was always built from LIGHT tokens. `probeTokensFor` now toggles the attribute on the real `<body>`, reads `getComputedStyle(body)`, and restores it in `finally` within the same task (no paint, no transition). Live path confirmed: `exports["./client"] -> ./lib/client.js`, so the browser half was rebuilt with the plugin's own tsdown (`D:/my_deepseek_harness/deepseek_plugins/node_modules/tsdown`, v0.22.2); the rebuilt bundle contains the body-toggle implementation and no longer contains the old `probe.remove()` child probe.
- **Still worth doing upstream:** a supported both-schemes token API, so third-party themes never probe the DOM at all. `packages/client/ui-theme/src/index.ts` exports no browser-side token reader today.
 The probe is in the external plugin `...\.dsh\profiles\node_modules\web-file-editor\src\client\theme.ts` (`probeTokensFor` sets `data-ds-dark-theme` on a detached div, but the host scopes dark under `body[data-ds-dark-theme]`).
- Repo-owned durable fix: export a supported **both-schemes token API** from `ui-theme`/`themeBus` so third-party themes no longer probe the DOM.
- Interim: patch the profile-store copy (probe `document.body`, restore in `finally`) and upstream the fix; document that DOM probing is unsupported.
- **Acceptance:** G2; dark Monaco theme matches dark host tokens.

---

## Phase 3 — Security outer shell (H3, H4, H5, M6, M7 + LOW Secure) · ~4–6 days — H3/H5/M6/M7 CLOSED, H4 headers landed (CSP nonce deferred), LOW `Secure` REJECTED

**H4 — security headers + CSP (largest exposure).**

- **PART DONE (2026-10-11):** the five enforced headers are in `packages/host/webserver/src/index.ts` (`SECURITY_HEADERS` + `applySecurityHeaders`, called first in `handle()`). Reproduced live first: the running server answered `401` with only `cache-control/content-type/Vary/Date/Connection` — no security headers at all. Covered by a new case in `packages/host/webserver/tests/webserver.spec.ts` over the routed / overridden / empty-fallback-404 / SPA-fallback paths (5 tests green); `tsc -b tsconfig.host.json` clean.
- **Deviation — no `frame-ancestors`:** `packages/client/ui-sidebar-documentpreview/src/client/html/HtmlBody.tsx:40,56` frames a same-origin `blob:` document, and a local-scheme document inherits its creator's policy container, so an inherited `frame-ancestors 'none'` would refuse that frame. `X-Frame-Options: DENY` carries no such inheritance and covers the same attack. Do not add `frame-ancestors` without first proving the preview still mounts.
- **CSP still open (H4-CSP):** enforcing it needs a per-response nonce threaded from `handle()` into the fallback's index render and into every plugin `script`/`style` injection row (and `apps/web/index.html`). A guessed Report-Only policy would add console noise without closing the hole, so it is deferred as its own change rather than shipped half-built.
- Central seam: `packages/host/webserver/src/index.ts` `handle()` (≈line 221). `res.setHeader` before dispatch so route `writeHead` calls merge. Header set:
  `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, `X-Frame-Options: DENY`, `Cross-Origin-Opener-Policy: same-origin`, `Permissions-Policy: camera=(), microphone=(), geolocation=()`, and a `Content-Security-Policy`.
- CSP: `default-src 'self'; base-uri 'none'; object-src 'none'; frame-ancestors 'none'; form-action 'none'; img-src 'self' data: blob:; connect-src 'self' ws: wss:; style-src 'self' 'unsafe-inline'; script-src 'self' '<nonce>'`. Nonce strategy: generate per response, inject into `renderIndex`/`renderIndexInjections` output for the `global`/`script`/`style` rows and the inline theme-bootstrap in `apps/web/index.html`. Because raw `html` injection rows come from plugins, rewrite `<script>`/`<style>` to carry the nonce during render; if a third-party row cannot be guaranteed, ship the policy as `Content-Security-Policy-Report-Only` for one release and collect violations before enforcing.
- `packages/host/frontend-static/src/index.ts`: the 403 (:81) / 404 (:100) / 405 (:128) bare writes inherit the central headers automatically; also add `Cache-Control` — `public, max-age=31536000, immutable` for hashed `/assets/*`, `no-store` for index.
- **Verify:** extend `packages/host/webserver` tests + a live probe asserting the full header set on 200/401/403/404/405; add a CSP-violation assertion in the browser lane.

**H5 — stop shipping\serving source maps.**
- `apps/web/vite.config.ts`: make `sourcemap` conditional on a non-release profile/env flag (default off for release).
- `packages/client/tsdown.client.ts` (:274, :451): same gating.
- Serving: drop `.map` from the `frontend-static` MIME table and return 404 for `.map` in release builds; keep maps in local dev only.
- **Verify:** assert no `*.map` in release dist; live `.map` request → 404.
- **DONE (2026-10-11):** all three halves closed, using the repository's own profile marker rather than a new env var.
  1. The gate is `process.env.DSH_CLIENT_BUILD_PROFILE !== 'official'` at all three sites — `apps/web/vite.config.ts` and both `packages/client/tsdown.client.ts` configs (`:277`, `:455`). That marker is the one the build orchestrator already sets: `officialClientBuildEnvironment` puts it in `OFFICIAL_CLIENT_BUILD_ENVIRONMENT`, `scripts/build.ts` propagates it through `clientBuildProcessEnvironment` to the `build:lib`/`build:web` children, and `dev:web` goes through `repositoryClientBuildEnvironment`, which deliberately does **not** set it. So the release profile drops maps and the dev/local path keeps exactly the maps the shell needs to resolve a browser stack frame back to TSX.
  2. `frontend-static` no longer advertises `.map` in its MIME table and returns **404 for any `.map` target even when the file exists**, before touching the filesystem. This is the defense-in-depth half: `/assets/*` is deliberately public, so a map that a dev build left behind must not become a served spec of the RPC surface. The existing `/missing.js.map` miss case still passes, and a new fixture (`dist/app.js.map`) proves the refusal is policy rather than an absent file.
  3. Regression guard: a new case in `scripts/client-build-environment.client.spec.ts` asserts `sourcemap === false` under `DSH_CLIENT_BUILD_PROFILE=official` and `true` without it, for **both** the web Vite config and the tsdown browser config. It uses `vi.resetModules()` plus the file's existing const-variable import pattern to re-evaluate the Vite config under each profile — a literal `import('...?query')` is both a TS2307 and an "unknown variable dynamic import" in Vite.
  4. Checked before gating: the `dsh-tsc-sourcemap` plugin returns early when `<id>.map` is absent (`tsdown.client.ts:644`), so `sourcemap: false` cannot break the official build; and `scripts/client-bundle-purity.spec.ts:356` (`configs[0]?.sourcemap === true`) still passes because the test process carries no profile marker.
- **Remaining verification (Phase 8):** a real `pnpm run build --profile official` plus a scan for `*.map` under `apps/web/dist` and `packages/*/*/lib` — the config decision is proven, the emitted bytes are not yet.
- **Tests:** `frontend-static` 1/1, `client-bundle-purity` 39/39, `client-build-environment.client` 9/9 green; `tsc -b tsconfig.client.json` and `tsconfig.host.json` both clean.

**H3 — secret at rest / LAN posture.**
- Keep the documented local-trust posture for loopback; for non-loopback `--host`, wrap the `client-connection/browser-session` secret with DPAPI (Windows) / Keychain (macOS) / `libsecret` (Linux), or use a per-process secret with re-mint on restart.
- Set restrictive ACLs on `$DSH_HOME/.credentials.yaml` (0600 / user-only ACL).
- Operator docs: state plainly that read access to `DSH_HOME` == full GUI auth, and that LAN plaintext cookies make the fence moot on untrusted networks.
- **LOW:** add `Secure` to the cookie when the bind is non-loopback or behind HTTPS (`browser-auth.ts` cookie builder ≈:125).
- **RESOLVED / MOSTLY REFUTED (2026-10-11) — no code change, and the LOW must be *rejected*:**
  1. **The ACL half is already implemented, and stronger than the finding assumes.** `credentials-local` writes every document with `mode: 0o600` and `dirMode: 0o700` (five call sites: `src/index.ts:699,720,778,848`) and `assertOwnerOnly` (`:127-146`) stats the file and **refuses to load it** when any group/other bit is set, with the exact remedy in the message (`run "chmod 600 <file>"`). Windows is skipped deliberately and the reason is in the code comment (`:121-123`: no POSIX mode to inspect, rather than a faked check). The package README states the same contract (`:121`, `:139`).
  2. **The operator-docs half is already written, in three places.** `docs/subsystems/web-server.md:47` states the all-interfaces posture, the plaintext LAN cookie, and "restrict that posture to networks you trust"; `packages/client/connection/README.md:37` names the exact record and its file, and says the cookie "deliberately omit[s] `Secure` because the shipped server uses loopback HTTP"; `:39` repeats the LAN warning and `:41` documents that `DSH_DISABLE_BROWSER_AUTH` is loopback-only and jointly refused with a non-loopback bind. So "state it plainly" is already satisfied; the consequence the finding wants named — that file readability *is* the auth gate — follows from the three statements above.
  3. **The LOW `Secure` flag must not be implemented as written.** DSH's carrier owns no TLS (`webserver` config has no TLS surface; the only binds are `127.0.0.1` and `0.0.0.0`), and a browser drops a `Secure` cookie delivered over `http://`. Setting it on a non-loopback bind would therefore **delete the operator's own session and lock them out** of the LAN GUI — the opposite of the intended hardening. It only becomes correct once DSH terminates TLS itself or a documented TLS-terminating proxy makes the origin HTTPS; the trust fence must also learn to trust that proxy's `x-forwarded-proto`, which is a design change, not a one-line edit. Declined, with the reasoning recorded here so the next reviewer does not "fix" it.
  4. **Genuine residual (scoped, not closed):** an OS-keychain/DPAPI wrapper for the `client-connection/browser-session` record is not implemented. It is the correct answer *before* any non-loopback deployment, and the plan's own alternative — a per-process secret with re-mint on restart — would trade LAN convenience for it. Trigger: the first supported non-loopback production deployment. Note that wrapping this one record would not protect the API keys in the same file, so the real fix is a keychain-backed credentials provider, which is a provider-level feature with three platform implementations and no way to verify macOS/Linux keychain paths from this Windows box. Loopback remains the default and only supported production posture for 0.1.6.

**M6 — unauthenticated, unbounded channels.**
- Mount `/plugins/events` (`packages/client/hmr/src/index.ts:159-200`) and `/plugins` combo (`packages/client/modules/src/index.ts:540`) behind the connection auth check — `ctx.connection.isAuthenticated(req)` exists (`rpc.ts:192`, `browser-auth.ts:324`); return 401/403 otherwise.
- Cap the SSE `Set` (max connections + max total), add an idle timeout / periodic keep-alive with forced close, and clean up on shutdown (already partially there).

- **DONE (2026-10-11):** both routes now call `ctx.get('connection')?.requestRejection(req)` and answer 401/403 before doing any work — the real API is `requestRejection` (Host/Origin fence **plus** cookie), not the plan's `isAuthenticated`. A composition without the connection service (unit harnesses, the shell carrier's `fetchBundle`) has no auth concept and keeps serving, so existing tests are untouched. HMR additionally caps the channel: `MAX_EVENT_STREAMS = 16` (17th open → 503 + `retry-after`), `MAX_EVENT_BUFFER_BYTES = 1 MiB` per stream (a stalled reader is destroyed, never buffered without bound), and a 30 s unref'd keep-alive that doubles as the reap window for half-open sockets; dispose clears the interval and destroys every live stream. New cases: `hmr/tests/node-half.client.spec.ts` (401 + 16×200 + 503 + dispose-closes) and `modules/tests/node-half.client.spec.ts` (401/403 fence, plus the admitted path). 94 tests green across the two packages.
- **Considered and declined:** `packages/host/frontend-static/src/index.ts:88-95` authorizes only the index path; `/assets/*` stays public. That is a deliberate, tested contract (`frontend-static.spec.ts:125` asserts an unauthenticated `200` for `/app.js`), and the boot payload itself is injected into the gated index. Source maps and the live module graph live under `/plugins`, which is now fenced. Revisit only if the asset graph is ever considered sensitive.

**M7 — transport limits.**
- `packages/api/gateway/src/stream-server.ts:26`: set explicit `maxPayload`; cap streams per connection and globally.
- `packages/client/connection/src/http-bridge.ts:14`: replace the 300 MiB in-memory buffer for uploads with a streaming path or a much lower configurable ceiling; keep the image aggregate cap honest.
- `packages/host/webserver/src/index.ts:294`: set explicit `requestTimeout`/`headersTimeout`/`keepAliveTimeout` (and document that Node defaults were previously relied on).
- `packages/client/file-upload/src/http-route.ts`: enforce a byte ceiling.
- **DONE (2026-10-11):** all four items closed, and two of the review's premises turned out to be wrong.
  1. `stream-server.ts` now pins `maxPayload: 16 MiB` (ws's own default is 100 MiB) and caps logical streams at `MAX_STREAMS_PER_CONNECTION = 8` and `MAX_STREAMS_GLOBAL = 64`. An over-budget open is answered with a per-stream `error` frame using the existing failure mapper, **not** by failing the physical socket, so sibling streams keep running; only an undeliverable refusal closes the carrier (1011). The global permit counter is a small `StreamBudget` handed to every accepted connection, released exactly once when the pump settles. Tests: over-budget refusal leaves the socket OPEN and reports the per-connection message; a forced global-exhaustion path reports the global message; an undeliverable refusal closes with 1011; and a well-formed open frame padded past the ceiling never reaches the opener (ws reports 1006/1009 — the connection's error handler terminates rather than answering).
  2. `packages/host/webserver/src/index.ts` pins `headersTimeout = 60 s`, `requestTimeout = 300 s`, `keepAliveTimeout = 5 s` in the listen path with a comment. These equal today's Node defaults, so the change is about *pinning* them (a future runtime cannot silently widen the slow-loris window) rather than about a behaviour change — the test asserts the pinned values on the live server.
  3. **Wrong premise — uploads were never buffered.** `file-upload` registers its route with `requestBody: 'streaming' (`src/index.ts:76`), so `http-bridge` streams it and the 300 MiB cap never applies. The cap is the *JSON RPC envelope* ceiling, and it is honest: `assertImageBodyCapacity` (`client/connection/src/index.ts:54-66`) validates at load that it covers `ceil(maxMessageImageBytes × 4/3) + 1 MiB`, and throws otherwise. No bridge change was made; the residual (concurrent buffered envelopes are bounded only by the cap) is a concurrency, not a cap, problem and is recorded as residual.
  4. **Real gap found — the streaming upload route had no ceiling at all**, so an authenticated client could stream to disk until the device filled. `http-route.ts` now enforces `MAX_UPLOAD_BYTES = 1 GiB` while the body streams (`boundedUploadBytes`): it counts bytes, `reader.cancel()`s the body and throws on overflow, and the route answers `413` instead of the 200 JSON envelope. A `Content-Length` fast path was written and then **removed**: the Fetch carrier strips that header (undici treats it as forbidden), so it was dead code in production as well as untestable — the streaming counter is the only correct guard and covers chunked bodies too.
  5. **Residual (deliberately not closed):** concurrent `/api` requests may each hold a full 300 MiB buffered envelope resident. A process-wide cap on concurrent buffered bodies would bound it, but the buffer's lifetime extends into the handler, so a correct bound needs the permit released at handler completion; a naive read-window permit would only look bounded. Recorded for a future round rather than shipped as a misleading guard.
- **Tests:** `packages/api/gateway/tests/stream-server.host.spec.ts` 13/13, `packages/host/webserver/tests/webserver.spec.ts` 6/6, `packages/client/file-upload/tests/file-upload-http.host.spec.ts` 5/5; `tsc -b tsconfig.host.json` and `tsc -b tsconfig.client.json` both clean.
- **Acceptance:** G3.

---

## Phase 4 — Robustness & correctness (M8, M11, residual M7) · ~2–3 days — M8/M11 CLOSED, M7 residual open (deliberate)

**M8 — persistence hot path.** `packages/client/store/src/index.ts:146-166`: debounce/write-behind the `localStorage.setItem`, and on the **first** failure disable persistence (set a flag) instead of logging on every update — make the code honor its `:138-145` doc contract.
- **DONE (2026-10-11):** the contract violation is fixed; one half of the suggestion is deliberately implemented differently.
  1. **The real bug was that the code did not do what its own doc said.** `:138-145` promised "storage failures only disable persistence" while the subscriber logged a failure on **every** update and retried forever. Now the first failed write sets a `disabled` flag, is reported once, and later updates return before doing the `JSON.stringify` work at all.
  2. **Read/parse failures are treated differently on purpose:** they are logged once and persistence stays **enabled**, so the next update overwrites a corrupt snapshot instead of freezing it. Disabling persistence on a `JSON.parse` failure would have made a corrupt value permanent.
  3. **Deviation — writes stay synchronous; the guard is value-coalescing, not a debounce.** Deferring `setItem` to a timer/microtask would break the same-task rehydration contract the existing tests encode (`:126-137` and `:139-151` create a *second* store in the same task and expect the value just written), and would lose the final update if the tab closed before the flush. The shipped change instead skips a write when the serialized text equals the last successful write, which removes the redundant-write amplification without moving durability.
  4. **Semantics discovered while testing (worth knowing):** the engine only notifies on a real change — zustand's `setState` short-circuits on `Object.is`, and immer's `produce` returns the *same* reference for a no-op mutation. So the equality guard only fires when the object identity changed but the JSON did not (e.g. `set()` with a structurally equal value, or the raf-flush path). A first draft of the test used `init()` state and mutated `n = 1` — a no-op — so it passed while never entering the guard; both new tests now make real changes and were validated against the raw v8 counts.
  5. **Coverage mechanics:** the early returns are written with braces because a single-line `if (x) return` is remapped by the v8 reporter as an **untaken** branch and would fail the 100% gate even though it executes.
- **Tests:** `packages/client/store/tests/store.client.spec.ts` 22/22, with two new cases — "disables persistence after the first failed write instead of retrying every update" (two real updates, `setItem` called once, one report, `getSnapshot` still current) and "skips a redundant write when the serialized state is unchanged". Coverage on the file is **100% branches**; its only uncovered statement is the pre-existing `:55` subscriber-failure log, which `packages/client/ui-settings/tests/settings-scope.client.spec.ts:314-320` covers in the global run. Adjacent lanes green: store + ui-settings + ui-session 520/520 across 29 files; `tsc -b tsconfig.client.json` exits 0.
- **Acceptance:** G2, G3.

**M11 —**
- Webhook: add delivery throttling + a replay window (validate signature timestamp/nonce) in `packages/webhook/webhook-github/src/index.ts:49-61`.
- Electron: add `setPermissionRequestHandler` (deny by default) and `will-attach-webview` guards in `apps/desktop/src/main.ts` (context isolation/sandbox/webSecurity already correct).
- `frontend-static`: add `realpath` containment so an in-dist symlink cannot escape the dist root (current check at :76-84 is lexical only).
- **DONE (2026-10-11): all three items closed; one half of the webhook suggestion does not apply to GitHub.**
  1. **Webhook — GitHub sends no signed timestamp, so "validate signature timestamp" is inapplicable.** GitHub signs the raw body with HMAC-SHA256 and ships **no** timestamp; its replay handle is the `X-GitHub-Delivery` UUID. The replay window is therefore keyed on that ID: a repeat inside a 30-minute window is answered `202` (so GitHub stops retrying it) **without** a second dispatch. Replay state is written only after the signature verified **and** the dispatch succeeded, so an unauthenticated sender cannot poison the window and a retry after a `503` is still processed. It is a bounded insertion-ordered `Map` (4096 IDs, oldest evicted), so memory cannot grow with traffic. Admission control caps concurrent deliveries at 64 and sheds the excess with `429` + `Retry-After: 5` — backpressure rather than a dropped event, because GitHub retries a non-2xx later.
  2. **`frontend-static` symlink containment.** The lexical check could not see a link living *inside* the dist that points outside it, and `readFile` follows links. `serveStatic` now re-checks containment on the `realpath` of both the target and the root — the root too, deliberately: a dist reached through a symlinked parent (macOS `/tmp` → `/private/tmp`, a junctioned build directory, a linked checkout) must not be refused as an escape. The refusal is `403` and sits in the non-index, non-map branch only, so the source-map policy keeps its `404` and never confirms whether a target exists.
  3. **Electron.** `createWindow` now installs both `setPermissionRequestHandler` and `setPermissionCheckHandler` — fail closed, with the two clipboard permissions as the only grants so the GUI's copy buttons keep working — plus a `will-attach-webview` handler that strips `preload`, forces `nodeIntegration: false`/`contextIsolation: true`, and refuses any `src` outside `dsh-app://`. `webviewTag` remains off, so the webview guard is defense in depth.
  4. **Verification.** Webhook 37/37 with **100% statements/branches/functions/lines** across all three sources, including a deterministic concurrency test that holds one body open behind a gate (no timers, no races) and the capacity/expiry arms of the replay window. `frontend-static` 1/1 at 100% file coverage with the escaping link (`403`), the in-dist alias (`200`), and the link-to-root (`404`) all asserted; the trio skips itself only when the host cannot create symlinks at all. Desktop `tsc -b` clean, `tsc -b tsconfig.host.json` clean, oxlint clean. The new `429` and the replay semantics are documented in **both** `packages/webhook/webhook-github/README.md` and `README.zh.md`, with `README.i18n.yaml` re-recorded and re-verified.
  5. **Honest gap:** `apps/desktop` still has **no test lane at all**, so the Electron guards rest on typecheck plus review. That is what Phase 6's desktop CI item fixes; recorded rather than implied.
- **Acceptance:** G3, G2.

**LOW sweep:** sanitize `rpc-host.ts:250` (do not reflect raw internal error text; map to a stable code + log the detail server-side); add bodies/`content-type` to 401/403/404/405; launch-token rotation/single-use/attempt backoff; locale missing-key logging; `rafBatch` hidden-tab flush.

---

## Phase 5 — Accessibility & UX (M9, M10) · ~2–3 days

- **M9 Modal** (`packages/client/ui-primitives/src/Modal.tsx:41-60`): move focus into the dialog on open, trap Tab/Shift-Tab, restore focus on close, and mark background inert; add tests in the primitives lane.
- **M10:**
  - Tooltip (`Tooltip.tsx:139-158`): stable `id` + `aria-describedby` on the trigger.
  - `AppFrame.tsx:105-117` drag handle: `role="separator"`, `tabIndex`, `aria-valuenow`, arrow-key resize.
  - `Menu.module.css:98-162`: add a `:focus-visible` style.
  - Shell landmarks: add `header/nav/main/aside/footer` (currently zero).
  - Chat transcript: decide between virtualizing `ui-chat` ChatView and documenting the cap; `ui-trajectory` already virtualizes.
- Add an a11y regression test (axe or DOM assertions) to the browser lane.

---

## Phase 6 — CI & release gates (M1, M2, M3 corrected, M4) · ~3–5 days

- **M1 desktop:** new workflow (or extend `ci.yml`) building/packaging `apps/desktop` via `apps/desktop/electron-builder.config.mjs`; unsigned package on PR, sign/notarize on release. Currently zero `electron|desktop` references under `.github`.
- **M2 web lane OS coverage:** `webSnapshotGate` (`scripts/run-gates.ts:483`) runs only in `ci-linux-primary` (`ci-master.yml:183`). Add a Windows leg (reuse `windows-build`/new job) and a macOS leg (re-enable `serial-macos`, `ci-master.yml:190`). Fix the `bash` vs `pwsh` tool-roster differential that makes Windows `shipped-composition.e2e.ts` fail (the report pins that lane to Linux).
- **M3 (corrected):** keep the real-key suite gated; add a **keyless** boot→authenticate→reconnect scenario that always runs, and make a missing dist a hard failure (`requireDist()`) rather than a silent `notReady` skip.
- **M4 coverage debt:** shrink the `TODO(gui)` excludes in `vitest.config.ts:224,291,308,344` — prioritize `client/web/src` (boot), `host/webserver`, `client/connection/{index,http-bridge}`, then the `ui-*` files.
- Add a **Windows release smoke** for the GUI so the primary dev OS is exercised.

---

## Phase 7 — Docs, contracts & polish (M12, LOW) · ~1–2 days

- **M12:** add `zustand`+immer to `VENDOR_PACKAGES` in `apps/web/vite.config.ts:113` (or amend the `:84-101` docblock); confirm with the bundle source maps.
- Document the browser floor (es2022 target / ES2024 lib → ~Chrome 119 / Safari 17.4) and add a `browserslist` so it is enforced rather than folklore.
- Pin or bound the whole-major carets on the shipped bundle family (shiki/lexical/xterm/katex/mdast/micromark); remove the alpha `@playwright/mcp` from any runtime closure; align the react floor (`^18.2.0` across 51 manifests) with the locked 18.3.1.
- Build-env allowlist: move from prefix-only to an exact key set (`scripts/client-build-environment.ts:173-177`).
- Update the review report itself with §0.3 corrections so future readers are not misled.

---

## 8. Suggested sequencing & effort

| Order | Phase | Depends on | Rough effort |
|---|---|---|---|
| 0 | Land in-flight fixes + hygiene | — | 0.5 d |
| 1 | B1 distribution | 0 | 0.5 d |
| 2 | H1/H2 token crash | 0 | 1–2 d |
| 3 | H4/H5/H3/M6/M7 headers, maps, secrets, limits | 0 | 4–6 d |
| 4 | M8/M11 + LOW | 1,3 | 2–3 d |
| 5 | M9/M10 a11y | 0 | 2–3 d |
| 6 | M1/M2/M3/M4 CI gates | 1,2,3 | 3–5 d |
| 7 | M12 + docs/pins | 3 | 1–2 d |
| 8 | Full gate run + release checklist | all | 1 d |

Total ≈ **3–4 focused weeks** (matches the review's "weeks-scoped polish, not redesign").

## 9. Risks / rollback
- **CSP nonce** can break third-party inline injection rows → ship `Report-Only` first, enforce after a clean violation window.
- **Removing served source maps** breaks local debugging → gate by profile, keep dev maps.
- **Secret migration** (DPAPI/Keychain) is stateful → migrate-in-place with a fallback to the legacy plaintext record for one release, and re-mint on first run.
- **Coverage excludes** removal may spike failures → peel one path per PR.

## 10. Verification commands (run after each phase)
- `pnpm run check:ci`  (and `check:ci:linux-primary` for the web lane)
- `pnpm run check:all`
- Targeted: `vitest` for `packages/client/ui-theme`, `packages/host/webserver`, `packages/client/connection`, `packages/client/store`, `packages/client/ui-primitives`
- Live: rebuild Web artifacts and re-verify `http://127.0.0.1:3080` (headers, `.map` 404, theme toggle, file editor, console clean)
