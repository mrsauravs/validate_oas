# **OpenAPI Spec Validator & Publisher 🚀 (v2.0)**

A **Streamlit** application that validates OpenAPI Specifications (OAS) and publishes them to [ReadMe.io](https://readme.io), with multi-provider AI assistance for diagnosing errors, auto-fixing YAML, and reviewing specs against a baked-in OpenAPI style guide.

This is an updated version of the original app. See **"What changed in v2.0"** below for the full list of fixes and additions.

## **✨ Features**

* **Multi-Validator Support:** Runs Swagger CLI, Redocly CLI, and ReadMe's CLI (`rdme`) in one go.
* **Git Integration:** Clones your private repository directly within the app (supports PAT & SSO).
* **Real-time Logging:** See validation errors and process logs instantly in the UI.
* **Automatic Fixes:** Injects `x-readme` extensions and updates server URLs before uploading.
* **Multi-Provider AI Assistant:** Choose **Google Gemini**, **OpenAI**, or **DeepSeek** to analyze validation errors, auto-fix your YAML, and explicitly check your spec against a structured OpenAPI style guide.
* **Baked-in OpenAPI Style Guide:** A structured, explicit rule set (naming, structure, descriptions, error conventions, etc.) — viewable in the app and used verbatim by the AI reviewer, not just "give general feedback."
* **Self-healing Node.js toolchain:** Automatically provisions a modern local Node.js runtime if the host's system Node is too old for the current CLI tools (see below).
* **Downloadable Logs:** Download full execution logs for debugging.

## **🛠️ Prerequisites**

1. **Python 3.9+**
2. **Node.js & npm** (only needed as a fallback bootstrap — see [Node.js Runtime](#-nodejs-runtime-important) below)
3. **Git**

## **🚀 Quick Start (Local)**

1. **Clone this repository:**
   ```
   git clone <this-repo-url>
   cd <repo-folder>
   ```

2. **Install Python Dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Set up Secrets (optional but recommended)** — create `.streamlit/secrets.toml`:
   ```
   README_API_KEY = "your-readme-api-key"
   GIT_USERNAME = "your-github-handle"
   GIT_TOKEN = "your-github-pat"
   ```
   (Note: the current UI takes these as sidebar inputs rather than reading secrets automatically — wiring secrets in is a good first customization if you deploy this for a team.)

4. **Run the App:**
   ```
   streamlit run streamlit_app.py
   ```

## **☁️ Deployment (Streamlit Cloud)**

1. Push this code to a GitHub repository.
2. Connect your repo to [Streamlit Cloud](https://share.streamlit.io/).
3. Ensure `packages.txt` includes `nodejs`, `npm`, and `git` (already set).
4. Deploy. On first run, the app will check the platform's Node.js version and, if needed, transparently download a modern Node runtime the first time you click Validate/Upload (see next section) — this can take ~30–60 seconds once, then it's cached for the life of the container.

### ⚠️ Node.js Runtime (important)

Streamlit Community Cloud runs on Debian, and the `nodejs` package available via `apt`/`packages.txt` is frequently **older** than what current CLI tools require — as of this update, **`@redocly/cli` 2.x requires Node ≥ 20.19/22.12** and **`rdme` 9.x requires Node ≥ 20.10**, while Debian's apt `nodejs` package can lag behind that.

To make this robust without depending on custom apt repos (which `packages.txt` doesn't support), the app now:
1. Checks the system `node`/`npx` version at runtime.
2. If it's missing or older than Node 20, it uses the `nodeenv` Python package (no root required) to download and install an isolated, modern Node.js LTS runtime into `~/.cache/openapi_validator_node/`.
3. Uses that runtime's `npx` for every CLI call, and caches the resolution for the life of the container (`st.cache_resource`).

If this provisioning fails (e.g. no outbound network access to nodejs.org), the app now fails **loudly and clearly** in the log panel instead of crashing with an opaque `TypeError` (this was bug #2 below).

## **📖 How to Use**

### **1. Configuration (Sidebar)**

* **ReadMe API Key:** Your project key from ReadMe.io.
* **ReadMe API Mode:** Pick **Classic (API v1)** or **ReadMe Refactored (API v2)** — see the dedicated section below if you're not sure which one applies to you.
* **AI Config:** Pick a provider (Gemini / OpenAI / DeepSeek), paste that provider's API key, and optionally override the model name.
* **Git Repo URL / Username / Token:** Credentials for the repo containing your OpenAPI specs.
  * *Note:* If your organization uses **SSO**, you must authorize your PAT for that organization.
* **Internal Paths:** The relative path(s) inside your repo where `.yaml` files live.

### **2. Main Dashboard**

1. **Select OpenAPI File:** The app lists all `.yaml` files found in your repo.
2. **API Version:** Enter the version string (e.g., `1.0`, `2024.3`).
3. **Validation Settings:** Choose which validators to run:
   * **Swagger CLI:** Legacy OAS 2.0/3.0 check (**unmaintained upstream** — see Known Limitations).
   * **Redocly CLI:** Modern OAS 3.0/3.1 linting — also now used for bundling on new-API uploads.
   * **ReadMe CLI:** Checks specifically for ReadMe platform compatibility.
4. **Validate** to dry-run, or **Upload** to publish to ReadMe.
5. **View baked-in OpenAPI style guide** (expander): see the exact rules the AI style-guide checker uses.

### **3. AI Helper (after a Validate/Upload run, once a key is entered)**

* **🧐 Analyze Errors** — explains the validator log output and suggests fixes.
* **✨ Auto-Fix** — rewrites the YAML to address the logged errors (now validated as parseable YAML before it's offered for upload — see bug #4 below).
* **📐 Check Style Guide** — runs the spec through every rule in the baked-in style guide and reports pass/fail per rule with the specific path/operation/schema at fault, not generic advice.

### **4. Troubleshooting**

* **SSO Error (403):** If you see a "SAML SSO" error in the logs, click the authorization link provided in the error message to authorize your token.
* **401 from ReadMe:** As of v2.0 the Basic Auth header is correctly base64-encoded (see bug #1) — a 401 now almost always means the API key itself is wrong/revoked, not an encoding bug.
* **Validation Failed:** Read the logs. If Redocly or ReadMe reports errors (like trailing slashes or missing `$ref`), fix them **in your source YAML file** in your repository. The script does not silently fix content errors for you (the "Auto-Fix" button is opt-in and explicit).

## **📂 File Structure**

* `streamlit_app.py` — the main Streamlit application.
* `requirements.txt` — Python dependencies.
* `packages.txt` — system dependencies for Streamlit Cloud (Node.js, npm, git).

---

## **🐛 What changed in v2.0**

### Bugs fixed (as requested)

1. **ReadMe API Basic Auth was not base64-encoded.** The original code sent `Authorization: Basic <raw_api_key>`. HTTP Basic Auth requires `Basic base64("<api_key>:")`. This silently broke (or 401'd) every direct `requests` call to the ReadMe API (`check_and_create_version`, `get_api_id`, `create_new_api_via_requests`). Fixed via a single `readme_auth_header()` helper used everywhere.
2. **`get_npx_path()`'s return value was never checked.** If `npx` wasn't on `PATH`, `None` was passed straight into `subprocess.Popen`, raising an opaque `TypeError` deep in `run_command`. `ensure_npx()` now returns `None` explicitly on failure, and the caller checks for that and stops with a clear, actionable log message before running anything.
3. **`process_yaml_content` assumed `info` always exists.** A spec with no `info` block (or a YAML file that isn't a mapping at all — e.g. empty, or a bare list) crashed with an unhandled exception. The function now validates that the parsed YAML is a dict, and synthesizes a minimal `info` block (with a warning in the log) if one is missing, instead of crashing.
4. **`apply_ai_fixes` silently swallowed exceptions**, returning `None` with no indication of what went wrong. It now returns `(result, error)` and the UI surfaces the actual error message (auth failure, network error, bad YAML from the model, etc.) instead of just failing quietly.
5. **`clear_creds()` didn't clear all stored keys.** It only cleared `readme_key`/`git_user`/`git_token`, leaving the (single, Gemini-only) AI key in session state. Now clears the ReadMe key, git credentials, and **all three** provider AI keys.

### Other bugs found and fixed along the way

* **Unpinned/inconsistent tool versions.** `swagger-cli` and `rdme@8` had no version pin (or an old one), so behavior could silently change between runs. All three CLI tools are now pinned to specific, verified-working versions (see below), so upgrades are a deliberate, testable choice rather than something that happens under you.
* **Bundling used `swagger-cli bundle`**, which only understands OpenAPI 3.0 and is officially unmaintained (see below) — it would fail outright on OAS 3.1 specs. New-API uploads now bundle with `@redocly/cli`, which supports both 3.0 and 3.1.
* **`rdme@8`'s deprecated `openapi:validate` colon syntax** was in use; updated to the current `openapi validate` space syntax (the colon form still works but is deprecated upstream).
* **AI response from Auto-Fix was used as-is with no sanity check.** If the model returned malformed YAML or prose instead of a YAML block, that broken content could be saved and offered as the "corrected" file for upload. `apply_ai_fixes` now parses the candidate YAML and rejects it (with a clear error) if it isn't valid.
* Minor: ReadMe API error responses (401, non-200) were previously ignored/uninspected in `check_and_create_version` and `get_api_id`; specific status codes are now logged to make auth vs. server errors distinguishable.

## **⬆️ Dependency upgrades**

Checked against the npm registry at the time of writing:

| Tool | Old (implicit/pinned) | New (pinned) | Notes |
|---|---|---|---|
| `swagger-cli` | unpinned (`swagger-cli` → latest) | `swagger-cli@4.0.4` | **Officially abandoned upstream** — npm prints `This package has been abandoned. Please switch to using the actively maintained @redocly/cli` on every install. Still functional for OAS 2.0/3.0 validation, but treat it as a legacy/optional checkbox, not your primary validator. Bundling now uses Redocly instead. |
| `@redocly/cli` | `@redocly/cli@1.25.0` | `@redocly/cli@2.47.0` | **Breaking:** v2 requires **Node.js ≥ 20.19 (or ≥ 22.12)** — this is the main reason the Node self-provisioning was added. Some v1 config/rule names changed in v2's `redocly.yaml` schema if you use a custom config (not used by this app by default). |
| `rdme` | `rdme@8` | `rdme@9` **and** `rdme@10` | **Breaking:** both require **Node.js ≥ 20.10**. Command topic separator changed from colon to space (`openapi:validate` → `openapi validate`; colon form still accepted but deprecated). `rdme@9` targets *classic* ReadMe projects (API v1, Basic Auth); `rdme@10` targets **ReadMe Refactored** projects (API v2, Bearer auth, `openapi upload` with `--branch`/`--slug` instead of `--version`/`--id`). The app now picks the right one automatically based on the **ReadMe API Mode** you select — see the dedicated section below. |

## **🤖 AI provider setup**

Pick a provider in the sidebar's **AI Config** panel; only that provider's key is required.

| Provider | Where to get a key | Default model (editable) |
|---|---|---|
| Google Gemini | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | `gemini-2.5-flash` |
| OpenAI | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | `gpt-4o-mini` |
| DeepSeek | [platform.deepseek.com](https://platform.deepseek.com) | `deepseek-chat` |

OpenAI and DeepSeek are both called via their OpenAI-compatible Chat Completions REST endpoint (no extra SDK needed, keeps `requirements.txt` lean); Gemini uses the `google-genai` SDK. The **Model Name** field is free text, so you can point it at any current model your account has access to (e.g. a newer DeepSeek or OpenAI release) — check the provider's docs for the latest model names, since those move faster than this app can track.

## **☝️ One Upload button, mode-aware (not two)**

Since adding ReadMe Refactored support, it might look like "Upload" needs two separate buttons — one for classic hex-ID-based updates, one for Refactored's filename/slug-based ones — since those are genuinely different upload mechanics. It doesn't, and deliberately so: the sidebar's **ReadMe API Mode** dropdown is the single source of truth for the *entire* pipeline (hostname, auth scheme, and which `rdme` major version runs), and it's re-read fresh on every click — including Upload, which always re-runs the full validate step before publishing rather than reusing anything from a prior Validate click. So the Upload button already branches internally:

* **Classic (v1) selected:** looks up an existing API by title (`get_api_id`), corrects the title if there's a fuzzy match, then updates via `rdme openapi <file> --id <hex-id> --key ...` — or bundles + creates a new one via a raw multipart request if no match is found.
* **ReadMe Refactored (v2) selected:** checks for an existing definition by **filename** (`get_api_definition_v2`), then always runs `rdme openapi upload <file> --branch <branch> --key ...` — which creates *or* updates in one call, since v2 has no hex ID to look up.

Two separate buttons would still each need to agree with the mode dropdown for auth/hostname to be correct, which just relocates the "did I pick the right one" risk rather than removing it. Instead, there's now a confirmation line directly above the Upload button showing exactly what will run before you click it, e.g.:

> Will upload using **ReadMe Refactored (API v2 · Bearer Auth)** → `rdme@10` → `https://api.readme.com/v2` (create/update via filename, no ID needed).

...and the button label itself includes the active `rdme` package (`🚀 Upload via rdme@10: Original (Edited)`), so the mode is visible without having to scroll back up to the sidebar.

## **🌿 Switching Git Branches from the UI**

You're not limited to `main`/`master`. In the sidebar's **Git Config** section:

* **Branch Name** is a free-text field — type any branch (a feature branch, an engineer's working branch, a release branch, etc.) and click **Validate** or **Upload**; the app always switches the local clone to that exact branch first, then runs.
* **🔎 List Branches** fetches every branch on the remote (via `git ls-remote`, no clone needed) and shows a picker — select one and click **✅ Use Selected Branch** to fill in the Branch Name field for you, if you don't want to type it from memory.
* **🔀 Switch Now** switches the local clone to the typed branch immediately, without running any validators — handy if you just want to confirm you're pointed at the right branch before kicking off a full run.
* A **📍 Local clone is on: `<branch>` (`<commit>`)** caption is always visible under Git Config, reflecting what's actually checked out on disk right now — so you're never guessing which branch you're about to validate.

**Bug fixed while adding this:** the original clone command (`git clone --depth 1 --branch <branch>`) created a shallow, *single-branch* clone. On a later run, switching to a *different* branch via `git fetch origin <branch>` followed by `git checkout <branch>` would fail with `pathspec '<branch>' did not match any file(s)`, because that plain fetch only updates the anonymous `FETCH_HEAD`, not a named local ref — there was nothing for `checkout` to switch to unless it happened to be the branch originally cloned. Branch switching now uses a forced refspec fetch (`git fetch --depth 1 origin +<branch>:<branch>`), which creates/updates a same-named local branch directly from the remote every time, so checkout always finds it — verified against a real local repo with two diverging branches, switching back and forth cleanly.

## **🔀 ReadMe API Mode: Classic (v1) vs. ReadMe Refactored (v2)**

As of this update, the sidebar has a **ReadMe API Mode** dropdown so the app works whether or not your ReadMe project has migrated to **ReadMe Refactored**. This replaces the "known limitation" from the previous version, where the app only spoke API v1.

**Not sure which one you're on?** Check your ReadMe dashboard's project settings, or just try **Classic** first — if your project has actually moved to Refactored, you'll get a clear `401`/`404` in the logs (the classic `/version` and `/api-specification` endpoints don't exist there), and you can switch modes and re-run.

| | **Classic (API v1)** | **ReadMe Refactored (API v2)** |
|---|---|---|
| Hostname | `https://dash.readme.com/api/v1` | `https://api.readme.com/v2` |
| Auth | Basic (`base64("<key>:")`) | Bearer (`Bearer <key>`) |
| Concept | "Versions" (`x-readme-version` header) | "Branches" (in the URL path, e.g. `/branches/stable/...`) |
| `rdme` pin | `rdme@9` | `rdme@10` |
| Resource identity | Hex ID, matched to your file by **title** (fuzzy) | **Filename/slug** — the file you upload *is* the identifier |
| Create vs. update | The app looks up an existing hex ID by title; if found, updates via `rdme openapi`; if not, bundles with Redocly and `POST`s directly to `/api-specification` to create it | `rdme openapi upload <file> --branch <branch> --key <key>` handles **both** create and update automatically — no separate bundle/create-via-request step needed, since v10 resolves local `$ref`s itself |

**Field labels adapt automatically:** the "API Version" input becomes "Branch" when you select ReadMe Refactored, and defaults to `stable` instead of `1.0`.

**Creating a new branch/version if it doesn't exist yet:** both modes support this on Upload. For v2, branch creation POSTs to `/branches` with a best-effort body (`{"name": ..., "base": ...}`) inferred from ReadMe's published field-mapping table — ReadMe's docs render the exact request schema via an interactive JS widget that isn't fully captured in static docs, so if branch auto-creation fails, check the [Create a branch reference](https://docs.readme.com/main/reference/createbranch), adjust the JSON body in `check_and_create_readme_version()`, or just create the branch once manually in the ReadMe dashboard and re-run.

## **📐 OpenAPI style guide**

The app now bakes in a structured, explicit rule set (visible in-app under **"View baked-in OpenAPI style guide"**) covering:

* Root/metadata requirements (`info`, `servers`, `tags`, contact/license)
* Path & URL naming conventions (kebab-case, no verbs, param casing)
* `operationId` conventions and required `summary`/`description`/`tags`
* Parameter conventions (descriptions, consistent pagination/filter naming, typed schemas, enums)
* Schema naming (PascalCase, no `Model`/`DTO` suffixes, consistent property casing, required required-arrays)
* Request/response/example conventions
* HTTP status code usage and a single consistent error-response shape
* Security scheme requirements
* Versioning consistency

The **"Check Style Guide"** AI action sends this exact rule list (not a vague "review this spec" prompt) alongside your YAML, and asks the model to return a PASS/FAIL verdict **per rule**, citing the specific path/operation/schema/property at fault for each failure — so you get an auditable checklist, not generic prose.

## **⚠️ Known Limitations**

* **Trailing Slashes:** ReadMe strictly rejects paths ending in `/`. Keep your YAML paths clean (`/users`, not `/users/`).
* **ReadMe Refactored branch creation schema is best-effort.** Everything else in the v2 path (validate, upload/create/update via `rdme openapi upload`, existence lookup) is confirmed against ReadMe's current CLI and docs. The one exception is auto-creating a brand-new branch that doesn't exist yet (`POST /branches`) — ReadMe's docs render that endpoint's exact request body via an interactive widget that isn't captured in static documentation, so the body sent is a best-effort inference from ReadMe's v1→v2 field-mapping table. If it fails, create the branch once by hand in the dashboard, or adjust the request body in `check_and_create_readme_version()`.
* **`swagger-cli` is unmaintained upstream** (last published 2020, explicitly marked "abandoned" by its own install output). It's kept as an optional legacy check because it's fast and simple, but Redocly is the actively maintained tool and is now used for bundling; consider disabling the Swagger CLI checkbox entirely if you don't need OAS 2.0 support.
* **First-run Node provisioning latency.** If the host's system Node is too old, the very first Validate/Upload click on a fresh container will take longer (downloading a Node runtime via `nodeenv`) before CLI tools can run. Subsequent runs in the same container are fast.
* **AI suggestions are not guaranteed correct.** `Auto-Fix` output is validated to be parseable YAML before it's offered for upload, but it is not guaranteed to be *semantically* correct — always review the diff before uploading an AI-corrected spec. The **Check Style Guide** results are similarly a starting point, not a certification.
* **Fuzzy API matching.** `get_api_id`'s "smart match" falls back to token-set equality on titles, which could match the wrong existing API if two specs have very similar titles — review the "Correcting Title" log line before trusting an update-in-place.
* **Log rendering cost.** The log panel re-joins and re-renders the full log list on every single line for live updates; this is fine for typical validation runs but can get slow on extremely large/noisy log output.
