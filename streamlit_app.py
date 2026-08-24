import streamlit as st
import yaml
import subprocess
import shutil
import requests
import os
import sys
import logging
import base64
import hashlib
import urllib.parse
import re
from pathlib import Path
from google import genai

# =============================================================================
# Page Config
# =============================================================================
st.set_page_config(
    page_title="OpenAPI Spec Validator v2.0",
    page_icon=":material/api:",
    layout="wide"
)

MIN_NODE_MAJOR = 20  # @redocly/cli 2.x and rdme 9.x both require Node >= 20

# =============================================================================
# OpenAPI Style Guide (used both to lint via AI and to show in the UI)
# =============================================================================
# This is a condensed, structured rule set inspired by industry style guides
# (Stripe, Google, Microsoft, Zalando). It is intentionally explicit and
# checkable, rather than "write good docs" style vague advice.
OPENAPI_STYLE_GUIDE = [
    {
        "category": "Root & Metadata",
        "rules": [
            "`openapi` field must be present and pinned to a specific version (e.g. 3.1.0), not a range.",
            "`info.title`, `info.version`, and `info.description` are all required. `info.version` should follow semantic versioning (MAJOR.MINOR.PATCH).",
            "`info.description` should explain what the API does, who it's for, and link to auth/getting-started docs — not just restate the title.",
            "`info.contact` (name/url/email) and `info.license` should be present for any public-facing API.",
            "`servers` must be defined with real URLs (no `localhost` in production specs) and each server should have a `description`.",
            "Top-level `tags` array should exist, with every tag given a `description`, and operations should reference only tags declared there.",
        ],
    },
    {
        "category": "Paths & URL Design",
        "rules": [
            "Paths use kebab-case, lowercase, plural nouns for collections (e.g. `/user-accounts`, not `/UserAccount` or `/getUsers`).",
            "No verbs in paths — the HTTP method conveys the action (`GET /orders/{orderId}`, not `/getOrder/{id}`).",
            "Path parameters use camelCase inside braces (e.g. `{orderId}`) and must be declared with `required: true` and an explicit `schema`.",
            "No trailing slashes on paths, and no ambiguous overlapping path templates.",
            "Nesting reflects real ownership only (max ~2 levels deep); avoid deep nesting like `/a/{aId}/b/{bId}/c/{cId}/d`.",
        ],
    },
    {
        "category": "Operations & operationId",
        "rules": [
            "Every operation has a unique `operationId` in camelCase following a verb+resource pattern (`listOrders`, `getOrderById`, `createOrder`, `cancelOrder`).",
            "Every operation has a concise `summary` (<= ~120 chars) and a longer `description` written in full sentences.",
            "Every operation is assigned at least one `tag` matching a root-level tag definition.",
            "Deprecated operations set `deprecated: true` and explain the replacement/sunset timeline in the description (or an `x-sunset` extension).",
        ],
    },
    {
        "category": "Parameters",
        "rules": [
            "Every parameter (path, query, header, cookie) has a non-empty `description` explaining its meaning, format, and constraints.",
            "Query parameters use consistent, predictable names across the whole spec: pagination as `limit`/`offset` or `cursor`, sorting as `sort`, filtering with a clear prefix.",
            "Booleans avoid double negatives (`includeArchived`, not `notExcludeArchived`).",
            "Every parameter declares a `schema` with an explicit `type` (and `format` where relevant, e.g. `date-time`, `uuid`); avoid untyped `{}` schemas.",
            "Enums are declared with `enum` values rather than described only in prose.",
        ],
    },
    {
        "category": "Schemas & Naming",
        "rules": [
            "Schema names in `components/schemas` use PascalCase (e.g. `Order`, `CreateOrderRequest`) and avoid redundant suffixes like `Model`/`DTO`/`Object`.",
            "Request and response bodies are separate named schemas, not deeply inlined anonymous objects, so they're reusable and diffable.",
            "Every schema property has a `description`, and required properties are listed explicitly in a `required` array rather than relying on defaults.",
            "Property naming case is consistent across the entire document (pick camelCase or snake_case and use it everywhere — never mix).",
            "Avoid abbreviations and reserved words in property names (`identifier` not `id` alone where ambiguous, no `class`, `type` used loosely, etc.), and prefer explicit types over generic `object`/`any`.",
            "Free-form/additionalProperties objects are the exception, not the default — most payloads should have a defined shape.",
        ],
    },
    {
        "category": "Requests, Responses & Examples",
        "rules": [
            "Request and response bodies declare an explicit `content` type (`application/json` unless there's a specific reason otherwise).",
            "Every operation defines responses for its success case AND realistic error cases (400, 401/403 if secured, 404 where applicable, 429, 5xx).",
            "Each response (including errors) references a schema — no bare `description`-only responses for anything other than 204.",
            "At least one `example` or `examples` entry is provided for non-trivial request/response bodies.",
        ],
    },
    {
        "category": "HTTP Status Codes & Errors",
        "rules": [
            "200 for successful GET/PUT/PATCH, 201 for successful creation (with a `Location` header documented), 204 for successful deletes with no body.",
            "400 for malformed/invalid input, 401 for missing/invalid auth, 403 for authenticated-but-forbidden, 404 for missing resources, 409 for conflicts, 422 for semantically invalid input, 429 for rate limiting.",
            "All error responses share one consistent error schema across the whole API (e.g. `{ \"error\": { \"code\", \"message\", \"details\" } }` or RFC 7807 `application/problem+json`), not a different ad-hoc shape per endpoint.",
        ],
    },
    {
        "category": "Security",
        "rules": [
            "`components.securitySchemes` is defined and referenced via `security` (globally and/or per-operation) — auth is never left undocumented.",
            "API keys/tokens are passed via headers (or OAuth2/OpenID flows), never as query string parameters.",
            "Publicly documented operations that require auth explicitly show a 401/403 response.",
        ],
    },
    {
        "category": "Versioning & Compatibility",
        "rules": [
            "The API's version strategy is unambiguous — either in the URL path (`/v1/...`), a header, or `info.version` mapped to a ReadMe/host version — and it's applied consistently, not mixed.",
            "Breaking changes bump the major version; additive/backward-compatible changes do not.",
        ],
    },
]


def render_style_guide_markdown() -> str:
    lines = ["## OpenAPI Style Guide (baked-in rule set)\n"]
    for section in OPENAPI_STYLE_GUIDE:
        lines.append(f"**{section['category']}**")
        for rule in section["rules"]:
            lines.append(f"- {rule}")
        lines.append("")
    return "\n".join(lines)


def render_style_guide_prompt() -> str:
    """Numbered, compact form of the style guide for injection into an AI prompt."""
    lines = []
    n = 1
    for section in OPENAPI_STYLE_GUIDE:
        lines.append(f"### {section['category']}")
        for rule in section["rules"]:
            lines.append(f"{n}. {rule}")
            n += 1
    return "\n".join(lines)


# =============================================================================
# AI Provider Config (multi-provider: Gemini / OpenAI / DeepSeek)
# =============================================================================
AI_PROVIDERS = {
    "Google Gemini": {
        "default_model": "gemini-2.5-flash",
        "key_help": "Get a key from Google AI Studio (aistudio.google.com/apikey).",
    },
    "OpenAI": {
        "default_model": "gpt-4o-mini",
        "key_help": "Get a key from platform.openai.com/api-keys.",
    },
    "DeepSeek": {
        "default_model": "deepseek-chat",
        "key_help": "Get a key from platform.deepseek.com.",
    },
}


def call_ai(provider, api_key, model_name, prompt):
    """
    Unified call across providers. Returns (text, error) — exactly one is None.
    OpenAI and DeepSeek both speak the OpenAI-compatible Chat Completions API,
    so they're handled with plain `requests` calls; Gemini uses the google-genai SDK.
    """
    if not api_key:
        return None, "No API key provided for the selected provider."
    if not model_name:
        return None, "No model name provided for the selected provider."

    try:
        if provider == "Google Gemini":
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(model=model_name, contents=[prompt])
            text = getattr(response, "text", None)
            if not text:
                return None, "Gemini returned an empty response."
            return text, None

        elif provider == "OpenAI":
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model_name, "messages": [{"role": "user", "content": prompt}]},
                timeout=120,
            )
            if resp.status_code != 200:
                return None, f"OpenAI API error ({resp.status_code}): {resp.text[:500]}"
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content")
            if not text:
                return None, "OpenAI returned an empty response."
            return text, None

        elif provider == "DeepSeek":
            resp = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model_name, "messages": [{"role": "user", "content": prompt}]},
                timeout=120,
            )
            if resp.status_code != 200:
                return None, f"DeepSeek API error ({resp.status_code}): {resp.text[:500]}"
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content")
            if not text:
                return None, "DeepSeek returned an empty response."
            return text, None

        else:
            return None, f"Unknown AI provider: {provider}"

    except requests.exceptions.RequestException as e:
        return None, f"Network error calling {provider}: {e}"
    except Exception as e:
        return None, f"{provider} error: {e}"


def analyze_errors_with_ai(log_content, provider, api_key, model_name):
    prompt = (
        "Analyze these OpenAPI validation/upload logs and explain the errors, "
        "then suggest concrete fixes:\n\n" + log_content
    )
    text, error = call_ai(provider, api_key, model_name, prompt)
    if error:
        return None, error
    return text, None


def apply_ai_fixes(original_path, log_content, provider, api_key, model_name):
    """
    Returns (fixed_yaml_text_or_None, error_message_or_None).
    Errors are surfaced to the caller instead of being swallowed.
    """
    try:
        with open(original_path, "r") as f:
            yaml_content = f.read()
    except Exception as e:
        return None, f"Could not read source file '{original_path}': {e}"

    prompt = f"""
Fix the errors in the logs for this OpenAPI YAML file.
PRESERVE the 'x-readme', 'servers', and 'info' sections exactly unless they are the cause of an error.
Return ONLY valid YAML code, wrapped in a single ```yaml code block, with no other commentary.

Logs:
{log_content}

YAML:
{yaml_content}
"""
    text, error = call_ai(provider, api_key, model_name, prompt)
    if error:
        return None, error

    match = re.search(r"```yaml\s*\n(.*?)\n```", text, re.DOTALL)
    candidate = match.group(1) if match else text

    # Sanity-check that the AI actually returned parseable YAML before we hand
    # it back as a "corrected" file the user might upload.
    try:
        parsed = yaml.safe_load(candidate)
        if not isinstance(parsed, dict):
            return None, "AI response did not parse into a valid OpenAPI YAML object."
    except yaml.YAMLError as e:
        return None, f"AI response was not valid YAML: {e}"

    return candidate, None


def build_active_style_guide_text(mode, custom_text):
    """
    Combines the built-in OPENAPI_STYLE_GUIDE and/or a user-supplied custom
    style guide (uploaded or pasted as Markdown) into one block of text to
    hand to the AI reviewer, based on the selected mode:
    "Built-in only" | "Custom only" | "Both (built-in + custom)".
    """
    parts = []
    custom_text = (custom_text or "").strip()

    if mode in ("Built-in only", "Both (built-in + custom)"):
        parts.append("### Built-in Style Guide\n" + render_style_guide_prompt())

    if mode in ("Custom only", "Both (built-in + custom)") and custom_text:
        parts.append("### Custom Style Guide\n" + custom_text)

    if not parts:
        # Fallback: never send an empty guide (e.g. "Custom only" selected but
        # nothing was actually uploaded/pasted yet).
        parts.append("### Built-in Style Guide\n" + render_style_guide_prompt())

    return "\n\n".join(parts)


def check_spec_against_style_guide(yaml_content, provider, api_key, model_name, style_guide_text):
    """Explicit, rule-by-rule AI review against the active style guide (not generic advice)."""
    prompt = f"""
You are an OpenAPI spec reviewer. Check the OpenAPI document below AGAINST EVERY rule or
guideline in the style guide provided below, whether it's numbered or written as prose. Treat
each distinct guideline as one checkable item.

For every item, respond with one line:
- PASS — a short label/number for the rule and a 3-8 word reason, OR
- FAIL — a short label/number for the rule, the exact location in the spec (path/operation/schema/property), and a one-sentence fix.

Only elaborate on FAIL items; PASS items can stay a single compact line. Group your output under
the same section/category headings used in the style guide below (if it has a "Built-in Style
Guide" and a "Custom Style Guide" section, review the spec against both and keep your results
grouped under those same two headings). Be specific — cite actual paths, operationIds, and
property names from the document, not generic advice.

## Style Guide
{style_guide_text}

## OpenAPI Document
```yaml
{yaml_content}
```
"""
    text, error = call_ai(provider, api_key, model_name, prompt)
    return text, error


# =============================================================================
# Custom Logging Handler
# =============================================================================
class StreamlitLogHandler(logging.Handler):
    def __init__(self, container, download_placeholder=None):
        super().__init__()
        self.container = container
        self.download_placeholder = download_placeholder

    def emit(self, record):
        msg = self.format(record)
        st.session_state.logs.append(msg)
        self.container.code("\n".join(st.session_state.logs), language="text")

        if self.download_placeholder:
            unique_key = f"log_dl_rt_{len(st.session_state.logs)}"
            self.download_placeholder.download_button(
                label="Download Log File",
                icon=":material/download:",
                data="\n".join(st.session_state.logs),
                file_name="openapi_upload.log",
                mime="text/plain",
                key=unique_key,
            )


# =============================================================================
# Node.js / npx provisioning
# =============================================================================
# BUG FIX: the original get_npx_path() just returned shutil.which("npx") and
# callers passed that straight into subprocess.Popen without checking for
# None, which crashes with a confusing TypeError if Node isn't on PATH — or,
# worse, silently runs against a too-old system Node (Streamlit Community
# Cloud's Debian `nodejs` apt package is currently Node 18.x, but
# @redocly/cli 2.x and rdme 9.x both require Node >= 20). We now verify the
# system Node version and, if it's missing or too old, provision an isolated
# modern Node runtime on the fly via `nodeenv` (pure-Python, no root needed)
# and cache the result for the life of the container.
def _node_major_version(node_path):
    try:
        out = subprocess.run([node_path, "-v"], capture_output=True, text=True, timeout=10)
        version = out.stdout.strip().lstrip("v")
        return int(version.split(".")[0])
    except Exception:
        return None


@st.cache_resource(show_spinner=":material/build: Preparing Node.js toolchain (first run only)...")
def ensure_npx():
    """Returns a usable npx path with Node >= MIN_NODE_MAJOR, or None if unavailable."""
    system_node = shutil.which("node")
    system_npx = shutil.which("npx")
    if system_node and system_npx:
        major = _node_major_version(system_node)
        if major is not None and major >= MIN_NODE_MAJOR:
            return system_npx

    # Fall back to a self-provisioned Node runtime.
    env_dir = Path.home() / ".cache" / "openapi_validator_node"
    npx_path = env_dir / "bin" / "npx"
    if npx_path.exists():
        major = _node_major_version(str(env_dir / "bin" / "node"))
        if major is not None and major >= MIN_NODE_MAJOR:
            return str(npx_path)

    try:
        env_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, "-m", "nodeenv", "--node=lts", "--force", str(env_dir)],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if npx_path.exists():
            return str(npx_path)
    except Exception:
        pass

    return None


# =============================================================================
# Helper Functions
# =============================================================================
def validate_env(api_key, required=True):
    if not api_key:
        if required:
            st.error("ReadMe API Key is missing. Please enter it in the sidebar.", icon=":material/key:")
            st.stop()
        return False
    return True


def run_command(command_list, log_logger, cwd=None):
    # BUG FIX: guard against a None entry (e.g. npx not resolved) instead of
    # letting subprocess.Popen raise an opaque TypeError.
    if not command_list or command_list[0] is None:
        log_logger.error("Command failed: required executable (npx) was not found.")
        return 1
    try:
        cmd_str = " ".join(command_list)
        dir_msg = f" (in {cwd})" if cwd else ""
        log_logger.info(f"Running: {cmd_str}{dir_msg}")

        process = subprocess.Popen(
            command_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            cwd=cwd,
        )
        for line in process.stdout:
            clean = line.strip()
            if clean:
                log_logger.info(f"[CLI] {clean}")
        process.wait()
        return process.returncode
    except Exception as e:
        log_logger.error(f"Command failed: {e}")
        return 1


# =============================================================================
# Git Logic
# =============================================================================
def build_authenticated_repo_url(repo_url, git_username, git_token):
    """Builds a token-authenticated HTTPS clone URL. Raises on malformed input."""
    repo_url = repo_url.strip().strip('"').strip("'")
    if repo_url.count("https://") > 1:
        match = re.search(r"(https://github\.com/.*)$", repo_url)
        if match:
            repo_url = match.group(1)
    parsed = urllib.parse.urlparse(repo_url)
    safe_user = urllib.parse.quote(git_username.strip(), safe="")
    safe_token = urllib.parse.quote(git_token.strip(), safe="")
    clean_netloc = parsed.netloc.split("@")[-1] if "@" in parsed.netloc else parsed.netloc
    return urllib.parse.urlunparse(
        (parsed.scheme, f"{safe_user}:{safe_token}@{clean_netloc}", parsed.path, parsed.params, parsed.query, parsed.fragment)
    )


def git_clone_or_switch(repo_url, repo_dir, git_token, git_username, branch_name):
    """
    Pure git operation (no Streamlit calls) so it can be reused by both the
    main validate/upload pipeline (which logs to the on-screen log panel and
    stops on failure) and lightweight sidebar actions (which just want a
    quick success/fail + message, without aborting the whole page).

    Clones the repo if it isn't present locally yet, targeting `branch_name`
    directly. If it IS already present (from a previous run, possibly on a
    different branch), fetches, checks out, and pulls `branch_name` — this is
    what lets a user type any branch name and have the app switch to it,
    rather than being stuck on whatever branch was first cloned.

    Returns (success: bool, summary: str, detail_lines: list[str]).
    """
    repo_path = Path(repo_dir)
    lines = []
    try:
        auth_repo_url = build_authenticated_repo_url(repo_url, git_username, git_token)
    except Exception as e:
        return False, f"URL Error: {e}", lines

    clean_env = os.environ.copy()
    clean_env["GIT_TERMINAL_PROMPT"] = "0"

    if not repo_path.exists():
        lines.append(f"⬇️ Cloning branch '{branch_name}'...")
        cmd = ["git", "clone", "--depth", "1", "--branch", branch_name, "--single-branch", auth_repo_url, str(repo_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, env=clean_env)
        if result.returncode != 0:
            return False, f"Git Clone Failed: {result.stderr.strip()[:500]}", lines
        lines.append("Repo cloned successfully.")
        return True, f"Cloned and checked out '{branch_name}'.", lines

    lines.append(f"Switching to branch '{branch_name}'...")
    # BUG FIX (found while adding branch switching): the initial clone above
    # is shallow AND single-branch, so only the originally-cloned branch gets
    # a local ref. A later `git fetch origin <branch>` (no refspec) only
    # updates the anonymous FETCH_HEAD, not a named ref — so `git checkout
    # <branch>` fails with "pathspec did not match any file(s)" for any
    # branch other than the one first cloned. Using an explicit forced
    # refspec (`+<branch>:<branch>`) creates/updates a local branch ref
    # directly from the remote, so checkout always finds it — this is what
    # actually makes "type any branch name and switch to it" work reliably,
    # not just for the branch that happened to be cloned first.
    steps = [
        ["git", "-C", str(repo_path), "remote", "set-url", "origin", auth_repo_url],
        ["git", "-C", str(repo_path), "fetch", "--depth", "1", "origin", f"+{branch_name}:{branch_name}"],
        ["git", "-C", str(repo_path), "checkout", branch_name],
    ]
    for step in steps:
        result = subprocess.run(step, capture_output=True, text=True, env=clean_env)
        if result.returncode != 0:
            return False, f"Git Update Failed on `{' '.join(step[2:])}`: {result.stderr.strip()[:500]}", lines
    lines.append(f"Switched to '{branch_name}'.")
    return True, f"Switched to '{branch_name}'.", lines


def setup_git_repo(repo_url, repo_dir, git_token, git_username, branch_name, logger):
    """Logger-driven wrapper used inside the main Validate/Upload pipeline."""
    logger.info(f"Starting Git Operation for branch: {branch_name}...")
    success, summary, detail_lines = git_clone_or_switch(repo_url, repo_dir, git_token, git_username, branch_name)
    for line in detail_lines:
        logger.info(line)
    if not success:
        logger.error(f"{summary}")
        st.stop()


def list_remote_branches(repo_url, git_username, git_token):
    """Lists branch names on the remote without cloning. Returns (branches, error)."""
    if not repo_url or not git_username or not git_token:
        return [], "Fill in Git HTTPS URL, Git User, and Git Token first."
    try:
        auth_repo_url = build_authenticated_repo_url(repo_url, git_username, git_token)
    except Exception as e:
        return [], f"Could not parse repo URL: {e}"
    try:
        clean_env = os.environ.copy()
        clean_env["GIT_TERMINAL_PROMPT"] = "0"
        result = subprocess.run(
            ["git", "ls-remote", "--heads", auth_repo_url],
            capture_output=True, text=True, env=clean_env, timeout=30,
        )
        if result.returncode != 0:
            return [], f"git ls-remote failed: {result.stderr.strip()[:300]}"
        branches = []
        for line in result.stdout.splitlines():
            parts = line.strip().split("\t")
            if len(parts) == 2 and parts[1].startswith("refs/heads/"):
                branches.append(parts[1][len("refs/heads/"):])
        return sorted(branches), None
    except subprocess.TimeoutExpired:
        return [], "Timed out contacting the remote repository."
    except Exception as e:
        return [], f"Error listing branches: {e}"


def get_current_git_branch(repo_dir):
    """Returns (branch_name, short_commit_summary) for an existing local clone, or (None, None)."""
    path = Path(repo_dir)
    if not path.exists():
        return None, None
    try:
        branch = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "-C", str(path), "log", "-1", "--format=%h · %cr"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return (branch or None), (commit or None)
    except Exception:
        return None, None


def delete_repo(repo_dir):
    path = Path(repo_dir)
    if path.exists():
        try:
            shutil.rmtree(path)
            return True, "Deleted successfully."
        except Exception as e:
            return False, f"Error: {e}"
    return False, "Path does not exist."


# =============================================================================
# File Ops
# =============================================================================
SPEC_EXTENSIONS = (".yaml", ".yml", ".json")


def prepare_files(filename, paths, workspace, dependency_list, logger):
    # BUG FIX: file discovery was hardcoded to ".yaml" only, so a JSON (or
    # .yml) OpenAPI file would never be found here even if it showed up in
    # the file picker. Now tries all supported extensions, in each search
    # path, in order.
    source = None
    search_dirs = [Path(paths["specs"])]
    if paths.get("secondary"):
        search_dirs.append(Path(paths["secondary"]))
    for d in search_dirs:
        for ext in SPEC_EXTENSIONS:
            candidate = d / f"{filename}{ext}"
            if candidate.exists():
                source = candidate
                break
        if source:
            break

    if not source:
        tried = ", ".join(f"{filename}{ext}" for ext in SPEC_EXTENSIONS)
        logger.error(f"Source file not found. Tried: {tried}")
        st.stop()

    dest_dir = Path(workspace)
    dest_dir.mkdir(parents=True, exist_ok=True)
    destination = dest_dir / source.name
    shutil.copy(source, destination)
    logger.info(f"Copied {source.suffix.lstrip('.').upper()} spec to workspace: {destination.name}")

    for folder in dependency_list:
        clean = folder.strip()
        if not clean:
            continue
        src = Path(paths["specs"]) / clean
        dest = dest_dir / clean
        if src.exists():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
            logger.info(f"Copied dependency: {clean}")
    return destination


def process_yaml_content(file_path, version, api_domain, logger):
    logger.info("Injecting extensions...")
    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        # BUG FIX: the original code assumed `data` was always a populated
        # dict with an "info" key. An empty file, a non-mapping YAML document
        # (e.g. a bare list), or a spec missing "info" would raise an
        # unhandled AttributeError/TypeError deep inside this function.
        if not isinstance(data, dict):
            logger.error("YAML Process Error: the file did not parse into a YAML mapping (object) at the top level.")
            st.stop()

        if "openapi" in data:
            pos = list(data.keys()).index("openapi")
            items = list(data.items())
            items.insert(pos + 1, ("x-readme", {"explorer-enabled": False}))
            data = dict(items)

        if "info" not in data or not isinstance(data.get("info"), dict):
            logger.warning("Spec is missing a valid 'info' section — creating a minimal one.")
            data["info"] = {"title": file_path.stem, "version": version}

        data["info"]["version"] = version
        domain = api_domain if api_domain else "example.com"

        if "servers" not in data or not data["servers"]:
            data["servers"] = [{"url": f"https://{domain}", "variables": {}}]

        if "variables" not in data["servers"][0]:
            data["servers"][0]["variables"] = {}

        data["servers"][0]["variables"]["base-url"] = {"default": domain}
        data["servers"][0]["variables"]["protocol"] = {"default": "https"}

        edited_path = file_path.parent / (file_path.stem + "_edited.yaml")
        with open(edited_path, "w") as f:
            yaml.dump(data, f, sort_keys=False)
        logger.info(f"Edited YAML saved: {edited_path.name}")
        return edited_path
    except Exception as e:
        logger.error(f"YAML Process Error: {e}")
        st.stop()


# =============================================================================
# ReadMe Logic — supports BOTH the classic API v1 (Basic Auth, "versions")
# and ReadMe Refactored's API v2 (Bearer Auth, "branches"). Pick a mode in
# the sidebar; everything below branches on `mode_conf`.
# =============================================================================

# Which ReadMe project type you're on determines the hostname, auth scheme,
# terminology ("version" vs "branch"), and which rdme major version to use.
# rdme@9 only understands classic (API v1) projects; rdme@10 is required for
# ReadMe Refactored (API v2) projects and won't work against classic ones.
# See: https://docs.readme.com/main/reference/api-upgrade-guide
README_MODES = {
    "Classic (API v1 · Basic Auth)": {
        "id": "v1",
        "base_url": "https://dash.readme.com/api/v1",
        "rdme_pkg": "rdme@9",
        "term": "version",
        "term_label": "API Version",
    },
    "ReadMe Refactored (API v2 · Bearer Auth)": {
        "id": "v2",
        "base_url": "https://api.readme.com/v2",
        "rdme_pkg": "rdme@10",
        "term": "branch",
        "term_label": "Branch",
    },
}


def readme_auth_header(api_key):
    """
    BUG FIX: ReadMe's classic (API v1) API uses HTTP Basic Auth where the
    "username" is the API key and the password is empty — the header must be
    `Basic base64("<api_key>:")`, NOT the raw key. Sending the raw key (as
    the original app did) fails auth silently / returns 401s.
    """
    token = base64.b64encode(f"{api_key}:".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def readme_headers(api_key, mode_conf, extra=None):
    """Returns the correct auth header for the selected ReadMe API mode."""
    if mode_conf["id"] == "v2":
        headers = {"Authorization": f"Bearer {api_key}"}
    else:
        headers = readme_auth_header(api_key)
    if extra:
        headers.update(extra)
    return headers


def check_and_create_readme_version(version, api_key, base_url, mode_conf, logger, create_if_missing=False):
    """
    Ensures a project version (v1) / branch (v2) exists, optionally creating it.
    v1: GET/POST /version
    v2: GET/POST /branches (ReadMe Refactored only)
    """
    if not api_key:
        return
    headers = {**readme_headers(api_key, mode_conf), "Accept": "application/json"}

    if mode_conf["id"] == "v1":
        logger.info(f"Checking version '{version}'...")
        try:
            res = requests.get(f"{base_url}/version", headers=headers)
            if res.status_code == 200:
                versions = res.json()
                if any(v["version"] == version for v in versions):
                    logger.info(f"Version '{version}' exists.")
                    return
                if create_if_missing:
                    logger.info(f"Creating version '{version}'...")
                    fork_from = versions[0]["version"] if versions else "latest"
                    create_res = requests.post(
                        f"{base_url}/version", headers=headers,
                        json={"version": version, "is_stable": False, "from": fork_from},
                    )
                    if create_res.status_code not in (200, 201):
                        logger.error(f"Version creation failed ({create_res.status_code}): {create_res.text[:300]}")
            elif res.status_code == 401:
                logger.error("ReadMe auth failed (401). Double-check your API key.")
            else:
                logger.error(f"Version check failed ({res.status_code}): {res.text[:300]}")
        except Exception as e:
            logger.error(f"Version check failed: {e}")

    else:  # v2 — ReadMe Refactored branches
        logger.info(f"Checking branch '{version}'...")
        try:
            res = requests.get(f"{base_url}/branches", headers=headers, params={"prefix": version})
            if res.status_code == 200:
                body = res.json()
                branches = body.get("data", body if isinstance(body, list) else [])
                if any(b.get("name") == version for b in branches):
                    logger.info(f"Branch '{version}' exists.")
                    return
                if create_if_missing:
                    logger.info(f"Creating branch '{version}'...")
                    base_branch = next((b.get("name") for b in branches if b.get("release_stage") == "default"), "stable")
                    create_res = requests.post(
                        f"{base_url}/branches", headers=headers,
                        json={"name": version, "base": base_branch},
                    )
                    if create_res.status_code not in (200, 201):
                        # NOTE: ReadMe's exact create-branch request schema isn't fully
                        # documented publicly at time of writing; if this fails, check
                        # https://docs.readme.com/main/reference/createbranch and adjust
                        # the JSON body above, or create the branch once manually in the
                        # ReadMe dashboard and re-run.
                        logger.error(f"Branch creation failed ({create_res.status_code}): {create_res.text[:300]}")
            elif res.status_code == 401:
                logger.error("ReadMe auth failed (401). Double-check your API key.")
            elif res.status_code == 404:
                logger.error("Branches endpoint not found (404) — is this project actually on ReadMe Refactored?")
            else:
                logger.error(f"Branch check failed ({res.status_code}): {res.text[:300]}")
        except Exception as e:
            logger.error(f"Branch check failed: {e}")


def get_api_id(api_name, version, api_key, base_url, logger):
    """v1-only: classic API definitions are identified by hex ID, matched by title."""
    if not api_key:
        return None, None
    headers = {**readme_auth_header(api_key), "Accept": "application/json", "x-readme-version": version}

    try:
        logger.info(f"Looking for ID for Title: '{api_name}'")

        def tokenize(text):
            return set(re.findall(r"\w+", text.lower()))

        target_tokens = tokenize(api_name)

        res = requests.get(f"{base_url}/api-specification", headers=headers, params={"perPage": 100})
        if res.status_code == 200:
            apis = res.json()
            for api in apis:
                if api["title"] == api_name:
                    logger.info(f"Exact Match: {api['_id']}")
                    return api["_id"], api["title"]
            for api in apis:
                if target_tokens == tokenize(api["title"]):
                    logger.info(f"Smart Match: '{api['title']}' (ID: {api['_id']})")
                    return api["_id"], api["title"]
            logger.warning(f"No match found for '{api_name}'")
        elif res.status_code == 401:
            logger.error("ReadMe auth failed (401). Double-check your API key.")
        else:
            logger.error(f"API Error: {res.status_code}")
    except Exception as e:
        logger.error(f"ID Lookup Error: {e}")
    return None, None


def create_new_api_via_requests(file_path, version, api_key, base_url, logger):
    """v1-only: directly uploads a new spec to ReadMe via requests to bypass CLI prompts."""
    logger.info("Creating NEW API definition directly via API...")
    headers = {**readme_auth_header(api_key), "x-readme-version": version}

    try:
        with open(file_path, "rb") as f:
            files = {"spec": (file_path.name, f)}
            res = requests.post(f"{base_url}/api-specification", headers=headers, files=files)

        if res.status_code in [200, 201]:
            new_id = res.json().get("_id")
            logger.info(f"Successfully Created! ID: {new_id}")
            return new_id
        else:
            logger.error(f"API Upload Failed ({res.status_code}): {res.text[:300]}")
            return None
    except Exception as e:
        logger.error(f"Upload Exception: {e}")
        return None


def get_api_definition_v2(filename, branch, api_key, base_url, logger):
    """
    v2-only: ReadMe Refactored identifies API definitions by filename (slug),
    not a fuzzy title match, so we just check whether a definition with this
    exact filename already exists on this branch.
    GET /branches/{branch}/apis/{filename} -> 200 (exists) / 404 (doesn't).
    """
    if not api_key:
        return None
    headers = {**readme_headers(api_key, README_MODES["ReadMe Refactored (API v2 · Bearer Auth)"]), "Accept": "application/json"}
    try:
        url = f"{base_url}/branches/{urllib.parse.quote(branch, safe='')}/apis/{urllib.parse.quote(filename, safe='')}"
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            data = res.json().get("data", res.json())
            logger.info(f"Found existing API definition for '{filename}' on branch '{branch}'.")
            return data
        elif res.status_code == 404:
            logger.info(f"ℹ️ No existing API definition for '{filename}' on branch '{branch}' — will create new.")
            return None
        elif res.status_code == 401:
            logger.error("ReadMe auth failed (401). Double-check your API key.")
            return None
        else:
            logger.warning(f"Could not check existing API definition ({res.status_code}): {res.text[:300]}")
            return None
    except Exception as e:
        logger.warning(f"API definition lookup error: {e}")
        return None


def clear_creds():
    # BUG FIX: the original list only cleared readme_key/git_user/git_token,
    # leaving every AI provider's API key sitting in session state.
    keys_to_clear = [
        "readme_key", "git_user", "git_token",
        "gemini_key", "openai_key", "deepseek_key",
        "remote_branches", "remote_branch_picker",
        "logs",
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state.logs = []


def clear_logs():
    st.session_state.logs = []


# =============================================================================
# Main
# =============================================================================
def main():
    if "logs" not in st.session_state:
        st.session_state.logs = []

    st.sidebar.title(":material/settings: Configuration")

    str_defaults = ["readme_key", "gemini_key", "openai_key", "deepseek_key", "git_user", "git_token", "repo_url"]
    for k in str_defaults:
        if k not in st.session_state:
            st.session_state[k] = ""
    if "branch_name" not in st.session_state:
        st.session_state.branch_name = "main"
    for k in ["last_edited_file", "corrected_file"]:
        if k not in st.session_state:
            st.session_state[k] = None
    if "ai_provider" not in st.session_state:
        st.session_state.ai_provider = "Google Gemini"
    provider_key_map = {"Google Gemini": "gemini_key", "OpenAI": "openai_key", "DeepSeek": "deepseek_key"}
    provider_model_map = {"Google Gemini": "ai_model_gemini", "OpenAI": "ai_model_openai", "DeepSeek": "ai_model_deepseek"}
    for prov, model_key in provider_model_map.items():
        if model_key not in st.session_state:
            st.session_state[model_key] = AI_PROVIDERS[prov]["default_model"]

    readme_key = st.sidebar.text_input("ReadMe API Key", key="readme_key", type="password")

    if "readme_mode" not in st.session_state:
        st.session_state.readme_mode = "Classic (API v1 · Basic Auth)"
    readme_mode_label = st.sidebar.selectbox(
        "ReadMe API Mode",
        list(README_MODES.keys()),
        key="readme_mode",
        help=(
            "Classic = pre-\"ReadMe Refactored\" projects (Basic Auth, rdme@9). "
            "ReadMe Refactored = projects on the new branch-based API v2 (Bearer Auth, rdme@10). "
            "Not sure which you're on? Check Project Settings in your ReadMe dashboard, "
            "or start with Classic — you'll get a clear 404/401 in the logs if it's wrong."
        ),
    )
    mode_conf = README_MODES[readme_mode_label]

    with st.sidebar.expander("AI Config", icon=":material/smart_toy:", expanded=True):
        provider = st.selectbox("AI Provider", list(AI_PROVIDERS.keys()), key="ai_provider")
        active_key_name = provider_key_map[provider]
        active_model_key = provider_model_map[provider]
        ai_api_key = st.text_input(
            f"{provider} API Key", key=active_key_name, type="password",
            help=AI_PROVIDERS[provider]["key_help"],
        )
        # BUG FIX (rough edge, not in the original bug list, found while adding
        # multi-provider support): the model field is now keyed per-provider so
        # switching providers doesn't leave a stale model name from a different
        # provider sitting in the box.
        ai_model = st.text_input(f"{provider} Model Name", key=active_model_key)
        if not ai_model:
            ai_model = AI_PROVIDERS[provider]["default_model"]

    st.sidebar.subheader("Git Config")
    repo_path = st.sidebar.text_input("Local Clone Path", value="./cloned_repo")
    if st.sidebar.button("Delete Repo", icon=":material/delete:"):
        s, m = delete_repo(repo_path)
        if s:
            st.sidebar.success(m)
        else:
            st.sidebar.warning(m)

    repo_url = st.sidebar.text_input("Git HTTPS URL", key="repo_url")

    # Free-text branch input — type any branch name (feature branch, an
    # engineer's working branch, a release branch, etc.), not just main/master.
    branch_name = st.sidebar.text_input(
        "Branch Name", key="branch_name",
        help="Any branch on the repo — e.g. 'feature/payments-api'. Clicking Validate/Upload always switches the local clone to this branch first.",
    )
    git_user = st.sidebar.text_input("Git User", key="git_user")
    git_token = st.sidebar.text_input("Git Token", key="git_token", type="password")

    gb1, gb2 = st.sidebar.columns(2)
    if gb1.button("List Branches", icon=":material/search:", use_container_width=True):
        with st.spinner("Fetching branch list..."):
            branches, err = list_remote_branches(repo_url, git_user, git_token)
        if err:
            st.sidebar.error(err, icon=":material/error:")
            st.session_state.pop("remote_branches", None)
        else:
            st.session_state.remote_branches = branches
            st.sidebar.success(f"Found {len(branches)} branch(es).", icon=":material/check_circle:")

    switch_clicked = gb2.button("Switch Now", icon=":material/sync:", use_container_width=True)

    if st.session_state.get("remote_branches"):
        def _use_picked_branch():
            st.session_state.branch_name = st.session_state.remote_branch_picker

        st.sidebar.selectbox(
            "Pick a branch",
            st.session_state.remote_branches,
            key="remote_branch_picker",
            label_visibility="collapsed",
        )
        st.sidebar.button("Use Selected Branch", icon=":material/check_circle:", on_click=_use_picked_branch, use_container_width=True)

    if switch_clicked:
        if not (repo_url and git_user and git_token):
            st.sidebar.error("Fill in Git HTTPS URL, Git User, and Git Token first.", icon=":material/error:")
        else:
            with st.spinner(f"Switching local clone to '{branch_name}'..."):
                success, summary, _ = git_clone_or_switch(repo_url, repo_path, git_token, git_user, branch_name)
            if success:
                st.sidebar.success(summary, icon=":material/check_circle:")
            else:
                st.sidebar.error(summary, icon=":material/error:")

    # Always-visible indicator of what's actually checked out locally right
    # now, so switching branches (here or via Validate/Upload) is never a
    # guessing game about which branch you're about to validate against.
    current_branch, current_commit = get_current_git_branch(repo_path)
    if current_branch:
        commit_suffix = f" ({current_commit})" if current_commit else ""
        st.sidebar.caption(f":material/location_on: Local clone is on: **{current_branch}**{commit_suffix}")
    else:
        st.sidebar.caption(":material/location_on: No local clone yet — Validate/Upload/Switch Now will clone it.")

    st.sidebar.button("Clear Credentials", icon=":material/lock:", on_click=clear_creds)

    st.sidebar.subheader("Paths")
    spec_rel = st.sidebar.text_input("Main Specs Path", value="specs")
    sec_rel = st.sidebar.text_input("Secondary Path (Opt)", value="")
    dep_in = st.sidebar.text_input("Dependency Folders", value="common")
    deps = [x.strip() for x in dep_in.split(",")]
    domain = st.sidebar.text_input("API Domain", value="api.example.com")

    abs_spec = Path(repo_path) / spec_rel
    paths = {"repo": repo_path, "specs": abs_spec}
    if sec_rel:
        paths["secondary"] = Path(repo_path) / sec_rel
    workspace_dir = "./temp_workspace"

    st.title(":material/api: OpenAPI Spec Validator")

    if "custom_style_guide_text" not in st.session_state:
        st.session_state.custom_style_guide_text = ""
    if "custom_style_guide_name" not in st.session_state:
        st.session_state.custom_style_guide_name = None
    if "custom_style_guide_hash" not in st.session_state:
        st.session_state.custom_style_guide_hash = None
    if "style_guide_mode" not in st.session_state:
        st.session_state.style_guide_mode = "Built-in only"

    with st.expander("OpenAPI Style Guide (built-in, and/or your own)", icon=":material/rule:"):
        st.markdown(render_style_guide_markdown())

        st.markdown("---")
        st.markdown(
            "**Bring your own style guide.** Upload a Markdown/text file describing a different "
            "or house-specific set of OpenAPI conventions (e.g. your own team's naming rules, a "
            "different vendor's guide, an internal playbook), or paste/edit one directly below. "
            "The uploaded content is loaded into the text box so you can tweak it before use — "
            "nothing is written back to the file itself."
        )

        uploaded_guide = st.file_uploader(
            "Upload a custom style guide (.md / .txt)",
            type=["md", "markdown", "txt"],
            key="custom_style_guide_upload",
        )
        if uploaded_guide is not None:
            # BUG FIX: dedup was keyed on filename alone, so re-uploading the
            # same filename with edited content (a very likely workflow —
            # tweak your local style-guide.md, re-upload) silently did
            # nothing, since the name hadn't changed. Hash the actual bytes
            # instead, so any content change is picked up regardless of name.
            raw_bytes = uploaded_guide.read()
            content_hash = hashlib.md5(raw_bytes).hexdigest()
            if content_hash != st.session_state.get("custom_style_guide_hash"):
                try:
                    st.session_state.custom_style_guide_text = raw_bytes.decode("utf-8", errors="replace")
                    st.session_state.custom_style_guide_name = uploaded_guide.name
                    st.session_state.custom_style_guide_hash = content_hash
                    st.success(f"Loaded '{uploaded_guide.name}' — edit below if needed.", icon=":material/check_circle:")
                except Exception as e:
                    st.error(f"Could not read uploaded file: {e}", icon=":material/error:")

        st.text_area(
            "Custom style guide content (editable)",
            key="custom_style_guide_text",
            height=200,
            placeholder="# My Team's OpenAPI Style Guide\n\n- All paths must be kebab-case...\n- ...",
        )

        st.selectbox(
            "Which style guide should 'Check Style Guide' use?",
            ["Built-in only", "Custom only", "Both (built-in + custom)"],
            key="style_guide_mode",
            help="Applies to the 'Check Style Guide' AI action further down, after you run Validate/Upload.",
        )
        if st.session_state.style_guide_mode != "Built-in only" and not st.session_state.custom_style_guide_text.strip():
            st.warning("No custom style guide text yet — upload a file or paste one above, or the check will fall back to built-in only.", icon=":material/warning:")

    c1, c2 = st.columns(2)
    with c1:
        files = []
        if abs_spec.exists():
            for ext in SPEC_EXTENSIONS:
                files.extend([f.stem for f in abs_spec.glob(f"*{ext}")])
        if "secondary" in paths and paths["secondary"].exists():
            for ext in SPEC_EXTENSIONS:
                files.extend([f.stem for f in paths["secondary"].glob(f"*{ext}")])
        files = sorted(list(set(files)))
        selected_file = st.selectbox("Select File", files) if files else st.text_input("Filename", "audit")
        st.caption("Looks for `.yaml`, `.yml`, or `.json` files under the configured Specs path(s).")

    with c2:
        default_version = "1.0" if mode_conf["id"] == "v1" else "stable"
        version = st.text_input(mode_conf["term_label"], default_version)

    st.markdown("### :material/tune: Settings")
    ch1, ch2, ch3 = st.columns(3)
    with ch1:
        use_sw = st.checkbox("Swagger CLI (legacy, OAS 3.0 only)", True)
    with ch2:
        use_re = st.checkbox("Redocly CLI", True)
    with ch3:
        use_rd = st.checkbox("ReadMe CLI", False)

    st.markdown("---")
    u_opts = ["Original (Edited)"]
    if st.session_state.corrected_file:
        u_opts.append("AI Corrected")

    cs1, cs2 = st.columns([1, 2])
    with cs1:
        u_choice = st.radio("Upload:", u_opts, horizontal=True)
    with cs2:
        cb1, cb2 = st.columns(2)
        b_val = cb1.button("Validate", icon=":material/fact_check:", use_container_width=True)
        b_up = cb2.button(f"Upload via {mode_conf['rdme_pkg']}: {u_choice}", icon=":material/cloud_upload:", type="primary", use_container_width=True)

    # Confirmation line so the active ReadMe API mode is never a guess right
    # before you click Upload — a single mode toggle in the sidebar drives
    # BOTH the hostname/auth AND which rdme flow (hex-ID vs filename/slug)
    # runs, so this is the one thing worth double-checking before publishing.
    st.caption(
        f"Will upload using **{readme_mode_label}** → `{mode_conf['rdme_pkg']}` → `{mode_conf['base_url']}` "
        f"({'update via hex ID, matched by title' if mode_conf['id'] == 'v1' else 'create/update via filename, no ID needed'}). "
        "Change this in the sidebar's **ReadMe API Mode** if it's wrong."
    )

    st.markdown("### :material/terminal: Logs")
    log_con = st.empty()
    if st.session_state.logs:
        log_con.code("\n".join(st.session_state.logs), language="text")

    cd1, cd2, cd3 = st.columns([1, 1, 3])
    with cd1:
        dl_ph = st.empty()
        if st.session_state.logs:
            dl_ph.download_button("Logs", "\n".join(st.session_state.logs), "log.txt", icon=":material/download:", key=f"dl_{len(st.session_state.logs)}")

    if b_val or b_up:
        st.session_state.logs = []
        st.session_state.last_edited_file = None
        st.session_state.corrected_file = None

        logger = logging.getLogger("st_log")
        logger.setLevel(logging.INFO)
        if logger.handlers:
            logger.handlers = []
        handler = StreamlitLogHandler(log_con, dl_ph)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)

        has_key = validate_env(readme_key, required=bool(b_up))

        npx = ensure_npx()
        if npx is None:
            logger.error(
                "Could not find or provision a Node.js runtime (>= v%d) for npx. "
                "Check that packages.txt includes nodejs/npm, and that this container "
                "has outbound network access to install a local Node via nodeenv." % MIN_NODE_MAJOR
            )
            st.error("Node.js/npx is unavailable — validation and upload cannot proceed. See logs.", icon=":material/error:")
            st.stop()

        base_url = mode_conf["base_url"]
        rdme_pkg = mode_conf["rdme_pkg"]

        setup_git_repo(repo_url, repo_path, git_token, git_user, branch_name, logger)
        logger.info("Preparing workspace...")
        final_yaml = prepare_files(selected_file, paths, workspace_dir, deps, logger)

        abs_workspace_path = Path(workspace_dir).resolve()

        if has_key:
            check_and_create_readme_version(version, readme_key, base_url, mode_conf, logger, bool(b_up))

        edited = process_yaml_content(final_yaml, version, domain, logger)
        st.session_state.last_edited_file = str(edited)
        target = edited.resolve()

        if b_up and u_choice == "AI Corrected" and st.session_state.corrected_file:
            target = Path(st.session_state.corrected_file).resolve()

        do_s = True if b_up else use_sw
        do_r = False if b_up else use_re
        do_rm = True if b_up else use_rd
        fail = False

        if do_s:
            logger.info("Running Swagger CLI (legacy validator)...")
            if run_command([npx, "--yes", "swagger-cli@4.0.4", "validate", target.name], logger, cwd=abs_workspace_path) != 0:
                fail = True

        if do_r:
            logger.info("Running Redocly...")
            if run_command([npx, "--yes", "@redocly/cli@2.47.0", "lint", target.name], logger, cwd=abs_workspace_path) != 0:
                fail = True

        if do_rm and has_key:
            logger.info(f"Running ReadMe CLI ({rdme_pkg})...")
            if run_command([npx, "--yes", rdme_pkg, "openapi", "validate", target.name], logger, cwd=abs_workspace_path) != 0:
                fail = True

        if fail:
            logger.error("Validation Failed.")
            st.error("Errors found.", icon=":material/error:")
        else:
            logger.info("Validated.")
            if b_up:
                logger.info(f"Uploading via {mode_conf['id']} ({rdme_pkg})...")

                if mode_conf["id"] == "v1":
                    # --- Classic API v1 flow: hex-ID lookup + title correction ---
                    with open(target, "r") as f:
                        ydata = yaml.safe_load(f)
                        ytitle = (ydata or {}).get("info", {}).get("title", "")

                    api_id, matched_title = get_api_id(ytitle, version, readme_key, base_url, logger)

                    if api_id and matched_title and matched_title != ytitle:
                        logger.info(f"Correcting Title: '{ytitle}' -> '{matched_title}'")
                        ydata["info"]["title"] = matched_title
                        with open(target, "w") as f:
                            yaml.dump(ydata, f, sort_keys=False)

                    if api_id:
                        # Case 1: Existing API -> Update via CLI
                        cmd = [npx, "--yes", rdme_pkg, "openapi", target.name, "--useSpecVersion", "--version", version, "--id", api_id, "--key", readme_key]
                        if run_command(cmd, logger, cwd=abs_workspace_path) == 0:
                            logger.info("Updated Existing API!")
                            st.success("Success!", icon=":material/check_circle:")
                        else:
                            logger.error("Upload Failed.")
                    else:
                        # Case 2: New API -> Bundle (Redocly, supports OAS 3.0 & 3.1) + Request
                        logger.warning("No ID found. Treating as NEW API.")
                        logger.info("Bundling references with Redocly...")
                        bundled_name = f"{target.stem}_bundled.yaml"
                        bundle_cmd = [npx, "--yes", "@redocly/cli@2.47.0", "bundle", target.name, "-o", bundled_name, "--ext", "yaml"]
                        if run_command(bundle_cmd, logger, cwd=abs_workspace_path) == 0:
                            bundled_path = abs_workspace_path / bundled_name
                            create_new_api_via_requests(bundled_path, version, readme_key, base_url, logger)
                            st.success("Success!", icon=":material/check_circle:")
                        else:
                            logger.error("Bundling failed.")

                else:
                    # --- ReadMe Refactored / API v2 flow ---
                    # v2 identifies API definitions by filename, not a fuzzy title
                    # match, and rdme@10's `openapi upload` creates OR updates in a
                    # single call (the --id flag from v9 was renamed to --slug and
                    # is now optional/inferred from the filename), so there's no
                    # separate "bundle + raw multipart POST" fallback needed here —
                    # rdme resolves local $refs itself before uploading.
                    existing = get_api_definition_v2(target.name, version, readme_key, base_url, logger)
                    action = "Updating existing" if existing else "Creating new"
                    logger.info(f"{action} API definition '{target.name}' on branch '{version}'...")

                    cmd = [npx, "--yes", rdme_pkg, "openapi", "upload", target.name, "--branch", version, "--key", readme_key]
                    if run_command(cmd, logger, cwd=abs_workspace_path) == 0:
                        logger.info("Upload complete!")
                        st.success("Success!", icon=":material/check_circle:")
                    else:
                        logger.error("Upload Failed.")

            else:
                st.success("Done.", icon=":material/check_circle:")

    with cd2:
        if st.session_state.last_edited_file:
            p = Path(st.session_state.last_edited_file)
            if p.exists():
                with open(p, "r") as f:
                    st.download_button("Edited YAML", f.read(), p.name, "application/x-yaml", icon=":material/description:")

    with cd3:
        if st.session_state.logs:
            st.button("Clear Logs", icon=":material/delete_sweep:", on_click=clear_logs)

    active_provider = st.session_state.ai_provider
    active_ai_key = st.session_state.get(provider_key_map[active_provider], "")
    active_model = st.session_state.get(provider_model_map[active_provider]) or AI_PROVIDERS[active_provider]["default_model"]

    if st.session_state.logs and active_ai_key:
        st.markdown(f"### :material/smart_toy: AI Helper ({active_provider})")
        ca1, ca2, ca3 = st.columns(3)

        if ca1.button("Analyze Errors", icon=":material/troubleshoot:"):
            with st.spinner("Thinking..."):
                an, err = analyze_errors_with_ai("\n".join(st.session_state.logs), active_provider, active_ai_key, active_model)
                if err:
                    st.error(f"AI analysis failed: {err}", icon=":material/error:")
                elif an:
                    st.markdown(an)

        if ca2.button("Auto-Fix", icon=":material/auto_fix_high:"):
            if st.session_state.last_edited_file:
                with st.spinner("Fixing..."):
                    fix, err = apply_ai_fixes(
                        st.session_state.last_edited_file, "\n".join(st.session_state.logs), active_provider, active_ai_key, active_model
                    )
                    if err:
                        st.error(f"AI auto-fix failed: {err}", icon=":material/error:")
                    elif fix:
                        op = Path(st.session_state.last_edited_file)
                        cp = op.parent / (op.stem.replace("_edited", "") + "_corrected.yaml")
                        with open(cp, "w") as f:
                            f.write(fix)
                        st.session_state.corrected_file = str(cp)
                        st.success("Fixed! Choose 'AI Corrected' above.", icon=":material/check_circle:")
                        st.rerun()

        if ca3.button(f"Check Style Guide ({st.session_state.style_guide_mode})", icon=":material/rule:"):
            if st.session_state.last_edited_file:
                with st.spinner("Checking against style guide..."):
                    p = Path(st.session_state.last_edited_file)
                    yaml_text = p.read_text()
                    style_guide_text = build_active_style_guide_text(
                        st.session_state.style_guide_mode, st.session_state.custom_style_guide_text
                    )
                    review, err = check_spec_against_style_guide(yaml_text, active_provider, active_ai_key, active_model, style_guide_text)
                    if err:
                        st.error(f"Style guide check failed: {err}", icon=":material/error:")
                    elif review:
                        st.markdown(review)

    if st.session_state.corrected_file:
        cp = Path(st.session_state.corrected_file)
        if cp.exists():
            with open(cp, "r") as f:
                st.download_button("Corrected YAML", f.read(), cp.name, "application/x-yaml", icon=":material/auto_fix_high:")


if __name__ == "__main__":
    main()
