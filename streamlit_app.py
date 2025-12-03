import streamlit as st
import yaml
import subprocess
import shutil
import requests
import os
import logging
import sys
import urllib.parse
import re
import platform
from pathlib import Path
from google import genai

# Page Config
st.set_page_config(
    page_title="OpenAPI Spec Validator v1.0",
    page_icon="📘",
    layout="wide"
)

# --- Initialize Session State for Logs ---
if 'logs' not in st.session_state:
    st.session_state.logs = []

# --- Custom Logging Handler ---
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
                label="📥 Download Log File",
                data="\n".join(st.session_state.logs),
                file_name="openapi_upload.log",
                mime="text/plain",
                key=unique_key
            )

# --- Helper Functions ---
def get_npx_path():
    return shutil.which("npx")

def validate_env(api_key, required=True):
    if not api_key:
        if required:
            st.error("❌ ReadMe API Key is missing. Please enter it in the sidebar.")
            st.stop()
        return False
    return True

def run_command(command_list, log_logger, cwd=None):
    try:
        cmd_str = " ".join(command_list)
        dir_msg = f" (in {cwd})" if cwd else ""
        log_logger.info(f"Running: {cmd_str}{dir_msg}")
        
        process = subprocess.Popen(
            command_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            cwd=cwd 
        )
        for line in process.stdout:
            clean = line.strip()
            if clean:
                log_logger.info(f"[CLI] {clean}")
        process.wait()
        return process.returncode
    except Exception as e:
        log_logger.error(f"❌ Command failed: {e}")
        return 1

# --- AI Logic ---
def analyze_errors_with_ai(log_content, api_key, model_name):
    if not api_key: return None
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"Analyze these OpenAPI logs and suggest fixes:\n{log_content}"
        response = client.models.generate_content(model=model_name, contents=[prompt])
        return response.text
    except Exception as e:
        return f"AI Error: {e}"

def apply_ai_fixes(original_path, log_content, api_key, model_name):
    if not api_key: return None
    try:
        with open(original_path, 'r') as f: yaml_content = f.read()
        client = genai.Client(api_key=api_key)
        prompt = f"""
        Fix the errors in the logs for this YAML file.
        PRESERVE 'x-readme', 'servers', and 'info'.
        Return ONLY valid YAML code.
        Logs: {log_content}
        YAML: {yaml_content}
        """
        response = client.models.generate_content(model=model_name, contents=[prompt])
        text = response.text
        match = re.search(r'```yaml\n(.*?)\n```', text, re.DOTALL)
        return match.group(1) if match else text
    except Exception:
        return None

# --- Git Logic ---
def setup_git_repo(repo_url, repo_dir, git_token, git_username, branch_name, logger):
    logger.info(f"🚀 Starting Git Operation for branch: {branch_name}...")
    repo_path = Path(repo_dir)
    repo_url = repo_url.strip().strip('"').strip("'")
    
    if repo_url.count("https://") > 1:
        match = re.search(r"(https://github\.com/.*)$", repo_url)
        if match: repo_url = match.group(1)

    try:
        parsed = urllib.parse.urlparse(repo_url)
        safe_user = urllib.parse.quote(git_username.strip(), safe='')
        safe_token = urllib.parse.quote(git_token.strip(), safe='')
        clean_netloc = parsed.netloc.split("@")[-1] if "@" in parsed.netloc else parsed.netloc
        auth_repo_url = urllib.parse.urlunparse((parsed.scheme, f"{safe_user}:{safe_token}@{clean_netloc}", parsed.path, parsed.params, parsed.query, parsed.fragment))
    except Exception as e:
        logger.error(f"❌ URL Error: {e}")
        st.stop()

    clean_env = os.environ.copy()
    clean_env["GIT_TERMINAL_PROMPT"] = "0"
    
    if not repo_path.exists():
        logger.info(f"⬇️ Cloning branch '{branch_name}'...")
        try:
            cmd = ["git", "clone", "--depth", "1", "--branch", branch_name, auth_repo_url, str(repo_path)]
            subprocess.run(cmd, check=True, capture_output=True, env=clean_env)
            logger.info("✅ Repo cloned successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Git Clone Failed: {e.stderr}")
            st.stop()
    else:
        logger.info(f"🔄 Switching to branch '{branch_name}'...")
        try:
            subprocess.run(["git", "-C", str(repo_path), "remote", "set-url", "origin", auth_repo_url], check=True, capture_output=True, env=clean_env)
            subprocess.run(["git", "-C", str(repo_path), "fetch", "origin", branch_name], check=True, capture_output=True, env=clean_env)
            subprocess.run(["git", "-C", str(repo_path), "checkout", branch_name], check=True, capture_output=True, env=clean_env)
            subprocess.run(["git", "-C", str(repo_path), "pull", "origin", branch_name], check=True, capture_output=True, env=clean_env)
            logger.info(f"✅ Switched to '{branch_name}'.")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Git Update Failed: {e}")

def delete_repo(repo_dir):
    path = Path(repo_dir)
    if path.exists():
        try:
            shutil.rmtree(path)
            return True, "Deleted successfully."
        except Exception as e:
            return False, f"Error: {e}"
    return False, "Path does not exist."

# --- File Ops ---
def prepare_files(filename, paths, workspace, dependency_list, logger):
    source = None
    main_c = Path(paths['specs']) / f"{filename}.yaml"
    if main_c.exists(): source = main_c
    elif paths.get('secondary') and (Path(paths['secondary']) / f"{filename}.yaml").exists():
        source = Path(paths['secondary']) / f"{filename}.yaml"

    if not source:
        logger.error(f"❌ Source file '{filename}.yaml' not found.")
        st.stop()

    dest_dir = Path(workspace)
    dest_dir.mkdir(parents=True, exist_ok=True)
    destination = dest_dir / source.name
    shutil.copy(source, destination)
    logger.info(f"📂 Copied YAML to workspace: {destination.name}")

    for folder in dependency_list:
        clean = folder.strip()
        if not clean: continue
        src = Path(paths['specs']) / clean
        dest = dest_dir / clean
        if src.exists():
            if dest.exists(): shutil.rmtree(dest)
            shutil.copytree(src, dest)
            logger.info(f"📂 Copied dependency: {clean}")
    return destination

def process_yaml_content(file_path, version, api_domain, logger):
    logger.info("🛠️ Injecting extensions...")
    try:
        with open(file_path, "r") as f: data = yaml.safe_load(f)
        
        if "openapi" in data:
            pos = list(data.keys()).index("openapi")
            items = list(data.items())
            items.insert(pos + 1, ("x-readme", {"explorer-enabled": False}))
            data = dict(items)
        
        data["info"]["version"] = version
        domain = api_domain if api_domain else "example.com"
        
        if "servers" not in data or not data["servers"]:
            data["servers"] = [{"url": f"https://{domain}", "variables": {}}]

        if "variables" not in data["servers"][0]:
            data["servers"][0]["variables"] = {}
        
        data["servers"][0]["variables"]["base-url"] = {"default": domain}
        data["servers"][0]["variables"]["protocol"] = {"default": "https"}

        edited_path = file_path.parent / (file_path.stem + "_edited.yaml")
        with open(edited_path, "w") as f: yaml.dump(data, f, sort_keys=False)
        logger.info(f"📝 Edited YAML saved: {edited_path.name}")
        return edited_path
    except Exception as e:
        logger.error(f"❌ YAML Process Error: {e}")
        st.stop()

# --- ReadMe Logic ---
def check_and_create_version(version, api_key, base_url, logger, create_if_missing=False):
    if not api_key: return
    headers = {"Authorization": f"Basic {api_key}", "Accept": "application/json"}
    logger.info(f"🔎 Checking version '{version}'...")
    try:
        res = requests.get(f"{base_url}/version", headers=headers)
        if res.status_code == 200:
            if any(v["version"] == version for v in res.json()):
                logger.info(f"✅ Version '{version}' exists.")
                return
        if create_if_missing:
            logger.info(f"⚠️ Creating version '{version}'...")
            fork_from = res.json()[0]['version'] if res.json() else "latest"
            requests.post(f"{base_url}/version", headers=headers, json={"version": version, "is_stable": False, "from": fork_from})
    except Exception as e:
        logger.error(f"❌ Version check failed: {e}")

def get_api_id(api_name, version, api_key, base_url, logger):
    if not api_key: return None, None
    headers = {"Authorization": f"Basic {api_key}", "Accept": "application/json", "x-readme-version": version}
    
    try:
        logger.info(f"🔎 Looking for ID for Title: '{api_name}'")
        def tokenize(text): return set(re.findall(r'\w+', text.lower()))
        target_tokens = tokenize(api_name)
        
        res = requests.get(f"{base_url}/api-specification", headers=headers, params={"perPage": 100})
        if res.status_code == 200:
            apis = res.json()
            for api in apis:
                if api["title"] == api_name:
                    logger.info(f"✅ Exact Match: {api['_id']}")
                    return api["_id"], api["title"]
            for api in apis:
                if target_tokens == tokenize(api["title"]):
                    logger.info(f"✨ Smart Match: '{api['title']}' (ID: {api['_id']})")
                    return api["_id"], api["title"]
            logger.warning(f"⚠️ No match found for '{api_name}'")
        else:
            logger.error(f"❌ API Error: {res.status_code}")
    except Exception as e:
        logger.error(f"❌ ID Lookup Error: {e}")
    return None, None

def create_new_api_via_requests(file_path, version, api_key, base_url, logger):
    """
    Directly uploads a new spec to ReadMe via requests to bypass CLI prompts.
    """
    logger.info("📤 Creating NEW API definition directly via API...")
    headers = {"Authorization": f"Basic {api_key}", "x-readme-version": version}
    
    try:
        with open(file_path, 'rb') as f:
            files = {'spec': (file_path.name, f)}
            res = requests.post(f"{base_url}/api-specification", headers=headers, files=files)
            
        if res.status_code in [200, 201]:
            new_id = res.json().get("_id")
            logger.info(f"✅ Successfully Created! ID: {new_id}")
            return new_id
        else:
            logger.error(f"❌ API Upload Failed: {res.text}")
            return None
    except Exception as e:
        logger.error(f"❌ Upload Exception: {e}")
        return None

def clear_creds():
    for k in ['readme_key', 'git_user', 'git_token', 'logs']:
        if k in st.session_state: del st.session_state[k]
    st.session_state.logs = []

def clear_logs():
    st.session_state.logs = []

# --- Main ---
def main():
    st.sidebar.title("⚙️ Configuration")
    
    for k in ['readme_key', 'gemini_key', 'git_user', 'git_token', 'repo_url', 'last_edited_file', 'corrected_file']:
        if k not in st.session_state: st.session_state[k] = "" if 'file' not in k else None
    if 'ai_model' not in st.session_state: st.session_state.ai_model = "gemini-2.5-flash"

    readme_key = st.sidebar.text_input("ReadMe API Key", key="readme_key", type="password")
    with st.sidebar.expander("🤖 AI Config", expanded=True):
        gemini_key = st.text_input("Gemini API Key", key="gemini_key", type="password")
        ai_model = st.text_input("Model Name", key="ai_model")

    st.sidebar.subheader("Git Config")
    repo_path = st.sidebar.text_input("Local Clone Path", value="./cloned_repo")
    if st.sidebar.button("🗑️ Delete Repo"):
        s, m = delete_repo(repo_path)
        if s: st.sidebar.success(m)
        else: st.sidebar.warning(m)
        
    repo_url = st.sidebar.text_input("Git HTTPS URL", key="repo_url")
    branch_name = st.sidebar.text_input("Branch Name", value="main")
    git_user = st.sidebar.text_input("Git User", key="git_user")
    git_token = st.sidebar.text_input("Git Token", key="git_token", type="password")
    st.sidebar.button("🔒 Clear Credentials", on_click=clear_creds)

    st.sidebar.subheader("Paths")
    spec_rel = st.sidebar.text_input("Main Specs Path", value="specs")
    sec_rel = st.sidebar.text_input("Secondary Path (Opt)", value="")
    dep_in = st.sidebar.text_input("Dependency Folders", value="common")
    deps = [x.strip() for x in dep_in.split(",")]
    domain = st.sidebar.text_input("API Domain", value="api.example.com")

    abs_spec = Path(repo_path) / spec_rel
    paths = {"repo": repo_path, "specs": abs_spec}
    if sec_rel: paths["secondary"] = Path(repo_path) / sec_rel
    workspace_dir = "./temp_workspace" 

    st.title("🚀 OpenAPI Spec Validator")
    
    c1, c2 = st.columns(2)
    with c1:
        files = []
        if abs_spec.exists(): files.extend([f.stem for f in abs_spec.glob("*.yaml")])
        if "secondary" in paths and paths["secondary"].exists(): files.extend([f.stem for f in paths["secondary"].glob("*.yaml")])
        files = sorted(list(set(files)))
        selected_file = st.selectbox("Select File", files) if files else st.text_input("Filename", "audit")

    with c2: version = st.text_input("API Version", "1.0")

    st.markdown("### 🚀 Settings")
    ch1, ch2, ch3 = st.columns(3)
    with ch1: use_sw = st.checkbox("Swagger CLI", True)
    with ch2: use_re = st.checkbox("Redocly CLI", True)
    with ch3: use_rd = st.checkbox("ReadMe CLI", False)
    
    st.markdown("---")
    u_opts = ["Original (Edited)"]
    if st.session_state.corrected_file: u_opts.append("AI Corrected")
    
    cs1, cs2 = st.columns([1, 2])
    with cs1: u_choice = st.radio("Upload:", u_opts, horizontal=True)
    with cs2:
        cb1, cb2 = st.columns(2)
        b_val = cb1.button("🔍 Validate", use_container_width=True)
        b_up = cb2.button(f"🚀 Upload: {u_choice}", type="primary", use_container_width=True)

    st.markdown("### 📜 Logs")
    log_con = st.empty()
    if st.session_state.logs: log_con.code("\n".join(st.session_state.logs), language="text")

    cd1, cd2, cd3 = st.columns([1, 1, 3])
    with cd1:
        dl_ph = st.empty()
        if st.session_state.logs:
            dl_ph.download_button("📥 Logs", "\n".join(st.session_state.logs), "log.txt", key=f"dl_{len(st.session_state.logs)}")

    if b_val or b_up:
        st.session_state.logs = []
        st.session_state.last_edited_file = None
        st.session_state.corrected_file = None
        
        logger = logging.getLogger("st_log")
        logger.setLevel(logging.INFO)
        if logger.handlers: logger.handlers = []
        handler = StreamlitLogHandler(log_con, dl_ph)
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(handler)

        has_key = validate_env(readme_key, required=bool(b_up))
        npx = get_npx_path()
        base_url = "https://dash.readme.com/api/v1"

        setup_git_repo(repo_url, repo_path, git_token, git_user, branch_name, logger)
        logger.info("📂 Preparing workspace...")
        final_yaml = prepare_files(selected_file, paths, workspace_dir, deps, logger)
        
        abs_workspace_path = Path(workspace_dir).resolve()
        
        if has_key: check_and_create_version(version, readme_key, base_url, logger, bool(b_up))
        
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
            logger.info("🔍 Running Swagger...")
            if run_command([npx, "--yes", "swagger-cli", "validate", target.name], logger, cwd=abs_workspace_path) != 0: fail = True
        
        if do_r:
            logger.info("🔍 Running Redocly...")
            if run_command([npx, "--yes", "@redocly/cli@1.25.0", "lint", target.name], logger, cwd=abs_workspace_path) != 0: fail = True
            
        if do_rm and has_key:
            logger.info("🔍 Running ReadMe CLI (v8)...")
            if run_command([npx, "--yes", "rdme@8", "openapi:validate", target.name], logger, cwd=abs_workspace_path) != 0: fail = True

        if fail:
            logger.error("❌ Validation Failed.")
            st.error("Errors found.")
        else:
            logger.info("✅ Validated.")
            if b_up:
                logger.info("🚀 Uploading...")
                with open(target, "r") as f:
                    ydata = yaml.safe_load(f)
                    ytitle = ydata.get("info", {}).get("title", "")
                
                api_id, matched_title = get_api_id(ytitle, version, readme_key, base_url, logger)
                
                if api_id and matched_title and matched_title != ytitle:
                    logger.info(f"🔧 Correcting Title: '{ytitle}' -> '{matched_title}'")
                    ydata["info"]["title"] = matched_title
                    with open(target, "w") as f: yaml.dump(ydata, f, sort_keys=False)

                if api_id:
                    # Case 1: Existing API -> Update via CLI
                    cmd = [npx, "--yes", "rdme@8", "openapi", target.name, "--useSpecVersion", "--version", version, "--id", api_id, "--key", readme_key]
                    if run_command(cmd, logger, cwd=abs_workspace_path) == 0:
                        logger.info("🎉 Updated Existing API!")
                        st.success("Success!")
                    else:
                        logger.error("❌ Upload Failed.")
                else:
                    # Case 2: New API -> Create via Bundle + Request
                    logger.warning("⚠️ No ID found. Treating as NEW API.")
                    logger.info("📦 Bundling references...")
                    bundled_name = f"{target.stem}_bundled.yaml"
                    if run_command([npx, "--yes", "swagger-cli", "bundle", target.name, "-o", bundled_name, "-t", "yaml"], logger, cwd=abs_workspace_path) == 0:
                        bundled_path = abs_workspace_path / bundled_name
                        create_new_api_via_requests(bundled_path, version, readme_key, base_url, logger)
                        st.success("Success!")
                    else:
                        logger.error("❌ Bundling failed.")

            else:
                st.success("Done.")

    with cd2:
        if st.session_state.last_edited_file:
            p = Path(st.session_state.last_edited_file)
            if p.exists():
                with open(p, "r") as f: st.download_button("📄 Edited YAML", f.read(), p.name, "application/x-yaml")

    with cd3:
        if st.session_state.logs: st.button("🗑️ Clear Logs", on_click=clear_logs)

    if st.session_state.logs and gemini_key:
        st.markdown("### 🤖 AI Helper")
        ca1, ca2 = st.columns(2)
        if ca1.button("🧐 Analyze"):
            with st.spinner("Thinking..."):
                an = analyze_errors_with_ai("\n".join(st.session_state.logs), gemini_key, ai_model)
                if an: st.markdown(an)
        if ca2.button("✨ Auto-Fix"):
            if st.session_state.last_edited_file:
                with st.spinner("Fixing..."):
                    fix = apply_ai_fixes(st.session_state.last_edited_file, "\n".join(st.session_state.logs), gemini_key, ai_model)
                    if fix:
                        op = Path(st.session_state.last_edited_file)
                        cp = op.parent / (op.stem.replace("_edited", "") + "_corrected.yaml")
                        with open(cp, "w") as f: f.write(fix)
                        st.session_state.corrected_file = str(cp)
                        st.success("Fixed! Choose 'AI Corrected' above.")
                        st.rerun()

    if st.session_state.corrected_file:
        cp = Path(st.session_state.corrected_file)
        if cp.exists():
            with open(cp, "r") as f: st.download_button("✨ Corrected YAML", f.read(), cp.name, "application/x-yaml")

if __name__ == "__main__":
    main()
