import streamlit as st
import yaml
import subprocess
import shutil
import requests
import os
import logging
import sys
import urllib.parse
import base64
import re
import json
import platform
from pathlib import Path
from google import genai
from google.genai import types

# Page Config
st.set_page_config(
    page_title="ReadMe.io OpenAPI Spec Validator v1.0",
    page_icon="📘",
    layout="wide"
)

# --- Initialize Session State for Logs ---
if 'logs' not in st.session_state:
    st.session_state.logs = []

# --- Custom Logging Handler for Streamlit ---
class StreamlitLogHandler(logging.Handler):
    def __init__(self, container, download_placeholder=None):
        super().__init__()
        self.container = container
        self.download_placeholder = download_placeholder

    def emit(self, record):
        msg = self.format(record)
        st.session_state.logs.append(msg)
        self.container.code("\n".join(st.session_state.logs), language="text")
        
        # Update download button dynamically if placeholder exists
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

def run_command(command_list, log_logger):
    try:
        cmd_str = " ".join(command_list)
        log_logger.info(f"Running: {cmd_str}")
        
        process = subprocess.Popen(
            command_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8'
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

# --- AI Analysis Logic ---
def analyze_errors_with_ai(log_content, api_key, model_name):
    if not api_key: return None
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""
        You are an expert OpenAPI Validator. Analyze the following log output.
        Identify specific YAML errors and provide actionable solutions (code snippets).
        Logs:
        {log_content}
        """
        response = client.models.generate_content(model=model_name, contents=[prompt])
        return response.text
    except Exception as e:
        return f"Exception calling AI: {e}"

def apply_ai_fixes(original_path, log_content, api_key, model_name):
    if not api_key: return None
    try:
        with open(original_path, 'r') as f: yaml_content = f.read()
        client = genai.Client(api_key=api_key)
        prompt = f"""
        You are an expert OpenAPI Repair Agent.
        Fix the errors in the logs for this YAML file.
        PRESERVE 'x-readme', 'servers', and 'info'.
        Return ONLY the valid YAML code (wrapped in ```yaml blocks if needed).
        Logs: {log_content}
        YAML: {yaml_content}
        """
        response = client.models.generate_content(model=model_name, contents=[prompt])
        cleaned_text = response.text
        match = re.search(r'```yaml\n(.*?)\n```', cleaned_text, re.DOTALL)
        if match: return match.group(1)
        return cleaned_text
    except Exception as e:
        return None

# --- Git Logic ---
def setup_git_repo(repo_url, repo_dir, git_token, git_username, branch_name, logger):
    logger.info(f"🚀 Starting Git Operation for branch: {branch_name}...")
    repo_path = Path(repo_dir)
    repo_url = repo_url.strip().strip('"').strip("'")
    
    # Handle GitHub URL logic
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
        logger.error(f"❌ URL Construction Failed: {e}")
        st.stop()

    clean_env = os.environ.copy()
    clean_env["GIT_TERMINAL_PROMPT"] = "0"
    
    if not repo_path.exists():
        logger.info(f"⬇️ Cloning branch '{branch_name}'...")
        try:
            cmd = ["git", "clone", "--depth", "1", "--branch", branch_name, auth_repo_url, str(repo_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, env=clean_env)
            if result.returncode != 0:
                logger.error(f"❌ Git Clone Failed: {result.stderr}")
                st.error("Git Clone Failed. Check branch name and permissions.")
                st.stop()
            logger.info("✅ Repo cloned successfully.")
        except Exception as e:
            logger.error(f"❌ System Error: {e}")
            st.stop()
    else:
        logger.info(f"🔄 Switching/Updating to branch '{branch_name}'...")
        try:
            subprocess.run(["git", "-C", str(repo_path), "remote", "set-url", "origin", auth_repo_url], check=True, capture_output=True, env=clean_env)
            subprocess.run(["git", "-C", str(repo_path), "fetch", "origin", branch_name], check=True, capture_output=True, env=clean_env)
            subprocess.run(["git", "-C", str(repo_path), "checkout", branch_name], check=True, capture_output=True, env=clean_env)
            subprocess.run(["git", "-C", str(repo_path), "pull", "origin", branch_name], check=True, capture_output=True, env=clean_env)
            logger.info(f"✅ Successfully switched to '{branch_name}'.")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Git Operation failed: {e}")

def delete_repo(repo_dir):
    path = Path(repo_dir)
    if path.exists():
        try:
            shutil.rmtree(path)
            return True, "Deleted successfully."
        except Exception as e:
            return False, f"Error deleting: {e}"
    return False, "Path does not exist."

# --- File Operations (Generic) ---
def prepare_files(filename, paths, workspace, dependency_list, logger):
    source = None
    main_candidate = Path(paths['specs']) / f"{filename}.yaml"
    
    if main_candidate.exists():
        source = main_candidate
    elif paths.get('secondary') and (Path(paths['secondary']) / f"{filename}.yaml").exists():
        source = Path(paths['secondary']) / f"{filename}.yaml"

    if not source:
        logger.error(f"❌ Source file '{filename}.yaml' not found.")
        st.stop()

    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    destination = workspace_path / source.name
    shutil.copy(source, destination)
    logger.info(f"📂 Copied main YAML to workspace: {destination.name}")

    for folder in dependency_list:
        clean_folder = folder.strip()
        if not clean_folder: continue
        src_folder = Path(paths['specs']) / clean_folder
        dest_folder = workspace_path / clean_folder
        if src_folder.exists():
            if dest_folder.exists(): shutil.rmtree(dest_folder)
            shutil.copytree(src_folder, dest_folder)
            logger.info(f"📂 Copied dependency: {clean_folder}")

    return destination

def process_yaml_content(file_path, version, api_domain, logger):
    logger.info("🛠️ Injecting x-readme extensions...")
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
        logger.info(f"📝 Edited YAML saved to: {edited_path.name}")
        return edited_path
    except Exception as e:
        logger.error(f"❌ Error processing YAML: {e}")
        st.stop()

# --- ReadMe API Logic (UPDATED) ---
def check_and_create_version(version, api_key, base_url, logger, create_if_missing=False):
    if not api_key: return
    headers = {"Authorization": f"Basic {api_key}", "Accept": "application/json"}
    logger.info(f"🔎 Checking version '{version}' on ReadMe...")
    try:
        response = requests.get(f"{base_url}/version", headers=headers)
        if response.status_code == 200:
            if any(v["version"] == version for v in response.json()):
                logger.info(f"✅ Version '{version}' exists.")
                return
        
        if create_if_missing:
            logger.info(f"⚠️ Version '{version}' not found. Creating it...")
            fork_target = response.json()[0]['version'] if response.json() else "latest"
            payload = {"version": version, "is_stable": False, "from": fork_target}
            requests.post(f"{base_url}/version", headers=headers, json=payload)
    except Exception as e:
        logger.error(f"❌ Version check failed: {e}")

def get_api_id(api_name, version, api_key, base_url, logger):
    """
    Retrieves the API Definition ID by matching the Title.
    Replicates exact logic from update_openapi_and_upload.py but with better logging.
    """
    if not api_key: return None
    
    # Header logic matches original script
    headers = {
        "Authorization": f"Basic {api_key}", 
        "Accept": "application/json", 
        "x-readme-version": version
    }
    
    try:
        logger.info(f"🔎 Looking for API ID for Title: '{api_name}' (Version: {version})")
        
        # Use perPage=100 as per original script
        response = requests.get(f"{base_url}/api-specification", headers=headers, params={"perPage": 100})
        
        if response.status_code == 200:
            apis = response.json()
            
            # 1. Exact Match (Original Script Logic)
            for api in apis:
                if api["title"] == api_name:
                    logger.info(f"✅ MATCH FOUND! ID: {api['_id']}")
                    return api["_id"]
            
            # 2. Debugging Info (Why it failed)
            found_titles = [a.get("title", "Unknown") for a in apis]
            logger.warning(f"⚠️ No exact match for '{api_name}'. Found these titles: {found_titles}")
            
            # 3. Fallback: Case-Insensitive Match (Improvement over original script)
            for api in apis:
                if api["title"].strip().lower() == api_name.strip().lower():
                    logger.info(f"⚠️ Found case-insensitive match: {api['_id']} (Title: {api['title']})")
                    return api["_id"]
                    
        else:
            logger.error(f"❌ ReadMe API Error: {response.status_code} - {response.text}")

    except Exception as e:
        logger.error(f"❌ Exception looking up API ID: {e}")
    
    return None

def clear_credentials():
    st.session_state.readme_key = ""
    st.session_state.git_user = ""
    st.session_state.git_token = ""
    st.session_state.logs = []

def clear_logs():
    st.session_state.logs = []

# --- Main UI ---
def main():
    st.sidebar.title("⚙️ Configuration")
    
    # Session State Init
    for key in ['readme_key', 'gemini_key', 'git_user', 'git_token', 'repo_url', 'last_edited_file', 'corrected_file']:
        if key not in st.session_state: st.session_state[key] = "" if key != 'last_edited_file' and key != 'corrected_file' else None
    if 'ai_model' not in st.session_state: st.session_state.ai_model = "gemini-2.5-pro"

    readme_key = st.sidebar.text_input("ReadMe API Key", key="readme_key", type="password", help="Required for Upload or ReadMe Validation")
    
    with st.sidebar.expander("🤖 AI Configuration", expanded=True):
        gemini_key = st.text_input("Gemini API Key", key="gemini_key", type="password", help="Required for AI Analysis")
        ai_model = st.text_input("Model Name", key="ai_model")
    
    st.sidebar.subheader("Git Repo Config")
    repo_path = st.sidebar.text_input("Local Clone Path", value="./cloned_repo")
    
    if st.sidebar.button("🗑️ Reset / Delete Cloned Repo"):
        success, msg = delete_repo(repo_path)
        if success: st.sidebar.success(msg)
        else: st.sidebar.warning(msg)
    
    repo_url = st.sidebar.text_input("Git Repo HTTPS URL", key="repo_url")
    branch_name = st.sidebar.text_input("Branch Name", value="main")
    git_user = st.sidebar.text_input("Git Username", key="git_user", type="password")
    git_token = st.sidebar.text_input("Git Token/PAT", key="git_token", type="password")
    st.sidebar.button("🔒 Clear Credentials", on_click=clear_credentials)

    # Internal Paths
    st.sidebar.subheader("Internal Paths & Settings")
    spec_rel_path = st.sidebar.text_input("Main Specs Path", value="specs") 
    secondary_rel_path = st.sidebar.text_input("Secondary Specs Path (Optional)", value="")
    dep_input = st.sidebar.text_input("Dependency Folders", value="common")
    dependency_list = [x.strip() for x in dep_input.split(",")]
    api_domain = st.sidebar.text_input("API Base Domain", value="api.example.com")

    # Path Calc
    abs_spec_path = Path(repo_path) / spec_rel_path
    paths = {"repo": repo_path, "specs": abs_spec_path}
    if secondary_rel_path: paths["secondary"] = Path(repo_path) / secondary_rel_path
    workspace_dir = "./temp_workspace"

    st.title("🚀 OpenAPI Spec Validator")
    
    col1, col2 = st.columns(2)
    with col1:
        files = []
        if abs_spec_path.exists(): files.extend([f.stem for f in abs_spec_path.glob("*.yaml")])
        if "secondary" in paths and paths["secondary"].exists(): files.extend([f.stem for f in paths["secondary"].glob("*.yaml")])
        files = sorted(list(set(files)))
        
        if files: selected_file = st.selectbox("Select OpenAPI File", files)
        else: selected_file = st.text_input("Enter Filename (e.g. 'audit')", "audit")

    with col2:
        version = st.text_input("API Version", "1.0")

    st.markdown("### 🚀 Validation Settings")
    c_c1, c_c2, c_c3 = st.columns(3)
    with c_c1: use_swagger = st.checkbox("Swagger CLI", value=True)
    with c_c2: use_redocly = st.checkbox("Redocly CLI", value=True)
    with c_c3: use_readme = st.checkbox("ReadMe CLI", value=False)
    
    st.markdown("---")
    
    # Upload Options
    upload_options = ["Original (Edited)"]
    if st.session_state.corrected_file: upload_options.append("AI Corrected")
    
    c_sel, c_act = st.columns([1, 2])
    with c_sel: upload_choice = st.radio("File to Upload:", upload_options, horizontal=True)
    with c_act:
        c_b1, c_b2 = st.columns(2)
        btn_validate = c_b1.button("🔍 Validate Selected", use_container_width=True)
        btn_upload = c_b2.button(f"🚀 Upload: {upload_choice}", type="primary", use_container_width=True)

    # Logging UI
    st.markdown("### 📜 Execution Logs")
    log_container = st.empty()
    if st.session_state.logs: log_container.code("\n".join(st.session_state.logs), language="text")

    c_d1, c_d2, c_d3 = st.columns([1, 1, 3])
    with c_d1:
        dl_ph = st.empty()
        if st.session_state.logs:
            dl_ph.download_button("📥 Download Logs", "\n".join(st.session_state.logs), "openapi_upload.log", "text/plain", key=f"dl_{len(st.session_state.logs)}")

    # Execution Logic
    if btn_validate or btn_upload:
        st.session_state.logs = []
        st.session_state.last_edited_file = None
        st.session_state.corrected_file = None 
        
        logger = logging.getLogger("streamlit_logger")
        logger.setLevel(logging.INFO)
        if logger.handlers: logger.handlers = []
        handler = StreamlitLogHandler(log_container, dl_ph)
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(handler)

        has_key = validate_env(readme_key, required=bool(btn_upload))
        npx = get_npx_path()
        
        setup_git_repo(repo_url, repo_path, git_token, git_user, branch_name, logger)
        logger.info("📂 Preparing workspace...")
        final_yaml = prepare_files(selected_file, paths, workspace_dir, dependency_list, logger)

        if has_key: check_and_create_version(version, readme_key, "[https://dash.readme.com/api/v1](https://dash.readme.com/api/v1)", logger, create_if_missing=bool(btn_upload))

        edited_file = process_yaml_content(final_yaml, version, api_domain, logger)
        st.session_state.last_edited_file = str(edited_file)

        target_file = edited_file
        if btn_upload and upload_choice == "AI Corrected" and st.session_state.corrected_file:
             target_file = Path(st.session_state.corrected_file)

        do_sw = True if btn_upload else use_swagger
        do_re = False if btn_upload else use_redocly
        do_rd = True if btn_upload else use_readme
        failed = False
        
        if do_sw:
            logger.info("🔍 Running Swagger CLI...")
            if run_command([npx, "--yes", "swagger-cli", "validate", str(target_file)], logger) != 0: failed = True
        
        if do_re:
            logger.info("🔍 Running Redocly CLI...")
            if run_command([npx, "--yes", "@redocly/cli@1.25.0", "lint", str(target_file)], logger) != 0: failed = True
            
        if do_rd and has_key:
            logger.info("🔍 Running ReadMe CLI (v9)...")
            if run_command([npx, "--yes", "rdme@9", "openapi", "validate", str(target_file)], logger) != 0: failed = True

        if failed:
            logger.error("❌ Validation failed.")
            st.error("Validation Failed.")
        else:
            logger.info("✅ Selected validations passed.")
            if btn_upload:
                logger.info("🚀 Uploading to ReadMe...")
                with open(target_file, "r") as f:
                    title = yaml.safe_load(f).get("info", {}).get("title", "")
                
                # REPLICATED LOGIC FROM ORIGINAL SCRIPT
                api_id = get_api_id(title, version, readme_key, "[https://dash.readme.com/api/v1](https://dash.readme.com/api/v1)", logger)
                
                cmd = [npx, "--yes", "rdme@9", "openapi", str(target_file), "--useSpecVersion", "--key", readme_key, "--version", version]
                if api_id: cmd.extend(["--id", api_id])
                
                if run_command(cmd, logger) == 0:
                    logger.info("🎉 Upload Successful!")
                    st.success(f"Done! Uploaded: {upload_choice}")
                else:
                    logger.error("❌ Upload failed.")
            else:
                st.success("Validation Check Complete.")

    # Post-Execution Buttons
    with c_d2:
        if st.session_state.last_edited_file:
            p = Path(st.session_state.last_edited_file)
            if p.exists():
                with open(p, "r") as f: st.download_button("📄 Download Edited YAML", f.read(), p.name, "application/x-yaml")

    with c_d3:
        if st.session_state.logs: st.button("🗑️ Clear Logs", on_click=clear_logs)

    # AI Section
    if st.session_state.logs and gemini_key:
        st.markdown("### 🤖 AI Assistance")
        ca1, ca2 = st.columns(2)
        if ca1.button("🧐 Analyze Errors"):
            with st.spinner("Analyzing..."):
                an = analyze_errors_with_ai("\n".join(st.session_state.logs), gemini_key, ai_model)
                if an: st.markdown(an)
        if ca2.button("✨ Attempt Auto-Fix"):
            if st.session_state.last_edited_file:
                with st.spinner("Generating fix..."):
                    fix = apply_ai_fixes(st.session_state.last_edited_file, "\n".join(st.session_state.logs), gemini_key, ai_model)
                    if fix:
                        op = Path(st.session_state.last_edited_file)
                        cp = op.parent / (op.stem.replace("_edited", "") + "_corrected.yaml")
                        with open(cp, "w") as f: f.write(fix)
                        st.session_state.corrected_file = str(cp)
                        st.success("✅ Fix generated!")
                        st.rerun()
    
    if st.session_state.corrected_file:
        cp = Path(st.session_state.corrected_file)
        if cp.exists():
            with open(cp, "r") as f: st.download_button("✨ Download Corrected YAML", f.read(), cp.name, "application/x-yaml")
    elif st.session_state.logs and not gemini_key:
        st.info("💡 Enter Gemini API Key to use AI tools.")

if __name__ == "__main__":
    main()
