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
    page_title="ReadMe.io OpenAPI Manager v2.26",
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

# --- AI Analysis Logic (Google Gen AI SDK) ---
def analyze_errors_with_ai(log_content, api_key, model_name):
    """Sends logs to Google Gen AI for analysis."""
    if not api_key:
        return None
    
    try:
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
        You are an expert OpenAPI Validator. Analyze the following log output from a CI/CD pipeline. 
        It contains errors from Swagger CLI, Redocly CLI, or ReadMe CLI.
        
        Identify the specific YAML errors (like trailing slashes, missing references, schema issues) 
        and provide actionable solutions (code snippets) for the user to fix their OpenAPI YAML file.
        
        Keep it concise, professional, and use Markdown formatting.
        
        Logs:
        {log_content}
        """
        
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt]
        )
        
        return response.text
    except Exception as e:
        return f"Exception calling AI: {e}"

# --- Git Logic ---

def setup_git_repo(repo_url, repo_dir, git_token, git_username, logger):
    logger.info("🚀 Starting Git Operation...")
    
    repo_path = Path(repo_dir)
    repo_url = repo_url.strip().strip('"').strip("'")
    git_username = git_username.strip().strip('"').strip("'")
    git_token = git_token.strip().strip('"').strip("'")

    if repo_url.count("https://") > 1:
        match = re.search(r"(https://github\.com/.*)$", repo_url)
        if match:
            repo_url = match.group(1)

    try:
        parsed = urllib.parse.urlparse(repo_url)
        safe_user = urllib.parse.quote(git_username, safe='')
        safe_token = urllib.parse.quote(git_token, safe='')
        
        if "@" in parsed.netloc:
            clean_netloc = parsed.netloc.split("@")[-1]
        else:
            clean_netloc = parsed.netloc

        auth_netloc = f"{safe_user}:{safe_token}@{clean_netloc}"
        auth_repo_url = urllib.parse.urlunparse((
            parsed.scheme, auth_netloc, parsed.path, parsed.params, parsed.query, parsed.fragment
        ))
        masked_netloc = f"****:***@{clean_netloc}"
        masked_repo_url = urllib.parse.urlunparse((
            parsed.scheme, masked_netloc, parsed.path, parsed.params, parsed.query, parsed.fragment
        ))
        
    except Exception as e:
        logger.error(f"❌ URL Construction Failed: {e}")
        st.stop()

    clean_env = os.environ.copy()
    null_path = "NUL" if platform.system() == "Windows" else "/dev/null"
    clean_env["GIT_CONFIG_GLOBAL"] = null_path
    clean_env["GIT_CONFIG_SYSTEM"] = null_path
    clean_env["GIT_TERMINAL_PROMPT"] = "0"

    if not repo_path.exists():
        logger.info(f"⬇️ Cloning from: {masked_repo_url}")
        git_args = ["-c", "core.askPass=echo"] 
        try:
            cmd = ["git"] + git_args + ["clone", "--depth", "1", auth_repo_url, str(repo_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, env=clean_env)
            
            if result.returncode != 0:
                sso_match = re.search(r"(https://github\.com/orgs/[^/]+/sso\?authorization_request=[^\s]+)", result.stderr)
                if sso_match:
                    sso_url = sso_match.group(1)
                    logger.error("❌ SSO AUTHORIZATION REQUIRED")
                    st.error("🚨 Organization requires SAML SSO Authorization.")
                    st.markdown(f"👉 **[Click here to Authorize your Token]({sso_url})**")
                    st.stop()
                elif "403" in result.stderr:
                    st.error("🚨 Authentication Failed (403).")
                    st.info("Ensure your Token has 'repo' scope and SSO is configured.")
                    st.stop()
                else:
                    safe_err = result.stderr.replace(git_token, "***").replace(git_username, "****")
                    logger.error(f"❌ Git Output:\n{safe_err}")
                    st.error("Git Clone Failed.")
                    st.stop()
            logger.info("✅ Repo cloned successfully.")
        except Exception as e:
            logger.error(f"❌ System Error: {e}")
            st.stop()
    else:
        logger.info(f"🔄 Pulling latest changes...")
        try:
            subprocess.run(["git", "-C", str(repo_path), "remote", "set-url", "origin", auth_repo_url], 
                         check=True, capture_output=True, env=clean_env)
            cmd = ["git", "-C", str(repo_path), "pull"]
            subprocess.run(cmd, check=True, capture_output=True, env=clean_env)
            logger.info("✅ Repo updated successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Pull failed: {e}")
            logger.warning("⚠️ Continuing with existing files...")

def delete_repo(repo_dir):
    path = Path(repo_dir)
    if path.exists():
        try:
            shutil.rmtree(path)
            return True, "Deleted successfully."
        except Exception as e:
            return False, f"Error deleting: {e}"
    return False, "Path does not exist."

# --- File Operations ---

def prepare_files(filename, paths, workspace, logger):
    if filename in ["field", "field_value"]:
        source = Path(paths['logical']) / f"{filename}.yaml"
    else:
        source = Path(paths['specs']) / f"{filename}.yaml"

    if not source.exists():
        logger.error(f"❌ Source file not found: {source}")
        logger.info(f"ℹ️ Searched in: {source.parent}")
        st.stop()

    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    destination = workspace_path / source.name
    shutil.copy(source, destination)
    logger.info(f"📂 Copied main YAML to workspace: {destination.name}")

    for folder in ["common", "data_products"]:
        src_folder = Path(paths['specs']) / folder
        dest_folder = workspace_path / folder
        if src_folder.exists():
            if dest_folder.exists():
                shutil.rmtree(dest_folder)
            shutil.copytree(src_folder, dest_folder)
            logger.info(f"📂 Copied dependency folder: {folder}")
        else:
            logger.warning(f"⚠️ Dependency folder not found: {src_folder}")

    return destination

# --- ReadMe API Logic ---

def check_and_create_version(version, api_key, base_url, logger, create_if_missing=False):
    if not api_key:
        return

    headers = {"Authorization": f"Basic {api_key}", "Accept": "application/json"}
    logger.info(f"🔎 Checking version '{version}' on ReadMe...")
    try:
        response = requests.get(f"{base_url}/version", headers=headers)
        if response.status_code != 200:
            logger.error(f"❌ Failed to fetch versions: {response.text}")
            return
        
        versions = response.json()
        if any(v["version"] == version for v in versions):
            logger.info(f"✅ Version '{version}' exists.")
            return

        if not create_if_missing:
            logger.warning(f"⚠️ Version '{version}' not found on ReadMe. (Skipping creation during validation).")
            return

        logger.info(f"⚠️ Version '{version}' not found. Creating it...")
        fork_target = versions[0]['version'] if versions else "latest"
        payload = {"version": version, "is_stable": False, "from": fork_target}
        create_response = requests.post(f"{base_url}/version", headers=headers, json=payload)
        
        if create_response.status_code == 201:
            logger.info(f"✅ Version '{version}' created successfully.")
        else:
            logger.error(f"❌ Failed to create version: {create_response.text}")
    except Exception as e:
        logger.error(f"❌ Network error checking version: {e}")

def process_yaml_content(file_path, version, logger):
    logger.info("🛠️ Injecting x-readme extensions and updating server info...")
    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        if "openapi" in data:
            pos = list(data.keys()).index("openapi")
            items = list(data.items())
            items.insert(pos + 1, ("x-readme", {"explorer-enabled": False}))
            data = dict(items)
        
        data["info"]["version"] = version
        if "servers" not in data or not data["servers"]:
            data["servers"] = [{"url": "https://alation_domain", "variables": {}}]

        if "variables" not in data["servers"][0]:
            data["servers"][0]["variables"] = {}
            
        data["servers"][0]["variables"]["base-url"] = {"default": "alation_domain"}
        data["servers"][0]["variables"]["protocol"] = {"default": "https"}

        edited_path = file_path.parent / (file_path.stem + "_edited.yaml")
        with open(edited_path, "w") as f:
            yaml.dump(data, f, sort_keys=False)
        logger.info(f"📝 Edited YAML saved to: {edited_path.name}")
        return edited_path
    except Exception as e:
        logger.error(f"❌ Error processing YAML: {e}")
        st.stop()

def get_api_id(api_name, version, api_key, base_url, logger):
    if not api_key: return None
    
    headers = {"Authorization": f"Basic {api_key}", "Accept": "application/json", "x-readme-version": version}
    try:
        response = requests.get(f"{base_url}/api-specification", headers=headers, params={"perPage": 100})
        if response.status_code == 200:
            for api in response.json():
                if api["title"] == api_name:
                    return api["_id"]
    except Exception:
        pass
    return None

# --- CALLBACK FUNCTIONS ---
def clear_credentials():
    """Clears session state variables."""
    st.session_state.readme_key = ""
    st.session_state.git_user = ""
    st.session_state.git_token = ""
    st.session_state.gemini_key = ""
    st.session_state.logs = []

def clear_logs():
    """Clears the log history."""
    st.session_state.logs = []

# --- UI Layout ---

def main():
    st.sidebar.title("⚙️ Configuration")
    
    # --- CREDENTIAL MANAGEMENT ---
    if 'readme_key' not in st.session_state: st.session_state.readme_key = ""
    if 'gemini_key' not in st.session_state: st.session_state.gemini_key = ""
    if 'git_user' not in st.session_state: st.session_state.git_user = ""
    if 'git_token' not in st.session_state: st.session_state.git_token = ""
    if 'repo_url' not in st.session_state: st.session_state.repo_url = "https://github.com/alation/alation.git"
    
    if 'ai_model' not in st.session_state: st.session_state.ai_model = "gemini-2.0-flash"

    readme_key = st.sidebar.text_input("ReadMe API Key", key="readme_key", type="password", help="Required for Upload or ReadMe Validation")
    
    with st.sidebar.expander("🤖 AI Configuration", expanded=True):
        gemini_key = st.text_input("AI/Gemini API Key", key="gemini_key", type="password", help="Required for AI Analysis")
        ai_model = st.text_input("Model Name", key="ai_model", help="e.g., gemini-2.0-flash or gemini-1.5-pro")
    
    st.sidebar.subheader("Git Repo Config")
    default_cloud_path = "./cloned_repo"
    repo_path = st.sidebar.text_input("Local Clone Path", value=default_cloud_path)
    
    if st.sidebar.button("🗑️ Reset / Delete Cloned Repo"):
        success, msg = delete_repo(repo_path)
        if success: st.sidebar.success(msg)
        else: st.sidebar.warning(msg)
    
    repo_url = st.sidebar.text_input("Git Repo URL", key="repo_url")
    git_user = st.sidebar.text_input("Git Username", key="git_user", type="password", help="GitHub Handle")
    git_token = st.sidebar.text_input("Git Token/PAT", key="git_token", type="password", help="Personal Access Token")

    st.sidebar.button("🔒 Clear Credentials", on_click=clear_credentials)

    st.sidebar.subheader("Internal Paths")
    spec_rel_path = st.sidebar.text_input("Specs Path (relative to repo)", value="django/static/swagger/specs")
    abs_spec_path = Path(repo_path) / spec_rel_path
    abs_logical_path = abs_spec_path / "logical_metadata"
    
    paths = {"repo": repo_path, "specs": abs_spec_path, "logical": abs_logical_path}
    workspace_dir = "./temp_workspace"

    st.title("🚀 ReadMe.io Manager v2.26")
    
    col1, col2 = st.columns(2)
    with col1:
        files = []
        if abs_spec_path.exists():
            files = [f.stem for f in abs_spec_path.glob("*.yaml")]
            files += [f.stem for f in abs_logical_path.glob("*.yaml")]
            files = sorted(list(set(files)))
        
        if files:
            selected_file = st.selectbox("Select OpenAPI File", files)
        else:
            selected_file = st.text_input("Enter Filename (e.g. 'audit')", "audit")
            if not abs_spec_path.exists():
                st.warning(f"⚠️ Repo not synced yet. Click 'Start Process' to clone.")

    with col2:
        version = st.text_input("API Version", "1.0")

    st.markdown("### 🚀 Actions")
    
    c1, c2, c3 = st.columns(3)
    btn_swagger = c1.button("Validate Swagger CLI", use_container_width=True)
    btn_redocly = c2.button("Validate Redocly CLI", use_container_width=True)
    btn_readme = c3.button("Validate ReadMe CLI", use_container_width=True, help="Requires ReadMe API Key")
    
    st.markdown("---")
    
    c4, c5 = st.columns(2)
    btn_validate_all = c4.button("🔍 Validate All", use_container_width=True)
    btn_upload = c5.button("🚀 Upload to ReadMe", type="primary", use_container_width=True, help="Validates all and uploads if successful")

    # --- PERSISTENT LOG DISPLAY ---
    st.markdown("### 📜 Execution Logs")
    
    log_container = st.empty()
    if st.session_state.logs:
        log_container.code("\n".join(st.session_state.logs), language="text")

    col_d1, col_d2 = st.columns([1, 4])
    with col_d1:
        download_placeholder = st.empty()
        if st.session_state.logs:
            unique_key = f"dl_btn_persist_{len(st.session_state.logs)}"
            download_placeholder.download_button(
                label="📥 Download Log File",
                data="\n".join(st.session_state.logs),
                file_name="openapi_upload.log",
                mime="text/plain",
                key=unique_key
            )
    with col_d2:
        if st.session_state.logs:
             st.button("🗑️ Clear Logs", on_click=clear_logs)

    # AI Analysis Section (Using Google Gen AI SDK)
    if st.session_state.logs and gemini_key:
        if st.button(f"🤖 Analyze Logs with {ai_model}"):
            with st.spinner("Analyzing errors..."):
                log_text = "\n".join(st.session_state.logs)
                # Use the new SDK logic
                analysis = analyze_errors_with_ai(log_text, gemini_key, ai_model)
                if analysis:
                    st.markdown("### 🤖 AI Fix Suggestion")
                    st.markdown(analysis)
    elif st.session_state.logs and not gemini_key:
        st.info("💡 Enter a Gemini/AI API Key in the sidebar to unlock error analysis.")

    # --- MAIN ACTION LOGIC ---
    action = None
    if btn_swagger: action = 'swagger'
    elif btn_redocly: action = 'redocly'
    elif btn_readme: action = 'readme'
    elif btn_validate_all: action = 'all'
    elif btn_upload: action = 'upload'

    if action:
        st.session_state.logs = [] # Clear old logs
        
        logger = logging.getLogger("streamlit_logger")
        logger.setLevel(logging.INFO)
        if logger.handlers: logger.handlers = []
        
        handler = StreamlitLogHandler(log_container, download_placeholder)
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(handler)

        strict_key_req = True if action == 'upload' else False
        has_key = validate_env(readme_key, required=strict_key_req)
        
        npx_path = get_npx_path()
        if not npx_path:
            logger.error("❌ NodeJS/npx not found.")
            st.stop()

        setup_git_repo(repo_url, repo_path, git_token, git_user, logger)

        logger.info("📂 Preparing workspace...")
        final_yaml_path = prepare_files(selected_file, paths, workspace_dir, logger)

        if has_key:
            create_ver = True if action == 'upload' else False
            check_and_create_version(version, readme_key, "https://dash.readme.com/api/v1", logger, create_if_missing=create_ver)

        edited_file = process_yaml_content(final_yaml_path, version, logger)

        validation_failed = False
        
        if action in ['swagger', 'all', 'upload']:
            logger.info("🔍 Running Swagger CLI...")
            if run_command([npx_path, "--yes", "swagger-cli", "validate", str(edited_file)], logger) != 0: 
                validation_failed = True
        
        if action in ['redocly', 'all', 'upload']:
            logger.info("🔍 Running Redocly CLI (Pinned v1.25.0)...")
            if run_command([npx_path, "--yes", "@redocly/cli@1.25.0", "lint", str(edited_file)], logger) != 0: 
                validation_failed = True
            
        if action in ['readme', 'all', 'upload']:
            if has_key:
                logger.info("🔍 Running ReadMe CLI (Pinned v8)...")
                if run_command([npx_path, "--yes", "rdme@8", "openapi:validate", str(edited_file), "--key", readme_key], logger) != 0: 
                    validation_failed = True
            else:
                logger.warning("⚠️ Skipping ReadMe CLI validation (No API Key provided).")

        if validation_failed:
            logger.error("❌ Validation failed.")
            st.error("Validation Failed.")
            if action == 'upload':
                st.error("Aborting upload due to validation errors.")
        else:
            logger.info("✅ Selected validations passed.")

            if action == 'upload':
                logger.info("🚀 Uploading to ReadMe...")
                with open(edited_file, "r") as f:
                    title = yaml.safe_load(f).get("info", {}).get("title", "")
                
                api_id = get_api_id(title, version, readme_key, "https://dash.readme.com/api/v1", logger)
                cmd = [npx_path, "--yes", "rdme@8", "openapi", str(edited_file), "--useSpecVersion", "--key", readme_key, "--version", version]
                if api_id: cmd.extend(["--id", api_id])
                
                if run_command(cmd, logger) == 0:
                    logger.info("🎉 Upload Successful!")
                    st.success("Done!")
                else:
                    logger.error("❌ Upload failed.")
            else:
                st.success("Validation Check Complete.")

if __name__ == "__main__":
    main()
