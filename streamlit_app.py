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
    page_title="OpenAPI Spec Validator v1.0",
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

def apply_ai_fixes(original_path, log_content, api_key, model_name):
    """
    Sends the YAML and logs to Gemini and asks for a fully corrected YAML file.
    Returns the content of the fixed YAML.
    """
    if not api_key:
        return None
    
    try:
        with open(original_path, 'r') as f:
            yaml_content = f.read()

        client = genai.Client(api_key=api_key)
        
        prompt = f"""
        You are an expert OpenAPI Repair Agent. 
        
        CONTEXT:
        I have an OpenAPI YAML file that failed validation.
        
        THE LOGS:
        {log_content}
        
        THE YAML FILE:
        {yaml_content}
        
        TASK:
        1. Fix the errors listed in the logs.
        2. PRESERVE strictly all 'x-readme' extensions, 'servers' configurations, and 'info' metadata. Do not remove them.
        3. Return the FULL, VALID YAML file. 
        4. Do not include any markdown formatting, explanations, or chat. JUST THE YAML CODE.
        5. If you must use markdown, wrap the code in ```yaml ... ``` blocks only.
        """
        
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt]
        )
        
        # Extract YAML from Markdown block if present
        cleaned_text = response.text
        match = re.search(r'```yaml\n(.*?)\n```', cleaned_text, re.DOTALL)
        if match:
            return match.group(1)
        
        # Fallback: remove generic code blocks if no language specified
        match_generic = re.search(r'```\n(.*?)\n```', cleaned_text, re.DOTALL)
        if match_generic:
            return match_generic.group(1)

        return cleaned_text
    except Exception as e:
        return None

# --- Git Logic ---

def setup_git_repo(repo_url, repo_dir, git_token, git_username, branch_name, logger):
    logger.info(f"🚀 Starting Git Operation for branch: {branch_name}...")
    
    repo_path = Path(repo_dir)
    repo_url = repo_url.strip().strip('"').strip("'")
    git_username = git_username.strip().strip('"').strip("'")
    git_token = git_token.strip().strip('"').strip("'")

    # Clean URL Logic
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

    git_args = ["-c", "core.askPass=echo"] 

    if not repo_path.exists():
        # --- CLONE NEW REPO ---
        logger.info(f"⬇️ Cloning branch '{branch_name}' from: {masked_repo_url}")
        try:
            # Added --branch and branch_name here
            cmd = ["git"] + git_args + ["clone", "--depth", "1", "--branch", branch_name, auth_repo_url, str(repo_path)]
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
                    st.error(f"Git Clone Failed for branch '{branch_name}'. Check if branch exists.")
                    st.stop()
            logger.info("✅ Repo cloned successfully.")
        except Exception as e:
            logger.error(f"❌ System Error: {e}")
            st.stop()
    else:
        # --- UPDATE EXISTING REPO ---
        logger.info(f"🔄 Switching/Updating to branch '{branch_name}'...")
        try:
            # 1. Update Remote URL (in case tokens changed)
            subprocess.run(["git", "-C", str(repo_path), "remote", "set-url", "origin", auth_repo_url], 
                         check=True, capture_output=True, env=clean_env)
            
            # 2. Fetch the specific branch
            subprocess.run(["git", "-C", str(repo_path), "fetch", "origin", branch_name],
                         check=True, capture_output=True, env=clean_env)

            # 3. Checkout the branch
            subprocess.run(["git", "-C", str(repo_path), "checkout", branch_name],
                         check=True, capture_output=True, env=clean_env)

            # 4. Pull latest changes
            subprocess.run(["git", "-C", str(repo_path), "pull", "origin", branch_name],
                         check=True, capture_output=True, env=clean_env)
            
            logger.info(f"✅ Successfully switched to '{branch_name}'.")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Git Operation failed: {e}")
            logger.warning("⚠️ Attempting to continue with current files (might be outdated)...")

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

def prepare_files(filename, paths, workspace, dependency_list, logger):
    """
    Locates the selected file in either the main or secondary path,
    copies it to workspace, and copies any requested dependency folders.
    """
    source = None
    
    # 1. Search in Main Specs Path
    main_candidate = Path(paths['specs']) / f"{filename}.yaml"
    if main_candidate.exists():
        source = main_candidate
    
    # 2. Search in Secondary Path (if configured)
    elif paths.get('secondary') and (Path(paths['secondary']) / f"{filename}.yaml").exists():
        source = Path(paths['secondary']) / f"{filename}.yaml"

    if not source:
        logger.error(f"❌ Source file '{filename}.yaml' not found.")
        logger.info(f"ℹ️ Searched in: {paths['specs']}")
        if paths.get('secondary'):
            logger.info(f"ℹ️ Searched in: {paths['secondary']}")
        st.stop()

    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    destination = workspace_path / source.name
    shutil.copy(source, destination)
    logger.info(f"📂 Copied main YAML to workspace: {destination.name}")

    # 3. Copy Dependencies (Generic List)
    # Assumes dependencies are located inside the Main Specs folder
    for folder in dependency_list:
        clean_folder = folder.strip()
        if not clean_folder: continue
        
        src_folder = Path(paths['specs']) / clean_folder
        dest_folder = workspace_path / clean_folder
        
        if src_folder.exists():
            if dest_folder.exists():
                shutil.rmtree(dest_folder)
            shutil.copytree(src_folder, dest_folder)
            logger.info(f"📂 Copied dependency folder: {clean_folder}")
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

def process_yaml_content(file_path, version, api_domain, logger):
    """
    Injects ReadMe extensions and updates server URL with user-defined domain.
    """
    logger.info("🛠️ Injecting x-readme extensions and updating server info...")
    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        # Inject x-readme
        if "openapi" in data:
            pos = list(data.keys()).index("openapi")
            items = list(data.items())
            items.insert(pos + 1, ("x-readme", {"explorer-enabled": False}))
            data = dict(items)
        
        data["info"]["version"] = version
        
        # Use user-provided domain or default
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
    if 'repo_url' not in st.session_state: st.session_state.repo_url = ""
    
    # Track files
    if 'last_edited_file' not in st.session_state: st.session_state.last_edited_file = None
    if 'corrected_file' not in st.session_state: st.session_state.corrected_file = None
    
    if 'ai_model' not in st.session_state: st.session_state.ai_model = "gemini-2.5-pro"

    readme_key = st.sidebar.text_input("ReadMe API Key", key="readme_key", type="password", help="Required for Upload or ReadMe Validation")
    
    with st.sidebar.expander("🤖 AI Configuration", expanded=True):
        gemini_key = st.text_input("Gemini API Key", key="gemini_key", type="password", help="Required for AI Analysis")
        ai_model = st.text_input("Model Name", key="ai_model", help="e.g., gemini-2.5-pro")
    
    st.sidebar.subheader("Git Repo Config")
    default_cloud_path = "./cloned_repo"
    repo_path = st.sidebar.text_input("Local Clone Path", value=default_cloud_path)
    
    if st.sidebar.button("🗑️ Reset / Delete Cloned Repo"):
        success, msg = delete_repo(repo_path)
        if success: st.sidebar.success(msg)
        else: st.sidebar.warning(msg)
    
    repo_url = st.sidebar.text_input("Git Repo HTTPS URL", key="repo_url")
    branch_name = st.sidebar.text_input("Branch Name", value="main", help="Enter the specific feature branch name")

    git_user = st.sidebar.text_input("Git Username", key="git_user", type="password", help="GitHub Handle")
    git_token = st.sidebar.text_input("Git Token/PAT", key="git_token", type="password", help="Personal Access Token")

    st.sidebar.button("🔒 Clear Credentials", on_click=clear_credentials)

    # --- INTERNAL PATHS & SETTINGS ---
    st.sidebar.subheader("Internal Paths & Settings")
    spec_rel_path = st.sidebar.text_input("Main Specs Path (relative to repo)", value="specs", help="Folder containing your main OpenAPI files.") 
    secondary_rel_path = st.sidebar.text_input("Secondary Specs Path (Optional)", value="", help="Another folder to scan for YAML files. Leave empty if not needed.")
    dep_input = st.sidebar.text_input("Dependency Folders", value="common", help="Comma-separated list of folders to copy (e.g., 'common, types').")
    dependency_list = [x.strip() for x in dep_input.split(",")]
    api_domain = st.sidebar.text_input("API Base Domain", value="api.example.com", help="Domain to inject into servers.url")

    # Path Setup
    abs_spec_path = Path(repo_path) / spec_rel_path
    paths = {"repo": repo_path, "specs": abs_spec_path}
    if secondary_rel_path:
        paths["secondary"] = Path(repo_path) / secondary_rel_path
    workspace_dir = "./temp_workspace"

    st.title("🚀 OpenAPI Spec Validator")
    
    # --- FILE SELECTION ---
    col1, col2 = st.columns(2)
    with col1:
        files = []
        if abs_spec_path.exists():
            files.extend([f.stem for f in abs_spec_path.glob("*.yaml")])
        if "secondary" in paths and paths["secondary"].exists():
            files.extend([f.stem for f in paths["secondary"].glob("*.yaml")])
        files = sorted(list(set(files)))
        
        if files:
            selected_file = st.selectbox("Select OpenAPI File", files)
        else:
            selected_file = st.text_input("Enter Filename (e.g. 'audit')", "audit")
            if not abs_spec_path.exists():
                st.warning(f"⚠️ Repo not synced yet. Click 'Validate' to clone branch '{branch_name}'.")

    with col2:
        version = st.text_input("API Version", "1.0")

    # --- CHECKBOX UI ---
    st.markdown("### 🚀 Validation Settings")
    c_check1, c_check2, c_check3 = st.columns(3)
    with c_check1: use_swagger = st.checkbox("Swagger CLI", value=True)
    with c_check2: use_redocly = st.checkbox("Redocly CLI", value=True)
    with c_check3: use_readme = st.checkbox("ReadMe CLI", value=False, help="Requires ReadMe API Key")
    
    st.markdown("---")
    
    # --- UPLOAD SOURCE SELECTION ---
    # Determine options based on whether a corrected file exists
    upload_options = ["Original (Edited)"]
    if st.session_state.corrected_file:
        upload_options.append("AI Corrected")
    
    c_select, c_actions = st.columns([1, 2])
    
    with c_select:
        upload_choice = st.radio("File to Upload:", upload_options, horizontal=True)

    with c_actions:
        c_btn1, c_btn2 = st.columns(2)
        btn_validate_selected = c_btn1.button("🔍 Validate Selected", use_container_width=True)
        btn_upload = c_btn2.button(f"🚀 Upload: {upload_choice}", type="primary", use_container_width=True)

    # --- 1. SETUP UI LAYOUT (Must happen BEFORE logic) ---
    st.markdown("### 📜 Execution Logs")
    
    # Container for log text
    log_container = st.empty()
    if st.session_state.logs:
        log_container.code("\n".join(st.session_state.logs), language="text")

    # Columns for Buttons
    col_d1, col_d2, col_d3 = st.columns([1, 1, 3])
    
    with col_d1:
        download_placeholder = st.empty()
        if st.session_state.logs:
            unique_key = f"dl_btn_persist_{len(st.session_state.logs)}"
            download_placeholder.download_button(
                label="📥 Download Logs",
                data="\n".join(st.session_state.logs),
                file_name="openapi_upload.log",
                mime="text/plain",
                key=unique_key
            )

    # --- 2. EXECUTION LOGIC ---
    if btn_validate_selected or btn_upload:
        st.session_state.logs = [] 
        st.session_state.last_edited_file = None
        st.session_state.corrected_file = None # Reset corrected file on new run
        
        logger = logging.getLogger("streamlit_logger")
        logger.setLevel(logging.INFO)
        if logger.handlers: logger.handlers = []
        handler = StreamlitLogHandler(log_container, download_placeholder)
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(handler)

        # Determine Strict Key Req (Only strict if uploading)
        strict_key_req = True if btn_upload else False
        has_key = validate_env(readme_key, required=strict_key_req)
        
        npx_path = get_npx_path()
        if not npx_path:
            logger.error("❌ NodeJS/npx not found.")
            st.stop()

        # GIT CALL
        setup_git_repo(repo_url, repo_path, git_token, git_user, branch_name, logger)

        logger.info("📂 Preparing workspace...")
        final_yaml_path = prepare_files(selected_file, paths, workspace_dir, dependency_list, logger)

        if has_key:
            create_ver = True if btn_upload else False
            check_and_create_version(version, readme_key, "https://dash.readme.com/api/v1", logger, create_if_missing=create_ver)

        # PROCESS & STORE EDITED FILE
        edited_file = process_yaml_content(final_yaml_path, version, api_domain, logger)
        st.session_state.last_edited_file = str(edited_file)

        # --- UPLOAD PATH LOGIC ---
        # If user clicked Upload, we need to decide WHICH file to validate/upload.
        # If they chose "AI Corrected" but it doesn't exist yet (logic edge case), fallback to Edited.
        target_file_to_process = edited_file
        
        if btn_upload and upload_choice == "AI Corrected" and st.session_state.corrected_file:
             logger.info(f"👉 Processing AI Corrected File: {Path(st.session_state.corrected_file).name}")
             target_file_to_process = Path(st.session_state.corrected_file)
        # -------------------------

        # Validation Logic
        if btn_upload:
            do_swagger = True
            do_redocly = False
            do_readme = True
        else:
            do_swagger = use_swagger
            do_redocly = use_redocly
            do_readme = use_readme

        validation_failed = False
        
        # 1. SWAGGER
        if do_swagger:
            logger.info("🔍 Running Swagger CLI...")
            if run_command([npx_path, "--yes", "swagger-cli", "validate", str(target_file_to_process)], logger) != 0: 
                validation_failed = True
        
        # 2. REDOCLY
        if do_redocly:
            logger.info("🔍 Running Redocly CLI...")
            if run_command([npx_path, "--yes", "@redocly/cli@1.25.0", "lint", str(target_file_to_process)], logger) != 0: 
                validation_failed = True
            
        # 3. README
        if do_readme:
            if has_key:
                logger.info("🔍 Running ReadMe CLI (v9)...")
                if run_command([npx_path, "--yes", "rdme@9", "openapi", "validate", str(target_file_to_process)], logger) != 0: 
                    validation_failed = True
            else:
                logger.warning("⚠️ Skipping ReadMe CLI validation.")

        # RESULT
        if validation_failed:
            logger.error("❌ Validation failed.")
            st.error("Validation Failed.")
            if btn_upload: st.error("Aborting upload.")
        else:
            logger.info("✅ Selected validations passed.")
            if btn_upload:
                logger.info("🚀 Uploading to ReadMe...")
                # Read title from the file we are actually uploading
                with open(target_file_to_process, "r") as f:
                    title = yaml.safe_load(f).get("info", {}).get("title", "")
                
                api_id = get_api_id(title, version, readme_key, "https://dash.readme.com/api/v1", logger)
                
                cmd = [npx_path, "--yes", "rdme@9", "openapi", str(target_file_to_process), "--useSpecVersion", "--key", readme_key, "--version", version]
                if api_id: cmd.extend(["--id", api_id])
                
                if run_command(cmd, logger) == 0:
                    logger.info("🎉 Upload Successful!")
                    st.success(f"Done! Uploaded: {upload_choice}")
                else:
                    logger.error("❌ Upload failed.")
            else:
                st.success("Validation Check Complete.")

    # --- 3. POST-EXECUTION UI RENDERING ---
    
    # Column 2: Download Edited (Original)
    with col_d2:
        if 'last_edited_file' in st.session_state and st.session_state.last_edited_file:
            edited_path = Path(st.session_state.last_edited_file)
            if edited_path.exists():
                with open(edited_path, "r") as f:
                    yaml_content = f.read()
                st.download_button(
                    label="📄 Download Edited YAML",
                    data=yaml_content,
                    file_name=edited_path.name,
                    mime="application/x-yaml",
                    key="dl_yaml_btn"
                )

    # Column 3: Clear Logs
    with col_d3:
        if st.session_state.logs:
             st.button("🗑️ Clear Logs", on_click=clear_logs)

    # --- AI ANALYSIS & REPAIR SECTION ---
    if st.session_state.logs and gemini_key:
        st.markdown("### 🤖 AI Assistance")
        
        c_ai1, c_ai2 = st.columns(2)
        
        # 1. Analyze Button
        if c_ai1.button(f"🧐 Analyze Errors"):
            with st.spinner("Analyzing..."):
                log_text = "\n".join(st.session_state.logs)
                analysis = analyze_errors_with_ai(log_text, gemini_key, ai_model)
                if analysis:
                    st.markdown(analysis)

        # 2. Fix Button
        if c_ai2.button(f"✨ Attempt Auto-Fix"):
            if 'last_edited_file' in st.session_state and st.session_state.last_edited_file:
                with st.spinner("Generating fix..."):
                    log_text = "\n".join(st.session_state.logs)
                    fixed_content = apply_ai_fixes(st.session_state.last_edited_file, log_text, gemini_key, ai_model)
                    
                    if fixed_content:
                        # Save the fixed file
                        original_path = Path(st.session_state.last_edited_file)
                        corrected_path = original_path.parent / (original_path.stem.replace("_edited", "") + "_corrected.yaml")
                        
                        with open(corrected_path, "w") as f:
                            f.write(fixed_content)
                        
                        st.session_state.corrected_file = str(corrected_path)
                        st.success("✅ Fix generated! You can now select 'AI Corrected' in the upload options above.")
                        st.rerun() # Rerun to update the Radio Button options immediately
                    else:
                        st.error("Failed to generate a fix.")
    
    # Download Button for Corrected File (If exists)
    if st.session_state.corrected_file:
        corr_path = Path(st.session_state.corrected_file)
        if corr_path.exists():
            with open(corr_path, "r") as f:
                fixed_data = f.read()
            st.download_button(
                label="✨ Download Corrected YAML",
                data=fixed_data,
                file_name=corr_path.name,
                mime="application/x-yaml",
                key="dl_corr_btn"
            )

    elif st.session_state.logs and not gemini_key:
        st.info("💡 Enter a Gemini API Key in the sidebar to unlock AI tools.")

if __name__ == "__main__":
    main()
