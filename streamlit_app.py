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
import platform
from pathlib import Path
from dotenv import load_dotenv

# Page Config
st.set_page_config(
    page_title="ReadMe.io OpenAPI Manager v2.12",
    page_icon="📘",
    layout="wide"
)

# --- Custom Logging Handler for Streamlit ---
class StreamlitLogHandler(logging.Handler):
    def __init__(self, container, download_placeholder=None):
        super().__init__()
        self.container = container
        self.download_placeholder = download_placeholder
        self.logs = []

    def emit(self, record):
        msg = self.format(record)
        self.logs.append(msg)
        full_log = "\n".join(self.logs)
        
        # Update the code block
        self.container.code(full_log, language="text")
        
        # Update the download button in real-time
        if self.download_placeholder:
            self.download_placeholder.download_button(
                label="📥 Download Log File",
                data=full_log,
                file_name="openapi_upload.log",
                mime="text/plain",
                key="log_download_btn"
            )

# --- Helper Functions ---

def get_npx_path():
    return shutil.which("npx")

def validate_env(api_key):
    if not api_key:
        st.error("❌ README_API_KEY is missing. Please enter it in the sidebar.")
        st.stop()
    return True

def run_command(command_list, log_logger):
    """Runs a subprocess command and logs output real-time."""
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

# --- Git Logic (v2.12 - Privacy Update) ---

def setup_git_repo(repo_url, repo_dir, git_token, git_username, logger):
    """Clones/Pulls repo with specific handling for SAML SSO errors."""
    
    logger.info("🚀 Starting Git Operation...")
    
    repo_path = Path(repo_dir)
    repo_url = repo_url.strip().strip('"').strip("'")
    git_username = git_username.strip().strip('"').strip("'")
    git_token = git_token.strip().strip('"').strip("'")

    # 1. Fix double https://
    if repo_url.count("https://") > 1:
        match = re.search(r"(https://github\.com/.*)$", repo_url)
        if match:
            repo_url = match.group(1)

    # 2. Construct Authenticated URL
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
        
        # Masked URL for logging (Hides both User and Token for privacy)
        masked_netloc = f"****:***@{clean_netloc}"
        masked_repo_url = urllib.parse.urlunparse((
            parsed.scheme, masked_netloc, parsed.path, parsed.params, parsed.query, parsed.fragment
        ))
        
    except Exception as e:
        logger.error(f"❌ URL Construction Failed: {e}")
        st.stop()

    # 3. Env Setup
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
                # --- SSO ERROR DETECTION ---
                sso_match = re.search(r"(https://github\.com/orgs/[^/]+/sso\?authorization_request=[^\s]+)", result.stderr)
                
                if sso_match:
                    sso_url = sso_match.group(1)
                    logger.error("❌ SSO AUTHORIZATION REQUIRED")
                    st.error("🚨 Organization requires SAML SSO Authorization.")
                    st.markdown(f"👉 **[Click here to Authorize your Token]({sso_url})**")
                    st.caption("After authorizing, click 'Start Process' again.")
                    st.stop()
                
                elif "403" in result.stderr:
                    st.error("🚨 Authentication Failed (403).")
                    st.info("Ensure your Token has 'repo' scope and SSO is configured.")
                    st.stop()
                
                else:
                    # Filter sensitive data from error logs before showing
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
    """Deletes the repo directory to allow a fresh clone."""
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
    """Copies necessary files to a workspace."""
    
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

def check_and_create_version(version, api_key, base_url, logger, dry_run=False):
    headers = {
        "Authorization": f"Basic {api_key}",
        "Accept": "application/json"
    }
    
    logger.info(f"🔎 Checking version '{version}' on ReadMe...")
    try:
        response = requests.get(f"{base_url}/version", headers=headers)
        if response.status_code != 200:
            logger.error(f"❌ Failed to fetch versions: {response.text}")
            st.stop()
        
        versions = response.json()
        if any(v["version"] == version for v in versions):
            logger.info(f"✅ Version '{version}' exists.")
            return

        # Version does not exist
        if dry_run:
            logger.warning(f"⚠️ Version '{version}' not found. Skipping creation (Dry Run).")
            logger.info(f"ℹ️ The validation will proceed assuming '{version}' will be created later.")
            return

        # If NOT dry run, try to create it
        logger.info(f"⚠️ Version '{version}' not found. Creating it...")
        
        # FIND BEST FORK TARGET (Don't assume 'latest' exists)
        if versions and len(versions) > 0:
            fork_target = versions[0]['version'] # Use the most recent one
        else:
            fork_target = "latest" # Fallback
            
        logger.info(f"ℹ️ Forking new version from: {fork_target}")

        payload = {"version": version, "is_stable": False, "from": fork_target}
        create_response = requests.post(f"{base_url}/version", headers=headers, json=payload)
        
        if create_response.status_code == 201:
            logger.info(f"✅ Version '{version}' created successfully.")
        else:
            logger.error(f"❌ Failed to create version: {create_response.text}")
            st.stop()
            
    except Exception as e:
        logger.error(f"❌ Network error checking version: {e}")
        st.stop()

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
    headers = {
        "Authorization": f"Basic {api_key}",
        "Accept": "application/json",
        "x-readme-version": version
    }
    
    try:
        response = requests.get(f"{base_url}/api-specification", headers=headers, params={"perPage": 100})
        if response.status_code == 200:
            for api in response.json():
                if api["title"] == api_name:
                    return api["_id"]
    except Exception:
        pass
    return None

# --- UI Layout ---

def main():
    st.sidebar.title("⚙️ Configuration")
    
    # Load secrets
    load_dotenv()
    secrets = st.secrets if os.path.exists(".streamlit/secrets.toml") else {}
    
    # 1. ReadMe Credentials
    readme_key = st.sidebar.text_input("ReadMe API Key", value=secrets.get("README_API_KEY", os.getenv("README_API_KEY", "")), type="password")
    
    # 2. Git Configuration
    st.sidebar.subheader("Git Repo Config")
    
    default_local_path = str(Path.home() / "Developer" / "alation")
    default_cloud_path = "./cloned_repo"
    
    is_cloud = not Path(default_local_path).exists()
    repo_dir_default = default_cloud_path if is_cloud else default_local_path

    repo_url = st.sidebar.text_input("Git Repo URL", value="https://github.com/alation/alation.git")
    repo_path = st.sidebar.text_input("Local Clone Path", value=repo_dir_default)
    
    # Add Reset Button
    if st.sidebar.button("🗑️ Reset / Delete Cloned Repo"):
        success, msg = delete_repo(repo_path)
        if success:
            st.sidebar.success(msg)
        else:
            st.sidebar.warning(msg)
    
    git_user = st.sidebar.text_input("Git Username", value=secrets.get("GIT_USERNAME", ""))
    st.sidebar.caption("GitHub Handle (e.g., user-name-company)") # <--- UPDATED
    git_token = st.sidebar.text_input("Git Token/PAT", value=secrets.get("GIT_TOKEN", ""), type="password")

    # 3. Path Mapping
    st.sidebar.subheader("Internal Paths")
    spec_rel_path = st.sidebar.text_input("Specs Path (relative to repo)", value="django/static/swagger/specs")
    
    abs_spec_path = Path(repo_path) / spec_rel_path
    abs_logical_path = abs_spec_path / "logical_metadata"
    
    paths = {
        "repo": repo_path,
        "specs": abs_spec_path,
        "logical": abs_logical_path
    }
    
    workspace_dir = "./temp_workspace"

    # Main Content
    st.title("🚀 ReadMe.io Manager v2.12")
    st.markdown("Logic v2.12: UI Cleaned & Logs Masked")
    
    if is_cloud:
        st.info("☁️ Detected Cloud Environment.")
    else:
        st.success(f"💻 Detected Local Environment.")

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

    st.subheader("✅ Validation Settings")
    c1, c2, c3 = st.columns(3)
    run_swagger = c1.checkbox("Swagger CLI", value=False)
    run_redocly = c2.checkbox("Redocly CLI", value=True)
    run_readme = c3.checkbox("ReadMe CLI", value=True)

    st.subheader("🚀 Run")
    dry_run = st.checkbox("Dry Run (Validate Only)", value=True)
    start_btn = st.button("Start Process", type="primary")

    # Placeholders for logs and download button
    log_container = st.empty()
    download_placeholder = st.empty()
    
    if start_btn:
        # Init Logger with download placeholder
        logger = logging.getLogger("streamlit_logger")
        logger.setLevel(logging.INFO)
        if logger.handlers: logger.handlers = []
        
        # Pass the download_placeholder to the handler
        handler = StreamlitLogHandler(log_container, download_placeholder)
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(handler)

        validate_env(readme_key)
        npx_path = get_npx_path()
        if not npx_path:
            logger.error("❌ NodeJS/npx not found.")
            st.stop()

        # Step 1: Git Clone/Pull
        setup_git_repo(repo_url, repo_path, git_token, git_user, logger)

        # Step 2: Prepare Files
        logger.info("📂 Preparing workspace...")
        final_yaml_path = prepare_files(selected_file, paths, workspace_dir, logger)

        # Step 3: Version Check
        check_and_create_version(version, readme_key, "https://dash.readme.com/api/v1", logger, dry_run=dry_run)

        # Step 4: Edit YAML
        edited_file = process_yaml_content(final_yaml_path, version, logger)

        # Step 5: Validations
        validation_failed = False
        
        # NOTE: We pin versions to match Streamlit Cloud's Node v18 environment
        if run_swagger:
            logger.info("🔍 Running Swagger CLI...")
            if run_command([npx_path, "--yes", "swagger-cli", "validate", str(edited_file)], logger) != 0: validation_failed = True
        
        if run_redocly:
            logger.info("🔍 Running Redocly CLI (Pinned v1.25.0)...")
            if run_command([npx_path, "--yes", "@redocly/cli@1.25.0", "lint", str(edited_file)], logger) != 0: validation_failed = True
            
        if run_readme:
            logger.info("🔍 Running ReadMe CLI (Pinned v9.3.2)...")
            if run_command([npx_path, "--yes", "rdme@9.3.2", "openapi:validate", str(edited_file)], logger) != 0: validation_failed = True

        if validation_failed:
            logger.error("❌ Validation failed. Aborting upload.")
            st.error("Validation failed!")
            st.stop()
        else:
            logger.info("✅ All validations passed.")

        # Step 6: Upload
        if dry_run:
            logger.info("🏁 Dry run complete.")
            st.success("Dry run completed!")
        else:
            logger.info("🚀 Uploading to ReadMe...")
            
            with open(edited_file, "r") as f:
                title = yaml.safe_load(f).get("info", {}).get("title", "")
                
            api_id = get_api_id(title, version, readme_key, "https://dash.readme.com/api/v1", logger)
            
            # Use pinned version for upload as well
            cmd = [npx_path, "--yes", "rdme@9.3.2", "openapi", str(edited_file), "--useSpecVersion", "--key", readme_key, "--version", version]
            if api_id: cmd.extend(["--id", api_id])
            
            if run_command(cmd, logger) == 0:
                logger.info("🎉 Upload Successful!")
                st.success("Done!")
            else:
                logger.error("❌ Upload failed.")

if __name__ == "__main__":
    main()
