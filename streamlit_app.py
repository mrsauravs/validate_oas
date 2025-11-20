import streamlit as st
import yaml
import subprocess
import shutil
import requests
import os
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# Page Config
st.set_page_config(
    page_title="ReadMe.io OpenAPI Manager",
    page_icon="📘",
    layout="wide"
)

# --- Custom Logging Handler for Streamlit ---
class StreamlitLogHandler(logging.Handler):
    def __init__(self, container):
        super().__init__()
        self.container = container
        self.logs = []

    def emit(self, record):
        msg = self.format(record)
        self.logs.append(msg)
        # Update the container with the latest logs
        self.container.code("\n".join(self.logs), language="text")

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
                log_logger.info(f"[{command_list[0]}] {clean}")
        process.wait()
        return process.returncode
    except Exception as e:
        log_logger.error(f"❌ Command failed: {e}")
        return 1

# --- Main Logic Functions (Refactored for Streamlit) ---

def pull_latest_repo(repo_path, logger):
    logger.info(f"⬇️ Pulling latest changes from: {repo_path}")
    if not os.path.exists(repo_path):
        logger.error(f"❌ Repo path does not exist: {repo_path}")
        st.stop()
    
    try:
        subprocess.run(["git", "-C", str(repo_path), "pull"], check=True, capture_output=True)
        logger.info("✅ Repo updated successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Git pull failed: {e}")
        st.stop()

def prepare_files(filename, paths, workspace, logger):
    """Copies necessary files to a workspace to avoid messing up the main repo."""
    
    # Define source paths based on config
    if filename in ["field", "field_value"]:
        source = Path(paths['logical']) / f"{filename}.yaml"
    else:
        source = Path(paths['specs']) / f"{filename}.yaml"

    if not source.exists():
        logger.error(f"❌ Source file not found: {source}")
        st.stop()

    # Create workspace
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    
    destination = workspace_path / source.name
    shutil.copy(source, destination)
    logger.info(f"📂 Copied main YAML to workspace: {destination.name}")

    # Copy dependencies (common, data_products)
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

def check_and_create_version(version, api_key, base_url, logger):
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

        logger.info(f"⚠️ Version '{version}' not found. Creating it...")
        payload = {"version": version, "is_stable": False, "from": "latest"}
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

        # Insert x-readme extension
        if "openapi" in data:
            pos = list(data.keys()).index("openapi")
            items = list(data.items())
            items.insert(pos + 1, ("x-readme", {"explorer-enabled": False}))
            data = dict(items)
        
        # Update Info
        data["info"]["version"] = version
        
        # Ensure servers block exists
        if "servers" not in data or not data["servers"]:
            data["servers"] = [{"url": "https://alation_domain", "variables": {}}]

        # Update Server Variables safely
        if "variables" not in data["servers"][0]:
            data["servers"][0]["variables"] = {}
            
        data["servers"][0]["variables"]["base-url"] = {"default": "alation_domain"}
        data["servers"][0]["variables"]["protocol"] = {"default": "https"}

        # Save edited file
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
        if response.status_code != 200:
            logger.error("❌ Error retrieving API specs list.")
            return None

        for api in response.json():
            if api["title"] == api_name:
                return api["_id"]
    except Exception:
        pass
    
    logger.warning(f"⚠️ No existing API ID found for title '{api_name}'. A new one will be created.")
    return None

# --- UI Layout ---

def main():
    # 1. Sidebar Configuration
    st.sidebar.title("⚙️ Settings")
    
    # Try to load defaults from .env
    load_dotenv()
    default_key = os.getenv("README_API_KEY", "")
    default_home = str(Path.home() / "Developer" / "alation")
    
    api_key = st.sidebar.text_input("ReadMe API Key", value=default_key, type="password")
    repo_path = st.sidebar.text_input("Alation Repo Path", value=default_home)
    
    # Derived defaults
    default_specs = str(Path(repo_path) / "django" / "static" / "swagger" / "specs")
    default_logical = str(Path(default_specs) / "logical_metadata")
    
    specs_path = st.sidebar.text_input("Swagger Specs Path", value=default_specs)
    logical_path = st.sidebar.text_input("Logical Metadata Path", value=default_logical)
    
    paths = {
        "repo": repo_path,
        "specs": specs_path,
        "logical": logical_path
    }
    
    workspace_dir = st.sidebar.text_input("Workspace Directory", value="./temp_workspace")

    # 2. Main Content
    st.title("🚀 ReadMe.io OAS Uploader")
    st.markdown("Validate and upload OpenAPI specs with full **ReadMe compatibility checks**.")

    col1, col2 = st.columns(2)
    
    with col1:
        # Try to list files for better UX
        try:
            files = [f.stem for f in Path(specs_path).glob("*.yaml")]
            # Add logical files
            files += [f.stem for f in Path(logical_path).glob("*.yaml")]
            files = sorted(list(set(files)))
            selected_file = st.selectbox("Select OpenAPI File", files)
        except:
            selected_file = st.text_input("Enter Filename (no extension)", "audit")

    with col2:
        version = st.text_input("API Version", "1.0")

    # Validation Options
    st.subheader("✅ Validation Settings")
    c1, c2, c3 = st.columns(3)
    run_swagger = c1.checkbox("Swagger CLI (Legacy)", value=False)
    run_redocly = c2.checkbox("Redocly CLI (OAS 3.1)", value=True)
    run_readme = c3.checkbox("ReadMe CLI (Strict)", value=True)

    # Actions
    st.subheader("🚀 Actions")
    use_local = st.checkbox("Use Local Files (Skip Git Pull)", value=False)
    dry_run = st.checkbox("Dry Run (Validate Only)", value=True)
    
    start_btn = st.button("Run Process", type="primary")

    # Output Area
    log_container = st.empty()
    
    if start_btn:
        # Initialize Logger
        logger = logging.getLogger("streamlit_logger")
        logger.setLevel(logging.INFO)
        # Clear old handlers to avoid duplicates
        if logger.handlers:
            logger.handlers = []
        handler = StreamlitLogHandler(log_container)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        validate_env(api_key)
        npx_path = get_npx_path()
        
        if not npx_path:
            logger.error("❌ NodeJS/npx not found. Please install Node.js.")
            st.stop()

        # Step 1: Git Pull
        if not use_local:
            pull_latest_repo(paths['repo'], logger)
        else:
            logger.info("⏭️ Skipping git pull (Local Mode).")

        # Step 2: Prepare Files
        logger.info("📂 Preparing workspace...")
        final_yaml_path = prepare_files(selected_file, paths, workspace_dir, logger)

        # Step 3: Version Check
        check_and_create_version(version, api_key, "https://dash.readme.com/api/v1", logger)

        # Step 4: Edit YAML
        edited_file = process_yaml_content(final_yaml_path, version, logger)

        # Step 5: Validations
        validation_failed = False
        
        if run_swagger:
            logger.info("🔍 Running Swagger CLI...")
            code = run_command([npx_path, "--yes", "swagger-cli", "validate", str(edited_file)], logger)
            if code != 0: validation_failed = True

        if run_redocly:
            logger.info("🔍 Running Redocly CLI...")
            code = run_command([npx_path, "--yes", "@redocly/cli", "lint", str(edited_file)], logger)
            if code != 0: validation_failed = True

        if run_readme:
            logger.info("🔍 Running ReadMe CLI (Compatibility Check)...")
            # "openapi:validate" checks compatibility without uploading
            code = run_command([npx_path, "--yes", "rdme", "openapi:validate", str(edited_file)], logger)
            if code != 0: validation_failed = True

        if validation_failed:
            if dry_run:
                logger.error("❌ Validation failed. Fix errors before uploading.")
                st.error("Validation failed!")
            else:
                logger.error("❌ Validation failed. Aborting upload.")
                st.error("Validation failed! Upload aborted.")
                st.stop()
        else:
            logger.info("✅ All validations passed.")

        # Step 6: Upload
        if dry_run:
            logger.info("🏁 Dry run complete. No files uploaded.")
            st.success("Dry run completed successfully!")
        else:
            if validation_failed:
                st.stop() # Double check
                
            logger.info("🚀 Starting Upload to ReadMe...")
            
            # Get ID
            with open(edited_file, "r") as f:
                data = yaml.safe_load(f)
                api_title = data["info"]["title"]
            
            api_id = get_api_id(api_title, version, api_key, "https://dash.readme.com/api/v1", logger)
            
            # Construct RDME command
            cmd = [
                npx_path, "rdme", "openapi", str(edited_file),
                "--useSpecVersion", 
                "--key", api_key, 
                "--version", version
            ]
            
            if api_id:
                cmd.extend(["--id", api_id])
            
            code = run_command(cmd, logger)
            
            if code == 0:
                logger.info("🎉 Upload Successful!")
                st.success("Upload Complete!")
            else:
                logger.error("❌ Upload failed.")
                st.error("Upload failed.")

if __name__ == "__main__":
    main()
