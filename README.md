# **ReadMe.io OpenAPI Spec Validator 🚀**

This is a **Streamlit** application designed to automate the validation and uploading of OpenAPI Specifications (OAS) to [ReadMe.io](https://readme.io).

It solves common compatibility issues between standard OAS files and ReadMe's specific requirements by running validation checks using **Swagger CLI**, **Redocly CLI**, and **ReadMe's official CLI (rdme)** before uploading.

## **✨ Features**

* **Multi-Validator Support:** Runs Swagger, Redocly, and ReadMe validation in one go.  
* **Git Integration:** Clones your private repository directly within the app (supports PAT & SSO).  
* **Real-time Logging:** See validation errors and process logs instantly in the UI.  
* **Automatic Fixes:** Automatically injects x-readme extensions and updates server URLs before uploading.  
* **Secure:** Handles API keys and Git tokens via Streamlit Secrets or Environment Variables.  
* **Downloadable Logs:** Download full execution logs for debugging.

## **🛠️ Prerequisites**

To run this app locally or deploy it, you need:

1. **Python 3.8+**  
2. **Node.js & npm** (Required to run npx, rdme, and redocly)  
3. **Git**

## **🚀 Quick Start (Local)**

1. **Clone this repository:**  
   git clone \<this-repo-url\>  
   cd \<repo-folder\>

2. **Install Python Dependencies:**  
   pip install \-r requirements.txt

3. Set up Secrets (Optional but Recommended):  
   Create a file named .streamlit/secrets.toml:  
   README\_API\_KEY \= "your-readme-api-key"  
   GIT\_USERNAME \= "your-github-handle"  
   GIT\_TOKEN \= "your-github-pat"

4. **Run the App:**  
   streamlit run openapi\_manager.py

## **☁️ Deployment (Streamlit Cloud)**

1. Push this code to a GitHub repository.  
2. Connect your repo to [Streamlit Cloud](https://share.streamlit.io/).  
3. In the App Settings on Streamlit Cloud, go to **Secrets** and add your credentials:  
   README\_API\_KEY \= "rdme\_..."  
   GIT\_USERNAME \= "your-github-handle"  
   GIT\_TOKEN \= "ghp\_..."

4. **Important:** Ensure your packages.txt file includes nodejs, npm, and git so the cloud environment installs them.

## **📖 How to Use**

### **1\. Configuration (Sidebar)**

* **ReadMe API Key:** Your project key from ReadMe.io.  
* **Git Repo URL:** The HTTPS URL of the repository containing your OpenAPI specs (e.g., https://github.com/root-folder/git-folder-name.git).  
* **Git Username/Token:** Your GitHub credentials.  
  * *Note:* If your organization uses **SSO**, you must authorize your PAT for that organization.  
* **Internal Paths:** The relative path inside your repo where .yaml files are located.

### **2\. Main Dashboard**

1. **Select OpenAPI File:** The app lists all .yaml files found in your repo.  
2. **API Version:** Enter the version string (e.g., 1.0, 2024.3).  
3. **Validation Settings:** Choose which validators to run.  
   * **Swagger CLI:** Legacy OAS 3.0 check.  
   * **Redocly CLI:** Modern OAS 3.1 linting.  
   * **ReadMe CLI:** Checks specifically for ReadMe platform compatibility.  
4. **Dry Run:** **Keep this checked** to test your file without uploading.  
5. **Start Process:** Click to begin.

### **3\. Troubleshooting**

* **SSO Error (403):** If you see a "SAML SSO" error in the logs, click the authorization link provided in the error message to authorize your token.  
* **Validation Failed:** Read the logs. If Redocly or ReadMe reports errors (like trailing slashes or missing $ref), you must fix them **in your source YAML file** in your repository. The script will not fix content errors for you.

## **📂 File Structure**

* openapi\_manager.py: The main Streamlit application.  
* requirements.txt: Python dependencies.  
* packages.txt: System dependencies for Streamlit Cloud (Node.js).

## **⚠️ Known Limitations**

* **Trailing Slashes:** ReadMe strictly rejects paths ending in /. Ensure your YAML paths are clean (e.g., /users not /users/).  
* **Node Version:** The app currently pins rdme and redocly to versions compatible with Node.js v18 to ensure stability on Streamlit Cloud.
