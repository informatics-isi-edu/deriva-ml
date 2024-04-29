# Instructions of VM setup.

# Clone the Tensorflow Env
```bash
/data/conda-clone-tensorflow.sh
pip install git+https://github.com/informatics-isi-edu/deriva-ml 
```
After this step, you can see "My-Tensorflow" section on Luncher page:
 ![minid](instruction_image/Launcher.png)


# Get GitHub Credential
1. Open Terminal on the VM and activate the env: `conda activate my-tensorflow`
2. put a .git-credentials file in your homedir with: `https://<github-username>:ghp_<token_value>@github.com`
   -  `vim ~/.git-credentials`
   - paste the  `https://<github-username>:ghp_<token_value>@github.com` and change <github-username> to your won 
   GitHub username and <token_value> obtained from: https://github.com/settings/tokens
    - Save file
3. ```bash
   chmod 600 ~/.git-credentials
   git config --global credential.helper store
   ```
   

# Clone Catalog-ml and Catalog-exec repo
1. Create a directory in your homedir for GitHub Repos `mkdir Repos`
2. In the Repo dir, clone the catalog-ml repo which contains Catalog-ML method and ML model module, and catalog-exec repo
    
   Example:

       ```bash
       git clone https://github.com/informatics-isi-edu/eye-ai-ml.git
       git clone https://github.com/informatics-isi-edu/eye-ai-exec.git
       ```
3. Change the notebook and Catalog-ML tools accordingly.
4. Push the changes after test.

# Start a Notebook Workflow
See [ML Workflow Instruction](ml_workflow_instruction.md)