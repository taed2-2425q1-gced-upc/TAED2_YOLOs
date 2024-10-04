#!/bin/bash

# This script is used to automate the data uploading phase to DVC

# Get as optional parameter the commit message - Set also default value
commit_message = "${1:-Track data with DVC}"

# Step 1: Add data to local DVC repository
dvc add data/

# Step 2: Add .dvc files to Git
git add data.dvc

# Step 3: Commit changes to local Git repository
git commit -m "$commit_message"

# Step 4: Upload changes to remote Git repository
git push

# Step 5: Upload changes to remote DVC repository
dvc push

# Report end of process
echo "Data correctly uploaded to DVC"
