#!/bin/bash

# A script to download and extract specific zip files.
# It will stop immediately if any command fails.
set -e

# --- Configuration ---
# An array containing the base names of the files to process.
# This makes it easy to add more files in the future.
# The script will derive the zip filename and the destination directory from these.
FILES_TO_PROCESS=("1B_ref" "8B_ref")

# The base URL where the files are located.
BASE_URL="https://dataflowsim.sunshein.net/static/data"

# The parent directory for the extracted data.
DESTINATION_PARENT="."

# --- Main Logic ---
echo "Starting data download and extraction process..."
echo

# Loop through each item in our array
for file_base in "${FILES_TO_PROCESS[@]}"; do
    
    ZIP_FILE="${file_base}.zip"
    FULL_URL="${BASE_URL}/${ZIP_FILE}"
    
    # Create the destination directory name by removing "_ref" from the file base.
    # e.g., "1B_ref" becomes "1B"
    FULL_DEST_PATH="${DESTINATION_PARENT}"

    echo "--- Processing ${ZIP_FILE} ---"

    # 1. Create the destination directory.
    # The -p flag creates parent directories as needed and doesn't fail if it already exists.
    echo "Creating directory: ${FULL_DEST_PATH}"
    mkdir -p "${FULL_DEST_PATH}"

    # 2. Download the zip file.
    # -L: Follow redirects if any.
    # -o: Specify the output filename.
    echo "Downloading from: ${FULL_URL}"
    curl -L -o "${ZIP_FILE}" "${FULL_URL}"

    # 3. Extract the contents to the specific directory.
    # -o: Overwrite existing files without prompting.
    # -d: Specify the destination directory for extraction.
    echo "Extracting ${ZIP_FILE} to ${FULL_DEST_PATH}"
    unzip -o "${ZIP_FILE}" -d "${FULL_DEST_PATH}"

    # 4. Delete the leftover zip file.
    echo "Deleting archive: ${ZIP_FILE}"
    rm "${ZIP_FILE}"

    echo "--- Successfully processed ${ZIP_FILE} ---"
    echo # Add a blank line for readability

done

echo "All tasks completed successfully."
