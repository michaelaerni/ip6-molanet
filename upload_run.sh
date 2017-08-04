#!/bin/bash

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
    echo "Usage: upload_run.sh source-directory url [target-suffix] [key-file]"
    echo "source-directory: Run directory to upload"
    echo "url: URL (with username prefix) of the server"
    echo "target-suffix: Suffix appended to the source directory name"
    echo "key-file: SSH key file to use for authentication"
    exit 1
fi

SOURCE_DIRECTORY=$1
URL=$2
TARGET_SUFFIX=${3:-""}
KEY_FILE=$4

TARGET_DIRECTORY=$(basename "${SOURCE_DIRECTORY}")
if [ "$TARGET_SUFFIX" != "" ]; then
    TARGET_DIRECTORY+="_${TARGET_SUFFIX}"
fi

TARGET_PATH="/archive/${TARGET_DIRECTORY}"

echo "Source directory: ${SOURCE_DIRECTORY}/*"
echo "Target directory: ${TARGET_PATH}"

CMD="sftp"
if [ $KEY_FILE ]; then
    CMD+=" -i $KEY_FILE"
fi

CMD+=" -oBatchMode=no -b - ${URL}"

echo "Starting upload..."
$CMD << !
    mkdir "${TARGET_PATH}"
    put -r ${SOURCE_DIRECTORY%%/}/* "${TARGET_PATH}"
    bye
!
echo "Upload done"
