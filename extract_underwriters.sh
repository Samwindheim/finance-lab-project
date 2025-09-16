
#!/bin/bash

# This script is used to extract the underwriters and guarantors from a PDF file.
# It uses the main.py script to extract the data and the models.py script to validate the data.
# Uses the .venv virtual environment to run the Python script.
# Uses the src/main.py script to extract the data and the src/models.py script to validate the data.
# Uses the .venv virtual environment to run the Python script.

# Check if a PDF file path is provided
if [ -z "$1" ]; then
  echo "Usage: $0 path/to/document.pdf"
  exit 1
fi

# Activate the virtual environment and run the main Python script
source .venv/bin/activate
python3 -m src.main "$@"
