
#!/bin/bash

# Check if a PDF file path is provided
if [ -z "$1" ]; then
  echo "Usage: $0 path/to/document.pdf"
  exit 1
fi

# Activate the virtual environment and run the main Python script
source .venv/bin/activate
python3 main.py "$@"
