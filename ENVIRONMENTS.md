## General Setup

Please setup your environment as:

```bash
conda create --name cradle-dev python=3.10
conda activate cradle-dev
pip3 install -r requirements.txt
```

### To Install the OCR Tools
```
1. Option 1
# Download best-matching version of specific model for your spaCy installation
python -m spacy download en_core_web_lg

or

# pip install .tar.gz archive or .whl from path or URL
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1.tar.gz

2. Option 2
# Copy this url https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1.tar.gz
# Paste it in the browser and download the file to res/spacy/data
cd res/spacy/data
pip install en_core_web_lg-3.7.1.tar.gz
```

### Other Dependencies

Keep the Cradle requirements.txt file updated in your branch, but only add dependencies that are really required by the system.

runner.py is the entry point to run an agent. Currently not working code, just an introductory sample.

## Environment-specific Setup Instructions

To setup each environment correctly, please check out their specific pages for details.

[Software Applications](docs/envs/software.md)
[Digital Games](docs/envs/games.md)
