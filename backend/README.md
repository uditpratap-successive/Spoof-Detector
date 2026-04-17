# Spoof-Detector

FOR VIRTUAL ENV -

step 1 - python3.11 -m venv venv
step 2- source venv/bin/activate


to run project  - uvicorn app:app --host 0.0.0.0 --port 8000 OR python -m uvicorn app:app --host 0.0.0.0 --port 8000
  i


Command to install dependencies: # Make sure you have pip updated
pip install --upgrade pip

# Install PyTorch (choose the appropriate command for your system)
# For CPU only:
pip install torch torchvision torchaudio

# Then install other dependencies from requirements.txt (if exists)
pip install -r requirements.txt