# Remove old venv if exists
Remove-Item -Recurse -Force .\venv

# Create new virtual environment
python -m venv venv

# Activate venv
.\venv\Scripts\Activate.ps1

# Upgrade pip tools
pip install --upgrade pip setuptools wheel

# Install paddlepaddle-gpu with --find-links first
pip install paddlepaddle-gpu==3.1.0 --find-links https://www.paddlepaddle.org.cn/packages/stable/cu126/

# Then install the rest from requirements.txt excluding paddlepaddle-gpu
pip install -r requirements.txt
