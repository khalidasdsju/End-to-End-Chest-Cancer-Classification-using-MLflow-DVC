# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# For Apple Silicon (M1/M2) Macs
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
# Install Python 3.10
brew install python@3.10
# Create virtual environment
python3.10 -m venv chest

# Activate the environment
source chest/bin/activate

# Verify Python version
python --version