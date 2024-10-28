sudo apt update
sudo apt install unzip python3-pip -y
pip3 install kaggle

# Ensure kaggle.json is in the correct location
mkdir -p ~/.config/kaggle
mv /path/to/downloaded/kaggle.json ~/.config/kaggle/kaggle.json #Replace with the actual path where the kaggle.json file was downloaded
chmod 600 ~/.config/kaggle/kaggle.json

# Download competition data
kaggle competitions download -c birdclef-2024