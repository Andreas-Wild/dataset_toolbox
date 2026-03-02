#!/bin/bash

echo "Creating directories..."
mkdir -p data files sam2

cd sam2
git clone https://github.com/HumanSignal/label-studio-ml-backend.git
cd ..

echo "Creating .env file..."
cat > .env <<EOF
# Label Studio Legacy API Key
# Obtain this from Label Studio UI after initial setup (see README.md Step 4)
LS_API_KEY=<enter your token here>
EOF

echo "Setting permissions for data and files directories..."
sudo chown -R 1001:1001 data files

echo "Setup complete."