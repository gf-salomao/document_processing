#!/bin/bash

# Download real-world documents collection zip to Downloads
echo "Checking for existing document collection zip..."
if [ -f ~/Downloads/real-world-documents-collections.zip ]; then
  echo "Document collection zip already exists, skipping download."
else
  echo "Downloading document collection..."
  curl -L -o ~/Downloads/real-world-documents-collections.zip \
    https://www.kaggle.com/api/v1/datasets/download/shaz13/real-world-documents-collections
fi

# Create data/raw folder if it doesn't exist
mkdir -p data/raw

# Extract the downloaded zip into data/raw
echo "Extracting documents to data/raw..."
unzip -o ~/Downloads/real-world-documents-collections.zip -d data/raw

# Download LLaMA 2 Chat model GGUF file from TheBloke to models folder
echo "Checking for existing LLaMA 2 Chat model..."
mkdir -p models
if [ -f models/llama-2-7b-chat.Q4_K_M.gguf ]; then
  echo "Model file already exists, skipping download."
else
  echo "Downloading LLaMA 2 Chat model..."
  curl -L -o models/llama-2-7b-chat.Q4_K_M.gguf \
    https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
fi

echo "All done!"