# インストール前にアップグレード
pip install --upgrade pip setuptools wheel
# PyTorchのインストール（CUDA対応）
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
pip install spacy
python -m spacy download en_core_web_lg