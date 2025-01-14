apt-get update
apt-get install -y vim

git config --global user.email "quocdat.le.insacvl@gmail.com"
git config --global user.name "LE Quoc Dat"
git config --global credential.helper store

apt-get install git-lfs
git lfs install


# git clone https://quocdat-le-insacvl:<token>@github.com/quocdat-le-insacvl/fast-apply-model.git


git clone https://github.com/microsoft/OmniParser

cd OmniParser



mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all


conda create -n "omni" python==3.12 -y
conda activate omni
pip install -r requirements.txt

pip install huggingface_hub 
# wandb

export HUGGINGFACE_TOKEN=...
export GITHUB_TOKEN=...
export HF_HOME="/workspace/.cache/huggingface"

huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential

git clone https://github.com/kortix-ai/fast-apply
git clone https://github.com/kortix-ai/mirko

pip install unsloth
pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"