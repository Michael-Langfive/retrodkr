# 🐋 RetroDKR: Enhancing Retrosynthesis Prediction with Dual Knowledge Retrieval

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---



The directory contains source code of the article "RetroDKR: Enhancing Retrosynthesis Prediction with Dual Knowledge Retrieval".
 

Most existing single-step retrosynthesis approaches struggle to effectively incorporate diverse synthetic knowledge sources, such as reaction templates, molecular fragments, and database precedents, into a unified predictive framework. To address this challenge, we propose RetroDKR, a retrieval-augmented retrosynthesis model that unifies rule-based fragment extraction and large-scale database retrieval within a single framework, effectively incorporating multi-source domain knowledge. 

<div align=center>
<img src=figures/framework.png width="550"/>
</div>

## 🛠️ Setup

- Create the environment:

```
conda create -n retrodkr python=3.8.10
pip install -r requirements.txt
```

- Install pytorch following the command:
```
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
```


- Install OpenNMT for R-SMILES Backbone:
```
cd OpenNMT-py
pip install -e .
```

- Install Fairseq for EditRetro Backbone:
```
cd fairseq
pip install --editable ./
```

&nbsp;&nbsp;&nbsp; [!IMPORTANT] Installation Remarks:
1. Set export CUDA_HOME=/usr/local/cuda in .bashrc;
2. Please verify the versions of CUDA (11.6.0) and gcc (9.4.0);
3. To ensure a successful installation of fairseq, please make sure to install Ninja first.
```
sudo apt install re2c
sudo apt-get install ninja-build
```
4. Verification: If installed successfully, a file named libnat_cuda.cpython-38-x86_64-linux-gnu.so should be generated in the fairseq directory.



## ⚙️ Data preprocessing
 The original datasets used in this paper are from:

   USPTO-50K: https://github.com/Hanjun-Dai/GLN 

   USPTO-MIT: https://github.com/wengong-jin/nips17-rexgen/blob/master/USPTO/data.zip

   USPTO-FULL: https://github.com/Hanjun-Dai/GLN


Download **raw** datasets and put them in the _RetroDKR/datasets/XXX(e.g., USPTO_50K)/raw_ folder, and then run the command to get the preprocessed datasets which will be stored in _RetroDKR/datasets/XXX/aug_:

  ```shell
    cd preprocessing
    # 1. Basic Processing
    python process_datasets.py
    python build_fp_index.py
    # 2. R-SMILES Backbone Processing
    python generate_data.py -dataset USPTO_50K -augmentation 20 -method morgan -use_ref 
    python generate_data.py -dataset USPTO_FULL -augmentation 5 -method morgan -use_ref
    # 3. EditRetro Backbone Processing
    python preprocess_data.py -dataset USPTO_50K -augmentation 20 -spe -method morgan -retriever
    sh binarize.sh ../datasets/USPTO_50K/aug20 dict.txt
  ```




## 🚀 Train

### R-SMILES Backbone

Run the prepared shell command to start training according to your task.
```shell
onmt_train -config train-from-scratch/aug20/morgan_1.0_0.5/train_config.yml
onmt_train -config train-from-scratch/aug5/morgan_1.0_0.5/full_train_config.yml
```


### EditRetro Backbone

1. Download the pre-trained EditRetro checkpoint with 1000K updates from: https://drive.google.com/file/d/12kLcr7R0oBcsgqOSvgZAQk9a6EEkyW8S/view?usp=drive_link.
2. Run the prepared shell command.

```shell
sh ./scripts/1_finetune_50k.sh  
```

## 🧪 Inference


### R-SMILES Backbone
Run the following commands to generate and score the predictions on the test set:

* Step 1: Average and translate the last 5 checkpoints:
```shell
# USPTO_50K
onmt_average_models -output  ./exp/USPTO_50K/aug20/morgan_1.0_0.5/average_model_56-60.pt \
    -m  exp/USPTO_50K/aug20/morgan_1.0_0.5/model.product-reactants_step_560000.pt \
        exp/USPTO_50K/aug20/morgan_1.0_0.5/model.product-reactants_step_570000.pt \
        exp/USPTO_50K/aug20/morgan_1.0_0.5/model.product-reactants_step_580000.pt \
        exp/USPTO_50K/aug20/morgan_1.0_0.5/model.product-reactants_step_590000.pt \
        exp/USPTO_50K/aug20/morgan_1.0_0.5/model.product-reactants_step_600000.pt
onmt_translate -config train-from-scratch/aug20/morgan_1.0_0.5/translate.yml
```

```shell
# USPTO_FULL
onmt_average_models -output  ./exp/USPTO_FULL/aug5/morgan_1.0_0.5/average_model_156-160.pt \
    -m  exp/USPTO_FULL/aug5/morgan_1.0_0.5/model.product-reactants_step_1560000.pt \
        exp/USPTO_FULL/aug5/morgan_1.0_0.5/model.product-reactants_step_1570000.pt \
        exp/USPTO_FULL/aug5/morgan_1.0_0.5/model.product-reactants_step_1580000.pt \
        exp/USPTO_FULL/aug5/morgan_1.0_0.5/model.product-reactants_step_1590000.pt \
        exp/USPTO_FULL/aug5/morgan_1.0_0.5/model.product-reactants_step_1600000.pt
onmt_translate -config train-from-scratch/aug5/morgan_1.0_0.5/full_translate.yml
```
* Step 2: Score the predictions.
```shell
# USPTO_50K
python score.py \
    -augmentation 20 \
    -targets ./datasets/USPTO_50K/aug20/morgan/test/tgt-test.txt \
    -predictions ./exp/USPTO_50K/aug20/morgan_1.0_0.5/average_model_56-60-results.txt \
    -save_file ./results/USPTO_50K/aug20/final_results/final_results_morgan_1.0_0.5.txt \
    -metrics_file ./results/USPTO_50K/aug20/metrics/metrics_summary_morgan_1.0_0.5.csv
```

```shell
# USPTO_FULL
python score.py \
    -augmentation 5 \
    -targets ./datasets/USPTO_FULL/aug5/morgan/test/tgt-test.txt \
    -predictions ./exp/USPTO_FULL/aug5/morgan_1.0_0.5/average_model_156-160-results.txt \
    -save_file ./results/USPTO_FULL/aug5/final_results/final_results_morgan_1.0_0.5.txt \
    -metrics_file ./results/USPTO_FULL/aug5/metrics/metrics_summary_morgan_1.0_0.5.csv
```

### EditRetro Backbone
To generate and score the predictions on the test set with binarized data:
```shell
sh ./scripts/2_generate_50k.sh
```

## 📚 Reference
Our code is based on facebook fairseq-0.9.0 version modified from https://github.com/weijia-xu/fairseq-editor and https://github.com/nedashokraneh/fairseq-editor.

OpenNMT-py: https://github.com/OpenNMT/OpenNMT-py