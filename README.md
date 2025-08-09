# 詳細な説明 (日本語, 内部向け)
水野班での深層学習などの開発向けのテンプレートリポジトリ.  

## 最初にやること

### Gitフックによるファイルサイズ制限
- DVCでcheckpoints, outputs, dataなどが上がる仕組みになっている.  
- この制限を掛けるための設定が初回に必須.  
- 最大値は.envにて設定可能(デフォルト上限1GB).  

### DVCの初期設定
- 「githubに重いファイル上げられない」問題を解決するのがDVC.  
- DVCの初期設定が必要.  
- 詳細は [DVCガイド(docs/dvc_guide.md)](docs/dvc_guide.md)を参照.  

## 構成
------------  
    # 全体概要
    template-nn/
    ├── README.md
    ├── LICENSE
    ├── .env                          # 設定管理用
    ├── .gitignore                    
    ├── config/
    │   └── default.yml
    ├── requirements.txt
    ├── Makefile                      # タスク管理・自動化
    ├── notebooks/
    │   ├── exploratory/
    │   └── modeling/
    ├── data/                         # データ格納 (DVC推奨)
    ├── outputs/                      # 出力結果 (DVC推奨)
    ├── models/
    │   └── checkpoints/              # モデル (DVC推奨)
    ├── scripts/
    │   ├── pre-commit-check.sh       # ファイルサイズ制限フック
    │   └── setup-hooks.sh            # フック設定用
    ├── src/*
    ├── tests/                        # 適宜srcに併せて変更
    │   ├── test_data.py
    │   ├── test_model.py
    │   └── test_utils.py
    ├── .github/workflows/ci.yml      # CI環境（GitHub Actions）
    └── docs/
        └── dvc_guide.md              # DVCの利用ガイド

------------


------------  
    # src詳細
    src/
    ├── data/
    │ ├── dataset.py # DatasetとDataLoaderの組み立て
    │ └── preprocessing.py # 前処理(クリーニング・正規化・特徴生成)
    │
    ├── models.py # モデル定義
    │
    ├── trainer.py # 学習ロジックの中心 (fit/evaluateの公開API)
    │
    ├── utils/
    │ ├── general.py # シード固定・時間計測などの純汎用ユーティリティ
    │ ├── checkpoint.py # モデル/履歴の保存・読み込み(Checkpoint操作)
    │ ├── plot.py # 学習曲線などのプロット
    │ └── callbacks.py # 学習callback (EarlyStopping・DefaultLoggerなど)
    │
    ├── train.py # 学習のentry (薄いCLI)
    └── evaluate.py # 評価のentry (薄いCLI)

------------



# ＝＝＝　以上は公開時には削除する　＝＝＝

# REPOSITORY NAME

A flexible and practical template repository for developing deep learning projects.


## 📝 Note
This repository is under construction and will be officially released by [Mizuno group](https://github.com/mizuno-group).  
Please contact tadahaya[at]gmail.com before publishing your paper using the contents of this repository.  


## 🚀 Overview

This repository provides a structured and reusable environment tailored specifically for deep learning research and development. It streamlines common tasks such as training, evaluation, and version management of data and models.

**Main features include**:
- Unified and easy configuration management (`.env`, YAML configs).
- Simplified task automation via `Makefile`.
- Version control of large datasets and models using [DVC](https://dvc.org/).
- File size restriction and commit safety via Git hooks.

---

## 📦 Requirements and Installation

**Python:** 3.10+

**Main packages**:
- PyTorch
- PyYAML
- DVC (with Google Drive integration)

Install dependencies by running:

```bash
pip install -r requirements.txt
pip install dvc[gdrive]
```

---

## ⚙️ Initial Setup

### Git Pre-commit Hooks

To prevent accidental commits of large files, run this setup once:

```bash
make setup-hooks
```

You can set the maximum file size allowed (default 1GB) via `.env` file:

```bash
PRE_COMMIT_FILE_SIZE_GB=1
```

### DVC (Data Version Control)

This repository uses DVC to manage large datasets and models efficiently.

Detailed setup instructions can be found [here](docs/dvc_guide.md).

Quick-start commands:

```bash
# Initial setup (once)
dvc init
dvc remote add -d storage gdrive://<your-google-drive-folder-id>
git commit -am "Initialize DVC remote"

# Upload checkpoints or data
dvc add models/checkpoints
git add models/checkpoints.dvc
git commit -m "Add checkpoints"
dvc push

# Download checkpoints or data
git pull
dvc pull
```

---

## 🛠 Quick Usage Guide

### Training

Train your model using the command:

```bash
make train
```

### Evaluation

Evaluate your model using:

```bash
make evaluate
```

### Running Tests

Ensure the correctness of your code:

```bash
make test
```

---

## 📁 Project Structure

```
template-nn/
├── config/               # Configuration files
├── data/                 # Data (managed by DVC)
├── models/checkpoints/   # Model checkpoints (managed by DVC)
├── notebooks/            # Exploratory notebooks
├── outputs/              # Outputs and logs
├── scripts/              # Utility scripts
├── src/                  # Source code
│   ├── data/
│   ├── models/
│   ├── train.py
│   ├── evaluate.py
│   └── utils/
├── tests/                # Pytest unit tests
├── Makefile              # Task management and automation
├── requirements.txt      # Python dependencies
└── docs/
    └── dvc_guide.md      # Guide to using DVC
```

---

## ✏️ Citation

If you find this template useful for your research, please consider citing us:

```
Template-NN by Mizuno Group, https://github.com/mizuno-group/template-nn, 2025.
```

Please also notify us if you use this template in a published paper or presentation.

---

## 📝 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more details.

---

## 👥 Authors
- [自分の名前](自分のgithubのリンク)  
    - main contributor  
- [Tadahaya Mizuno](https://github.com/tadahayamiz)  
    - correspondence  

---

## 📧 Contact

If you have any questions or comments, please feel free to create an issue on github here, or email us:  
- {自分のアドレス}  
- tadahaya[at]gmail.com  
    - lead contact  

---