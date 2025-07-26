
# DVCガイド

## 1. DVCの概要

**DVC (Data Version Control)** は, 大容量データやモデルファイルを管理するためのツール.  
端的にはGitにモデルやデータを置けるようになる.  
- Gitにはデータやモデルの情報(.dvcファイル)のが保存
- 実際のデータ・モデルは外部ストレージ(Google Driveなど)に保存
- 容量の大きいデータをGitで管理する問題(リポジトリ肥大化)を解決

## 2. 導入方法

うちではオンプレ, HPC, Colabといくらか環境があるのでそれぞれ記載.  

### オンプレミス環境

Dockerfileに追加：

```dockerfile
RUN pip install dvc[gdrive]
```

ビルド後のコンテナで初期化：

```bash
dvc init
git commit -am "Initialize DVC"
```

### HPC環境

```bash
pip install dvc[gdrive]
dvc init
git commit -am "Initialize DVC"
```

※CUIだが初回認証は別端末でブラウザ認証可能とのこと(後述)

### Google Colab環境

```bash
!pip install dvc[gdrive]
!dvc init
!git commit -am "Initialize DVC"
```

※毎回セルで実行する必要があり

## 3. 使用方法

### オンプレミス環境

初回のみGoogle Drive連携設定：

```bash
dvc remote add -d storage gdrive://<folder-id>
git commit -am "Add DVC remote storage"
```

データやモデルの管理：

```bash
dvc add models/checkpoints
git add models/checkpoints.dvc
git commit -m "Track checkpoints"
dvc push
```

### HPC環境

初回のみGoogle Drive設定・認証：

```bash
dvc remote add -d storage gdrive://<folder-id>
git commit -am "Add DVC remote storage"
```

初回実行時に表示されるURLを別端末ブラウザで開き, 認証コードをターミナルに入力  

その後の使用例：

```bash
git pull
dvc pull
python src/train.py
dvc add models/checkpoints
git add models/checkpoints.dvc
git commit -m "Update checkpoints"
git push
dvc push
```

### Google Colab環境

毎回のセル実行：

```bash
!git clone <repo-url>
%cd <repo-name>
!pip install dvc[gdrive]
!dvc remote add -d storage gdrive://<folder-id>  # 初回のみ
!git pull
!dvc pull
# 学習後
!dvc add models/checkpoints
!git add models/checkpoints.dvc
!git commit -m "Update checkpoints"
!git push
!dvc push
```

## 4. 注意点

### プロジェクトルートについて

- DVCは必ず **Gitリポジトリのルート** で初期化する.  
- リポジトリ直下で `dvc init` を実行する.  

### リモートストレージ (Google Drive)の設定方法詳細

初回の設定方法：

1. Google Driveにフォルダを作成（URLのfolder-idをメモ）
2. リポジトリ直下で設定
```bash
dvc remote add -d storage gdrive://<folder-id>
git add .dvc/config
git commit -m "Configure Google Drive as DVC remote"
```
3. 初回の`dvc push`または`dvc pull`時にブラウザでの認証が必要
- URLをコピーして別端末のブラウザでアクセス
- Google認証後に表示されるコードをコピー
- コードをターミナルに貼り付け

### 認証情報

- 認証情報は `.dvc/tmp/gdrive-user-credentials.json` に保存される.
- 一度認証すれば, その後は認証なしでアクセス可.
- 各ユーザーはそれぞれ認証が必要(チーム共有の場合)


## 5. ToDo
- DVCの登録先をどのように管理するか確定  
- テスト！  