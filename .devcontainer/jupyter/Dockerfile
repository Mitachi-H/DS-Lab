# https://note.com/k_fukunaka/n/nabb5cd028595
FROM jupyter/datascience-notebook:x86_64-ubuntu-22.04

# メタ情報
LABEL maintainer="Hiroya Mitachi"
LABEL version="1.0"
LABEL description="vscodeでdocker+jupyterを使う"

# 作業ディレクトリを変更
WORKDIR /home/jovyan/work

# ファイルのコピー
COPY requirements.txt ${PWD}

# pip自体のアップデート
RUN pip install -U pip

# pythonのパッケージをインストール
RUN pip install -r requirements.txt
