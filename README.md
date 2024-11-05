# DS-Lab

## 実行方法（not backgroud）

### 1. vscodeの拡張機能: Remote - SSHでssh接続
これにより、Remote環境でvscodeを実行できるようになる。
### 2. 環境で仮想環境venv構築、pipでjupyter ダウンロード
これにより、jupyter notebookを利用できるようになる。

a. Python 3で仮想環境を作成
```
python3 -m venv myenv
source myenv/bin/activate
```
b. pipのアップグレード
```
pip install --upgrade pip
```
c. Jupyter Notebookのインストール
```
pip install jupyter
```
### 3. 実行
a. urlを取得する

以下のコマンドを実行し、得られるurlをコピーする
```
jupyter notebook --no-browser --port=8888
```
b. VSCodeでの接続設定

VSCodeのJupyterカーネル選択で「既存のJupyterサーバー」を指定し、先ほどのJupyterサーバーURLを入力する

## 実行方法（backgroud）
### 1. 下準備
実行方法（not backgroud）の1, 2参照

### 2. 実行
a. nohupコマンドを使ってJupyter Notebookを起動
```
nohup jupyter notebook &
```
b. VSCodeでの接続設定

VSCodeのJupyterカーネル選択で「既存のJupyterサーバー」を指定し、先ほどのJupyterサーバーURLを入力する

参考：サーバーの終了
```
jupyter notebook list
jupyter notebook stop (port_number_running)
```