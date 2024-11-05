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
urlを発行し、カーネルに登録する
a. urlを取得する
以下のコマンドを実行し、得られるurlをコピーする
```
jupyter notebook --no-browser --port=8888
```
b. 実行時に要求されるカーネルにaで取得したurlを入力する