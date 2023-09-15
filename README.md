# このリポジトリについて
これは、[DaXBench](https://github.com/AdaCompNUS/DaXBench)と呼ばれる柔軟物を扱える微分可能シミュレータ環境を
Dockerコンテナ上で構築するためのリポジトリです。

# インストール方法
## Docker&Docker-composeのインストール
公式サイトを確認してインストールしてください。

## リポジトリのクローン
```bash
git clone https://github.com/HoneyMack/docker_DaXBench.git
cd [path to docker_DaXBench]/work
# DaXBenchのリポジトリをクローンして、特定のコミットにチェックアウトする
git clone https://github.com/AdaCompNUS/DaXBench.git
cd DaXBench
git checkout 75c8ed9
```

## Docker Imageのビルド
```bash
cd [path to docker_DaXBench]
export PROJECT_NAME=hoge
./BUILD_DOCKER_IMAGE.sh
```
上のようにしてビルドすると、`hoge_daxbench`という名前のDocker Imageが作成されます。

## Docker Containerの起動&ログイン
ビルドしたDocker Imageを元に、Docker Containerを起動します。
ここで、環境変数```PROJECT_NAME```には、ビルドで指定したプロジェクト名を指定してください。
```bash
cd [path to docker_DaXBench]
export PROJECT_NAME=hoge
./RUN_DOCKER_CONTAINER.sh
```

## DaxBenchのインストール
docker containerの中で、DaXBenchをインストールします。
```bash
cd /root/work/DaXBench
pip install -e .
pip install protobuf==3.20.0 # apgアルゴリズムを正しく動作させるために、バージョンを指定してインストールし直す
```

