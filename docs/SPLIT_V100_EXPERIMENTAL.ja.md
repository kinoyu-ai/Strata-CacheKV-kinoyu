# V100用Split Engineの実験ビルド

通常のStrataとは別の実行ファイルを作る手順です。上流のSplit Engineには既にsm_70用の計算経路があり、このブランチはそれをLinux・CUDA 12.xで独立してビルドします。通常起動の設定やAutoの対象GPUは変更しません。

```sh
B="$PWD/build-split-v100" CUDA=/usr/local/cuda-12.9 \
GG=/path/to/pinned/llama.cpp JOBS=4 packaging/build_split_v100.sh
```

`GG`には`third_party/ggml/VERSION.txt`に記載されたコミットのllama.cppを指定してください。出力は`build-split-v100/split/strata-split-2`です。2枚構成、MTP用で、DFlash2はこの独立ビルドには含みません。CUDA 13ではV100向けのコンパイルができません。

## 現在の制約

**V100でビルドできることと、モデル全体を動かせることは別です。** 通常エンジンに追加したOptaneのNgramキャッシュやRAM→ディスクの会話KV退避は、この別エンジンには実装していません。KV形式も通常エンジンの`k8v4`とは異なります。

検証機のV100 16 GiB × 2、RAM約56 GiB、Qwen3.8-Flash-Next Unsloth v3.1 UD-IQ3_XXSでは、上流の適合判定は1枚あたり約32.75 GiBのVRAMを要求し、起動対象外になりました。エキスパートをRAMへ逃がす内部経路もありますが、全エキスパートのホストコピー約47.5 GiBと埋め込み等約27.3 GiBだけで搭載RAMを超えます。適合判定を外して通常運用に切り替えることはしていません。

Split Engineの起動方法、モデルやメモリ条件は[上流の説明](SPLIT_ENGINE.md)を参照してください。V100をAutoの検証済みGPUに追加するものではありません。

## 行った確認

CUDA 12.9 / sm_70で独立MTP版をビルドし、2枚のV100それぞれで実モデルの重みタイルをgguf-pyの参照値と比較しました。カード適合・expert配置のCPUテスト、CUDA peer API未参照の確認も通過しています。DFlash2なしのビルドで残っていたリンク参照を修正し、そのビルドで`--draft dflash2`を指定した場合はモデル読み込み前に明示的に拒否します。DFlash2ありの経路は構文検査のみです。

**モデル全体の推論、回答品質、速度、MTPの実行検証は行っていません。** 上記のメモリ不足があるためです。重みタイルの一致だけで、Split Engineの全計算がV100で正しいことを保証しません。
