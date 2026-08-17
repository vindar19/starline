---
license: creativeml-openrail-m
---

</p>

# controlnet_lineaetXL
- これはstable DiffusionのSDXLにおいて線画から色塗りを行うコントロールネットです。Lineartプリプロセッサで使用することができます。

# 使い方
コントロールネットに線画や色塗り済みの画像をセットします。

プリプロセッサはLineartに設定してください。線が太いとうまく作動しないため推奨はlineart_anime_denoiseまたはlineart_animeです。

白地に黒線の線画を用意した場合はinvert (from white bg & black line)プリプロセッサを使用してください。

fp16バージョンの推奨モデルはanimagineXL3.1です。pony系列ではあまりうまく動作しません。

またLoraタイプ(400MB)の方はanimagineXL3.1専用です。

- ![](test1.png)

線画から色塗りをした場合はこのようになります。

- ![](test2.png)

また、色塗りをした画像から色だけを塗りなおす場合はこのようになります。

- ![](test3.png)