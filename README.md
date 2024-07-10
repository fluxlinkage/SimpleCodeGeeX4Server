# SimpleCodeGeeX4Server
非常简单的CodeGeeX4模型API服务端，用于配合[插件](https://github.com/fluxlinkage/CodeGeeX4-QtCreator-Plugin)使用。

修改自[CodeGeeX2项目](https://github.com/THUDM/CodeGeeX2)中的[示例](https://github.com/THUDM/CodeGeeX2/blob/main/demo/run_demo.py)。

基本用法：

``` sh
python server.py --model-path <模型所在目录> --port <端口号>
```

之后，可以`POST`一个JSON给`/v1/completions`，得到推理结果。