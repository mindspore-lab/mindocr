
## About configs

This folder contains the configurations including 
    - model definition
    - training recipes
    - pretrained weights 
    - reported performance
    for all models trained with MindOCR.  

## Model Export 

To convert a trained model from mindspore checkpoint format to [MindIR](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/design/mindir.html) format for inference and deployment, please use the `tools/export.py` script. 

``` shell
# Export mindir of model `dbnet_resnet50` by downloading online ckpt
python tools/export.py --model_name dbnet_resnet50 --data_shape 736 1280

# Export mindir of model `dbnet_resnet50` by loading local ckpt
python tools/export.py --model_name dbnet_resnet50 --data_shape 736 1280 --local_ckpt_path /path/to/local_ckpt

```

For more usage, run `python tools/export.py -h`.
