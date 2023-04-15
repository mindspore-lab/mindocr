
## About configs

This folder contains the configurations including 
    - model definition
    - training recipes
    - pretrained weights 
    - reported performance
    for all models trained with MindOCR.  

## Model Export 

To convert a pretrained model from mindspore checkpoint format to [MindIR](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/design/mindir.html) format for deployment, please use the `tools/export.py` script. 

``` shell
# convert dbnet_resnet50 with pretrained weights to MindIR format 
python tools/export.py --model_name dbnet_resnet50  --pretrained 

# convert dbnet_resnet50 loaded with weights to MindIR format 
python tools/export.py --model_name dbnet_resnet50  --ckpt_load_path /path/to/checkpoint 
```

For more usage, run `python tools/export.py -h`.

I
