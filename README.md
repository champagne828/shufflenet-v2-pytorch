
# ShuffleNetV2 pytorch

This project implements ShuffleNetV2 written in pytorch. 

## Performance

| model | Top 1 | Top 5 | Top 1 reported in paper |
| --- | --- | --- | --- |
| 1.0x width | 0.686 | 0.882 | 0.694 |

## Train

```
python main.py $imagenet --symbol 1 --gpu 0,1,2,3 -b 1024 -j 16
```

## Evaluate

```
python main.py $imagenet --symbol 1 --gpu 0,1,2,3 -b 1024 -j 16 --evaluate --resume
```

## Acknowledgement

[CondenseNet](https://github.com/ShichenLiu/CondenseNet)

[Shufflenet-v2-Pytorch](https://github.com/ericsun99/Shufflenet-v2-Pytorch)
