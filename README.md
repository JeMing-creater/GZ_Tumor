# 广肿 MR 数据实验


## get project
```
git clone https://github.com/JeMing-creater/GZ_Tumor.git
```

## requirements
```
pip install -r requirements.txt
```

## training
single device training for GCM classification.
```
python3 train_classify_GCM.py
```
multi-devices training, user need to rewrite running target in this .sh flie.
```
sh run.sh
```

# tensorboard
```
tensorboard --logdir=/logs
```
