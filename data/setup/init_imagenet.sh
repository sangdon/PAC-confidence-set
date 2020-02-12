DATASET_FILE=imagenet_resnet152.tar.gz
## download dataset files
if test -f "$DATASET_FILE"; then
    echo "$DATASET_FILE exist"
else
    wget https://github.com/sangdon/PAC-confidence-set/releases/download/v1.0/imagenet_resnet152.tar.aa
    wget https://github.com/sangdon/PAC-confidence-set/releases/download/v1.0/imagenet_resnet152.tar.ab
    wget https://github.com/sangdon/PAC-confidence-set/releases/download/v1.0/imagenet_resnet152.tar.ac
    wget https://github.com/sangdon/PAC-confidence-set/releases/download/v1.0/imagenet_resnet152.tar.ad
    wget https://github.com/sangdon/PAC-confidence-set/releases/download/v1.0/imagenet_resnet152.tar.ae
    cat imagenet_resnet152.* > imagenet_resnet152.tar.gz
fi

## extract datasets
DATASET_ROOT=demo/conf_set/datasets
if test ! -f "$DATASET_ROOT"; then
   mkdir -p $DATASET_ROOT
fi
tar -xzvf $DATASET_FILE -C $DATASET_ROOT


