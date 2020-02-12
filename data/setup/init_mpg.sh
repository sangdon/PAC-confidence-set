DATASET_FILE=mpg.tar
## download dataset files
if test -f "$DATASET_FILE"; then
    echo "$DATASET_FILE exist"
else
    wget https://github.com/sangdon/PAC-confidence-set/releases/download/v1.0/mpg.tar
fi

## extract datasets
DATASET_ROOT=demo/conf_set/datasets
if test ! -f "$DATASET_ROOT"; then
   mkdir -p $DATASET_ROOT
fi
tar -xvf mpg.tar -C $DATASET_ROOT

## remove tar files
rm mpg.tar
