DATASET_FILE=halfcheetah.tar

## download dataset files
if test -f "$DATASET_FILE"; then
    echo "$DATASET_FILE exist"
else
    wget https://github.com/sangdon/PAC-confidence-set/releases/download/v1.0/halfcheetah.tar
fi

## extract datasets
DATASET_ROOT=demo/conf_set/datasets
if test ! -f "$DATASET_ROOT"; then
   mkdir -p $DATASET_ROOT
fi
tar -xvf $DATASET_FILE -C $DATASET_ROOT

