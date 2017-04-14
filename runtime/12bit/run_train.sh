RUN_BIN=../../build/tools
SCRIPT=../../runtime
HASH_CODE_BIT=12

echo "Run training ${HASH_CODE_BIT}bit deep neural network..."

GLOG_logtostderr=0 ${RUN_BIN}/caffe.bin train \
  --solver=${SCRIPT}/${HASH_CODE_BIT}bit/train${HASH_CODE_BIT}_solver.prototxt
