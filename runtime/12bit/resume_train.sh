RUN_BIN=../../build/tools
SCRIPT=../../runtime
ITER=80000 # resume training from pre-trained model of ITER iterations
HASH_CODE_BIT=12

echo "Resume training ${HASH_CODE_BIT}bit deep neural network..."

GLOG_logtostderr=0 ${RUN_BIN}/caffe.bin train \
--solver=${SCRIPT}/${HASH_CODE_BIT}bit/train${HASH_CODE_BIT}_solver.prototxt \
--snapshot=${SCRIPT}/model/${HASH_CODE_BIT}bit_iter_${ITER}.solverstate
