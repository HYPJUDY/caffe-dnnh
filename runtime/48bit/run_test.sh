RUN_BIN=../../build/tools
SCRIPT=../../runtime
ITER=35000
HASH_CODE_BIT=48
echo "Run testing ${HASH_CODE_BIT}bit deep neural network..."

GLOG_logtostderr=0 ${RUN_BIN}/caffe.bin test \
  -model ${SCRIPT}/${HASH_CODE_BIT}bit/test${HASH_CODE_BIT}_query.prototxt -weights \
  ${SCRIPT}/model/${HASH_CODE_BIT}bit_iter_${ITER}.caffemodel -iterations 1 # query 1k

GLOG_logtostderr=0 ${RUN_BIN}/caffe.bin test \
  -model ${SCRIPT}/${HASH_CODE_BIT}bit/test${HASH_CODE_BIT}_pool.prototxt -weights \
  ${SCRIPT}/model/${HASH_CODE_BIT}bit_iter_${ITER}.caffemodel -iterations 2 # pool 59k

cd ${SCRIPT}
g++ -std=c++11 evaluate_map.cpp
./a.out
