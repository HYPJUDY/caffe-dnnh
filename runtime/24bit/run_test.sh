RUN_BIN=../../build/tools
SCRIPT=../../runtime
ITER=80000
HASH_CODE_BIT=24
echo "Run testing ${HASH_CODE_BIT}bit deep neural network..."

GLOG_logtostderr=0 ${RUN_BIN}/caffe.bin test \
  -gpu=0 -model ${SCRIPT}/${HASH_CODE_BIT}bit/test${HASH_CODE_BIT}_query.prototxt -weights \
  ${SCRIPT}/model/${HASH_CODE_BIT}bit_iter_${ITER}.caffemodel -iterations 10 # query 1k=10*100(batch size)

GLOG_logtostderr=0 ${RUN_BIN}/caffe.bin test \
  -gpu=0 -model ${SCRIPT}/${HASH_CODE_BIT}bit/test${HASH_CODE_BIT}_pool.prototxt -weights \
  ${SCRIPT}/model/${HASH_CODE_BIT}bit_iter_${ITER}.caffemodel -iterations 590 # pool 59k=590*100

cd ${SCRIPT}
g++ -std=c++11 evaluate_map.cpp
./a.out
