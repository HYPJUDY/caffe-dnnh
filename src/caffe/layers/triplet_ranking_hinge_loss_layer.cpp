#include <algorithm>
#include <vector>

#include "caffe/layers/triplet_ranking_hinge_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

// added by fuchen long for hashing coding
namespace caffe {
template <typename Dtype>
void TripletRankingHingeLossLayer<Dtype>::LayerSetUp(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	LossLayer<Dtype>::LayerSetUp(bottom, top);
	dim_ = this->layer_param_.triplet_ranking_hinge_loss_param().dim();
	margin = this->layer_param_.triplet_ranking_hinge_loss_param().margin();
	batch_ = bottom[0]->num();
	CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
	CHECK_EQ(bottom[1]->channels(), bottom[2]->channels());
	CHECK_EQ(bottom[0]->channels(), dim_); //check the dim_ension
	CHECK_EQ(bottom[1]->channels(), dim_);
	CHECK_EQ(bottom[2]->channels(), dim_);
	diff_.Reshape(bottom[0]->num(), 1, 1, 1);
	dist_sq_.Reshape(bottom[0]->num(), 1, 1, 1);
	diff_sub_or_si.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1); // F-F+
	diff_sub_or_di.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1); // F-F-
	diff_pow_or_si.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1); // Pow (F-F+)
	diff_pow_or_di.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1); // Pow (F-F-)
	gradient.Reshape(1, bottom[0]->channels(), 1, 1);
}



template <typename Dtype>
void TripletRankingHingeLossLayer<Dtype>::Forward_cpu(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	int dim_v = batch_*dim_;
	const Dtype* sub_or_si;
	const Dtype* sub_or_di;
	Dtype Tripletlosstotal(0.0);
	//The triplet ranking loss
	for(int i = 0; i < dim_; ++i) {
		LOG(INFO) << bottom[0]->cpu_data()[i] << " " << bottom[1]->cpu_data()[i] << " " << bottom[2]->cpu_data()[i];
	}
	caffe_sub(dim_v, bottom[0]->cpu_data(), bottom[1]->cpu_data(), diff_sub_or_si.mutable_cpu_data()); // F-F+
	caffe_sub(dim_v, bottom[0]->cpu_data(), bottom[2]->cpu_data(), diff_sub_or_di.mutable_cpu_data()); // F-F-
	caffe_powx(dim_v, diff_sub_or_si.cpu_data(), Dtype(2.0), diff_pow_or_si.mutable_cpu_data());		  //Pow
	caffe_powx(dim_v, diff_sub_or_di.cpu_data(), Dtype(2.0), diff_pow_or_di.mutable_cpu_data());       //Pow

	for (int n = 0; n < batch_; n++){
		sub_or_si = diff_pow_or_si.cpu_data() + diff_pow_or_si.offset(n);
		sub_or_di = diff_pow_or_di.cpu_data() + diff_pow_or_di.offset(n);
		Dtype result1 = 0;
		Dtype result2 = 0;
		result1 = caffe_cpu_asum(dim_, sub_or_si); // absolute sum
		result2 = caffe_cpu_asum(dim_, sub_or_di);
		Dtype loss(0.0);
		loss = std::max(margin + result1 - result2, Dtype(0));// compute the loss
		if(n == 50) LOG(INFO) << "result1 " << result1 << ", result2 " << result2 << ", loss " << loss;
		diff_.mutable_cpu_data()[n] = loss; // save the loss[i]
	}
	for (int k = 0; k < batch_; k++){

		dist_sq_.mutable_cpu_data()[k] = diff_.cpu_data()[k];// save the loss[i] for BP
		Tripletlosstotal += dist_sq_.cpu_data()[k];
	}
	Tripletlosstotal = Tripletlosstotal / static_cast<Dtype>(bottom[0]->num()); //get the average loss
	top[0]->mutable_cpu_data()[0] = Tripletlosstotal;
}

template <typename Dtype>
void TripletRankingHingeLossLayer<Dtype>::Backward_cpu(
	const vector<Blob<Dtype>*>& top, const vector<bool> &propagate_down,
	const vector<Blob<Dtype>*>& bottom){
	const Dtype* orignalcode;
	const Dtype* similarcode;
	const Dtype* diffrcode;
	if (propagate_down[0]) {
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < batch_; ++j){
				Dtype* bout = bottom[i]->mutable_cpu_diff();
				orignalcode = bottom[0]->cpu_data() + bottom[0]->offset(j);
				similarcode = bottom[1]->cpu_data() + bottom[1]->offset(j);
				diffrcode = bottom[2]->cpu_data() + bottom[2]->offset(j);
				if (i == 0){
					if (dist_sq_.cpu_data()[j]>Dtype(0.0)){
						caffe_sub(dim_, diffrcode, similarcode,
							gradient.mutable_cpu_data());// the distance of F- and F+
						caffe_scal(dim_, Dtype(2) / Dtype(batch_),
							gradient.mutable_cpu_data());
					}
					else
						caffe_set(dim_, Dtype(0.0), gradient.mutable_cpu_data());
				}
				if (i == 1){
					if (dist_sq_.cpu_data()[j] > Dtype(0.0)){
						caffe_sub(dim_, similarcode, orignalcode, 
							gradient.mutable_cpu_data());// the distance of F+ and F
						caffe_scal(dim_, Dtype(2) / Dtype(batch_),
							gradient.mutable_cpu_data());
					}
					else
						caffe_set(dim_, Dtype(0.0), gradient.mutable_cpu_data());
				}
				if (i == 2){
					if (dist_sq_.cpu_data()[j] > Dtype(0.0)){
						caffe_sub(dim_, orignalcode, diffrcode,
							gradient.mutable_cpu_data()); // the distance of F and F-
						caffe_scal(dim_, Dtype(2) / Dtype(batch_),
							gradient.mutable_cpu_data());
					}
					else
						caffe_set(dim_, Dtype(0.0), gradient.mutable_cpu_data());
				}
				caffe_scal(dim_, Dtype(2.0), gradient.mutable_cpu_data());
				caffe_copy(dim_, gradient.cpu_data(), bout + (j*dim_));
			}
		}
	}
}


INSTANTIATE_CLASS(TripletRankingHingeLossLayer);
REGISTER_LAYER_CLASS(TripletRankingHingeLoss);

}  // namespace caffe