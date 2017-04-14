#ifndef CAFFE_TRIPLET_RANKING_HINGE_LOSS_LAYER_HPP_
#define CAFFE_TRIPLET_RANKING_HINGE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
* @ added by Fuchen Long, used in feature learning
*
* TODO(dox): thorough documentation for Forward, Backward, and proto params.
*/

template <typename Dtype>
class TripletRankingHingeLossLayer : public LossLayer<Dtype>{
public:
	explicit TripletRankingHingeLossLayer(const LayerParameter& param)
		:LossLayer<Dtype>(param), diff_() {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual inline int ExactNumBottomBlobs() const { return 3; }
	virtual inline const char* type() const{ return "TripletRankingHingeLoss"; }
	virtual inline bool AllowForceBackward(const int bottom_index) const
	{
		return bottom_index != 3;

	}
protected:
	/// @copydoc TripletRankingHingeLossLayer
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_dowm, const vector<Blob<Dtype>*>& bottom);
	int dim_;
	int batch_;
	Dtype margin;
	Blob<Dtype> diff_;
	Blob<Dtype> dist_sq_;
	Blob<Dtype> diff_sub_or_si; // F-F+
	Blob<Dtype> diff_sub_or_di; // F-F-
	Blob<Dtype> diff_pow_or_si; // ||F-F+||2
	Blob<Dtype> diff_pow_or_di; // ||F-F-||2
	Blob<Dtype> gradient;
};

} // namespace caffe

#endif