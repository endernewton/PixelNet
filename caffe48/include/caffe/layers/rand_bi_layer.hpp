#ifndef CAFFE_RAND_BI_LAYER_HPP_
#define CAFFE_RAND_BI_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
  
template <typename Dtype>
class RandBILayer : public Layer<Dtype> {
 public:
  explicit RandBILayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RandBI"; }
  // should have at least 1 as input, and the sample as 2nd input
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  //shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 // no. of data points 
 int N_;
 // no. of images
 int n_;
 // number of bottom blobs containing hypercol data 
 int n_hblobs_;
 // number of channels in the hypercol data --
 int n_channels_;
 // height, width
 vector<int> height_;
 vector<int> width_;
 vector<int> pixels_;
 // compression rate
 vector<int> comp_;
 // if do padding
 bool padded_;
 vector<Dtype> padding_;
};

}  // namespace caffe

#endif  // CAFFE_RAND_BI_LAYER_HPP_
