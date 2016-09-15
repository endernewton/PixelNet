#include <vector>
#include <cmath>
#include <cfloat>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/mil_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MILForward(const int n, const Dtype* in, Dtype* out,
    const int channels_, const int context_per_roi_) {
  // so index is the counter, and n is the range
  CUDA_KERNEL_LOOP(index, n) {
    Dtype prob = -FLT_MAX; 
    int i = index / channels_;
    int k = index % channels_;
    int st = i * context_per_roi_ * channels_;
    for(int j = 0; j < context_per_roi_; j++){
      prob = max(prob, in[st+k]);
      st += channels_;
    }
    out[index] = prob;
  }
}

template <typename Dtype>
void MILLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = top[0]->count();

  MILForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, channels_, context_per_roi_);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void MILBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, const Dtype* out_data, 
    const int channels_, const int context_per_roi_) {
  CUDA_KERNEL_LOOP(index, n) {
    int i = index / channels_ / context_per_roi_;
    int k = index % channels_;
    int sc = i * channels_ + k;
    out_diff[index] = in_diff[sc] * (out_data[index] == in_data[sc]);
  }
}

template <typename Dtype>
void MILLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();

    MILBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top_data, bottom_diff, bottom_data, channels_, context_per_roi_);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MILLayer);

}  // namespace caffe
