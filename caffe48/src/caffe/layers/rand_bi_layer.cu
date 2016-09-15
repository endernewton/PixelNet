#include <vector>

#include "caffe/layers/rand_bi_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"

namespace caffe {

template <typename Dtype>
__global__ void RandBIForward(const int N, const Dtype* bottom_data, const Dtype* bottom_layer, Dtype* top_data,
  const int offset, const int comp, const Dtype padding, const int n_channels, const int ch,
  const int pixels, const int width) {
  // so index is the counter, and n is the range
  CUDA_KERNEL_LOOP(index, N) {
    int j = index * 3;
    int s = index * n_channels + offset;
    int n = static_cast<int>(bottom_data[j++]);
    Dtype y = bottom_data[j++];
    Dtype x = bottom_data[j];
    Dtype ty = (y-padding)/comp;
    Dtype tx = (x-padding)/comp;
    int tx1 = static_cast<int>(floor(tx));
    int ty1 = static_cast<int>(floor(ty));
    int tx2 = static_cast<int>(ceil(tx));
    int ty2 = static_cast<int>(ceil(ty));
    Dtype rx, ry;
    int mind = (n * pixels + ty1 * width + tx1) * N + index;
    if ((tx1 == tx2) && (ty1 == ty2)) {
      int init = n * ch * pixels + ty1 * width + tx1;
      top_data[s++] = bottom_layer[init];
      for (int c = 1; c < ch; c++) {
        init += pixels;
        top_data[s++] = bottom_layer[init];
      }
    } else if (ty1 == ty2) {
      rx = tx - tx1;
      int init1 = n * ch * pixels + ty1 * width + tx1;
      int init2 = init1 + 1;
      top_data[s++] = bottom_layer[init1] * (1-rx) + bottom_layer[init2] * rx;
      for (int c = 1; c < ch; c++) {
        init1 += pixels;
        init2 += pixels;
        top_data[s++] = bottom_layer[init1] * (1-rx) + bottom_layer[init2] * rx;
      }
    } else if (tx1 == tx2) {
      ry = ty - ty1;
      int init1 = n * ch * pixels + ty1 * width + tx1;
      int init2 = init1 + width;
      top_data[s++] = bottom_layer[init1] * (1-ry) + bottom_layer[init2] * ry;
      for (int c = 1; c < ch; c++) {
        init1 += pixels;
        init2 += pixels;
        top_data[s++] = bottom_layer[init1] * (1-ry) + bottom_layer[init2] * ry;
      }
    } else {
      rx = tx - tx1;
      ry = ty - ty1;
      int init11 = n * ch * pixels + ty1 * width + tx1;
      int init12 = init11 + 1;
      int init21 = init11 + width;
      int init22 = init21 + 1;
      top_data[s++] = (bottom_layer[init11] * (1-ry) + bottom_layer[init21] * ry) * (1-rx) +
            (bottom_layer[init12] * (1-ry) + bottom_layer[init22] * ry) * rx;
      for (int c = 1; c < ch; c++) {
        init11 += pixels;
        init12 += pixels;
        init21 += pixels;
        init22 += pixels;
        top_data[s++] = (bottom_layer[init11] * (1-ry) + bottom_layer[init21] * ry) * (1-rx) +
              (bottom_layer[init12] * (1-ry) + bottom_layer[init22] * ry) * rx;
      }
    }
  }
}

template <typename Dtype>
void RandBILayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* bottom_data = bottom[n_hblobs_]->gpu_data();

  int offset = 0;
  for (int b = 0; b < n_hblobs_; b++) {
    int ch = bottom[b]->channels();

    RandBIForward<Dtype><<<CAFFE_GET_BLOCKS(N_), CAFFE_CUDA_NUM_THREADS>>>
      (N_, bottom_data, bottom[b]->gpu_data(), top_data,
      offset, comp_[b], padding_[b], n_channels_, ch, 
      pixels_[b], width_[b]);

    CUDA_POST_KERNEL_CHECK;

    offset += ch;
  }
}

template <typename Dtype>
__global__ void RandBIBackward(const int N, const Dtype* bottom_data, Dtype* bottom_layer, const Dtype* top_diff,
  const int offset, const int comp, const Dtype padding, const int n_channels, const int ch,
  const int pixels, const int width) {
  // so index is the counter, and n is the range
  CUDA_KERNEL_LOOP(index, N) {
    int j = index * 3;
    int s = index * n_channels + offset;
    int n = static_cast<int>(bottom_data[j++]);
    Dtype y = bottom_data[j++];
    Dtype x = bottom_data[j];
    Dtype ty = (y-padding)/comp;
    Dtype tx = (x-padding)/comp;
    int tx1 = static_cast<int>(floor(tx));
    int ty1 = static_cast<int>(floor(ty));
    int tx2 = static_cast<int>(ceil(tx));
    int ty2 = static_cast<int>(ceil(ty));
    Dtype rx, ry;
    if ((tx1 == tx2) && (ty1 == ty2)) {
      int init = n * ch * pixels + ty1 * width + tx1;
      caffe_gpu_atomic_add(top_diff[s++], bottom_layer + init);
      for (int c = 1; c < ch; c++) {
        init += pixels;
        caffe_gpu_atomic_add(top_diff[s++], bottom_layer + init);
      }
    } else if (ty1 == ty2) {
      rx = tx - tx1;
      int init1 = n * ch * pixels + ty1 * width + tx1;
      int init2 = init1 + 1;
      caffe_gpu_atomic_add(top_diff[s] * (1-rx), bottom_layer + init1);
      caffe_gpu_atomic_add(top_diff[s++] * rx, bottom_layer + init2);
      for (int c = 1; c < ch; c++) {
        init1 += pixels;
        init2 += pixels;
        caffe_gpu_atomic_add(top_diff[s] * (1-rx), bottom_layer + init1);
        caffe_gpu_atomic_add(top_diff[s++] * rx, bottom_layer + init2);
      }
    } else if (tx1 == tx2) {
      ry = ty - ty1;
      int init1 = n * ch * pixels + ty1 * width + tx1;
      int init2 = init1 + width;
      caffe_gpu_atomic_add(top_diff[s] * (1-ry), bottom_layer + init1);
      caffe_gpu_atomic_add(top_diff[s++] * ry, bottom_layer + init2);
      for (int c = 1; c < ch; c++) {
        init1 += pixels;
        init2 += pixels;
        caffe_gpu_atomic_add(top_diff[s] * (1-ry), bottom_layer + init1);
        caffe_gpu_atomic_add(top_diff[s++] * ry, bottom_layer + init2);
      }
    } else {
      rx = tx - tx1;
      ry = ty - ty1;
      int init11 = n * ch * pixels + ty1 * width + tx1;
      int init12 = init11 + 1;
      int init21 = init11 + width;
      int init22 = init21 + 1;
      caffe_gpu_atomic_add(top_diff[s] * (1-ry) * (1-rx), bottom_layer + init11);
      caffe_gpu_atomic_add(top_diff[s] * ry * (1-rx), bottom_layer + init21);
      caffe_gpu_atomic_add(top_diff[s] * (1-ry) * rx, bottom_layer + init12);
      caffe_gpu_atomic_add(top_diff[s++] * ry * rx, bottom_layer + init22);
      for (int c = 1; c < ch; c++) {
        init11 += pixels;
        init12 += pixels;
        init21 += pixels;
        init22 += pixels;
        caffe_gpu_atomic_add(top_diff[s] * (1-ry) * (1-rx), bottom_layer + init11);
        caffe_gpu_atomic_add(top_diff[s] * ry * (1-rx), bottom_layer + init21);
        caffe_gpu_atomic_add(top_diff[s] * (1-ry) * rx, bottom_layer + init12);
        caffe_gpu_atomic_add(top_diff[s++] * ry * rx, bottom_layer + init22);
      }
    }
  }
}

template <typename Dtype>
void RandBILayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[n_hblobs_]->gpu_data();
    
    int offset = 0;
    for (int b = 0; b < n_hblobs_; b++) {
      Dtype* bottom_layer = bottom[b]->mutable_gpu_diff();
      caffe_gpu_set(bottom[b]->count(), Dtype(0.), bottom_layer);
      int ch = bottom[b]->channels();

      RandBIBackward<Dtype><<<CAFFE_GET_BLOCKS(N_), CAFFE_CUDA_NUM_THREADS>>>
        (N_, bottom_data, bottom_layer, top_diff, 
        offset, comp_[b], padding_[b], n_channels_, ch,
        pixels_[b], width_[b]);

      CUDA_POST_KERNEL_CHECK;

      offset += ch;
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(RandBILayer);

}  // namespace caffe
