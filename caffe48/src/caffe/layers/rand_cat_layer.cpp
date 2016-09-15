#include <vector>

#include "caffe/layers/rand_cat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RandCatLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // the no. of input layers
  n_hblobs_ = bottom.size() - 1;
  // get the last layer, which should be the indexes
  CHECK_EQ(bottom[n_hblobs_]->num_axes(), 2);
  CHECK_EQ(bottom[n_hblobs_]->shape(1), 3);
  N_ = bottom[n_hblobs_]->shape(0);
  CHECK_GT(N_,0);
  // finally check if the data shape is the same
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  for (int i = 1; i < n_hblobs_; i++) {
    CHECK_EQ(bottom[i]->height(), height_);
    CHECK_EQ(bottom[i]->width(), width_);
  }
  increment_ = height_ * width_;
}

template <typename Dtype>
void RandCatLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  n_channels_ = 0;
  for (int i = 0; i < n_hblobs_; i++) {
    n_channels_ += bottom[i]->channels();
  }
  N_ = bottom[n_hblobs_]->shape(0);
  CHECK_GT(N_,0);
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  for (int i = 1; i < n_hblobs_; i++) {
    CHECK_EQ(bottom[i]->height(), height_);
    CHECK_EQ(bottom[i]->width(), width_);
  }
  increment_ = height_ * width_;
  vector<int> top_shape(2);
  top_shape[0] = N_;
  top_shape[1] = n_channels_;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void RandCatLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[n_hblobs_]->cpu_data();
  vector<const Dtype*> bottom_layers(n_hblobs_);
  for (int i = 0; i < n_hblobs_; i++) {
    bottom_layers[i] = bottom[i]->cpu_data();
  }

  int i = 0;
  int j = 0;
  int n, x, y;
  int s = 0; // index for the output
  for (; i < N_; i++) {
    n = int(bottom_data[j++]);
    y = int(bottom_data[j++]);
    x = int(bottom_data[j++]);
    // then find the corresponding locations
    for (int b = 0; b < n_hblobs_; b++) {
      int init = n * bottom[b]->channels() * increment_ + y * width_ + x;
      top_data[s++] = bottom_layers[b][init];
      for (int c = 1; c < bottom[b]->channels(); c++) {
        init += increment_;
        top_data[s++] = bottom_layers[b][init];
      }
    }
  }
}

template <typename Dtype>
void RandCatLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = bottom[n_hblobs_]->cpu_data();
  vector<Dtype*> bottom_layers(n_hblobs_);
  for (int i = 0; i < n_hblobs_; i++) {
    bottom_layers[i] = bottom[i]->mutable_cpu_diff();
    caffe_set(bottom[i]->count(), Dtype(0.), bottom_layers[i]);
  }

  if (propagate_down[0]) {
    int i = 0;
    int j = 0;
    int n, x, y;
    int s = 0; // index for the output
    for (; i < N_; i++) {
      n = int(bottom_data[j++]);
      y = int(bottom_data[j++]);
      x = int(bottom_data[j++]);
      // then find the corresponding locations
      for (int b = 0; b < n_hblobs_; b++) {
        int init = n * bottom[b]->channels() * increment_ + y * width_ + x;
        bottom_layers[b][init] = top_diff[s++];
        for (int c = 1; c < bottom[b]->channels(); c++) {
          init += increment_;
          bottom_layers[b][init] = top_diff[s++];
        }
      }
    }
  }
}

// #ifdef CPU_ONLY
// STUB_GPU(RandCatLayer);
// #endif

INSTANTIATE_CLASS(RandCatLayer);
REGISTER_LAYER_CLASS(RandCat);

}  // namespace caffe
