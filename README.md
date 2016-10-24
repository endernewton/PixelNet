# PixelNet

- Please note that this is rough version of the PixelNet code we are releasing. The code is for training only. This code works on our server, but you may need to change the paths here and there to get the python interface work; You may also need to download the networks provided by (fast) RCNN to fine tune the weights from if you want to do so.

- I guess the most important part in this release is the caffe layers, it is called rand_bi_layer. We have tested the CPU implementation extensively, but the GPU implementation was recently added but not fully tested. So be careful with GPU version (In fact, it would be wonderful if you can report bug to the GPU version if you notice anything strange).
