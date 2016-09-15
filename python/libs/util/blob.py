import numpy as np
import cv2

def prep_im_for_blob_segdb(im, im_scale, im_flip, pixel_means):
    im = im.astype(np.float32, copy=False)
    if im_flip:
        im = im[:, ::-1, :]
    im -= pixel_means
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    return im

def prep_gt_for_blob_segdb(gt, im_scale, im_flip):
    # make sure it is read as a gray-scale image before calling
    gt = gt.astype(np.float32, copy=False)
    if im_flip:
        gt = gt[:, ::-1]
    gt = cv2.resize(gt, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_NEAREST)
    return gt

# margin is added to the size of the image
# pad is put at the top-left corner
def im_list_to_blob_segdb(ims, pad, margin):
    max_shape = [x + margin for x in np.array([im.shape for im in ims]).max(axis=0)]
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, pad:im.shape[0]+pad, pad:im.shape[1]+pad, :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    # X: how to transpose a matrix in python
    blob = blob.transpose(channel_swap)
    return blob
