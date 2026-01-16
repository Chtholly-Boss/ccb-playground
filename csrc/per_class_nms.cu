#include "binding.cuh"

#define CHECK_CUDA_ERROR(call)                                                 \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", __FUNCTION__,          \
              __FILE__, __LINE__, cudaGetErrorString(err));                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

static const int BS = 512;

template <typename T> __device__ void swap(T &a, T &b) {
  T temp = a;
  a = b;
  b = temp;
}

template <typename T> struct Bbox {
  T xmin, ymin, xmax, ymax;
  __device__ Bbox(T xmin, T ymin, T xmax, T ymax)
      : xmin(xmin), ymin(ymin), xmax(xmax), ymax(ymax) {}
  __device__ Bbox() = default;
  __device__ T area(const bool normalized = false, T offset = T(0)) const {
    if (xmax < xmin || ymax < ymin) {
      return T(0);
    } else {
      T width = xmax - xmin;
      T height = ymax - ymin;
      if (normalized) {
        return width * height;
      } else {
        return (width + offset) * (height + offset);
      }
    }
  }
  __device__ void intersect(const Bbox<T> &other,
                            Bbox<T> *intersect_bbox) const {
    if (other.xmin > xmax || other.xmax < xmin || other.ymin > ymax ||
        other.ymax < ymin) {
      intersect_bbox->xmin = T(0);
      intersect_bbox->ymin = T(0);
      intersect_bbox->xmax = T(0);
      intersect_bbox->ymax = T(0);
    } else {
      intersect_bbox->xmin = max(xmin, other.xmin);
      intersect_bbox->ymin = max(ymin, other.ymin);
      intersect_bbox->xmax = min(xmax, other.xmax);
      intersect_bbox->ymax = min(ymax, other.ymax);
    }
  }
};

template <typename T_BBOX>
__device__ float jaccardOverlap(const Bbox<T_BBOX> &bbox1,
                                const Bbox<T_BBOX> &bbox2,
                                const bool normalized, T_BBOX offset) {
  Bbox<T_BBOX> intersect_bbox;
  bbox1.intersect(bbox2, &intersect_bbox);
  float intersect_width, intersect_height;
  if (normalized) {
    intersect_width = intersect_bbox.xmax - intersect_bbox.xmin;
    intersect_height = intersect_bbox.ymax - intersect_bbox.ymin;
  } else {
    intersect_width = intersect_bbox.xmax - intersect_bbox.xmin + offset;
    intersect_height = intersect_bbox.ymax - intersect_bbox.ymin + offset;
  }
  if (intersect_width > 0 && intersect_height > 0) {
    float intersect_size = intersect_width * intersect_height;
    float bbox1_size = bbox1.area(normalized, offset);
    float bbox2_size = bbox2.area(normalized, offset);
    return intersect_size / (bbox1_size + bbox2_size - intersect_size);
  } else {
    return 0.;
  }
}

template <typename T_BBOX>
__device__ void bbox_decode_clip_xywh(const float *roi_ptr,
                                      const float *delta_ptr,
                                      const float s_im_info[3],
                                      Bbox<T_BBOX> &out, bool flipXY = false) {
  float rois_scaled[4];
  for (int i = 0; i < 4; ++i) {
    rois_scaled[i] = roi_ptr[i] / s_im_info[2];
  }
  float width = rois_scaled[2] - rois_scaled[0] + 1.0f;
  float height = rois_scaled[3] - rois_scaled[1] + 1.0f;
  float ctr_x = rois_scaled[0] + 0.5f * (width - 1.0f);
  float ctr_y = rois_scaled[1] + 0.5f * (height - 1.0f);

  float dx = delta_ptr[0];
  float dy = delta_ptr[1];
  float dw = delta_ptr[2];
  float dh = delta_ptr[3];

  float pred_ctr_x = dx * width + ctr_x;
  float pred_ctr_y = dy * height + ctr_y;
  float pred_w = expf(dw) * width;
  float pred_h = expf(dh) * height;

  float offset_w = (pred_w - 1.0f) * 0.5f;
  float offset_h = (pred_h - 1.0f) * 0.5f;
  out.xmin = max(min(pred_ctr_x - offset_w, s_im_info[1] - 1.0f), 0.0f);
  out.ymin = max(min(pred_ctr_y - offset_h, s_im_info[0] - 1.0f), 0.0f);
  out.xmax = max(min(pred_ctr_x + offset_w, s_im_info[1] - 1.0f), 0.0f);
  out.ymax = max(min(pred_ctr_y + offset_h, s_im_info[0] - 1.0f), 0.0f);

  if (flipXY) {
    swap(out.xmin, out.ymin);
    swap(out.xmax, out.ymax);
  }
}

template <typename T_SCORE = float, typename T_BBOX = float>
__global__ void k_per_class_nms(float *nmsed_scores, int *nmsed_indices,
                                float *bbox_delta, float *rois, float *im_info,
                                float *scores, int *indices, int num_rois,
                                int num_classes, int topk, float iou_threshold,
                                bool flipXY = false, bool shareLocation = false,
                                bool isNormalized = false) {
  // Different from TRT implementation, we don't hold bbox in registers, so the
  // max topk is limited by shared memory size.
  // this because register count in different architecture may be alwarys 65536,
  // but shared memory size can increase

  // 128KB --> MAX_TOPK = 5120
  // 256KB --> MAX_TOPK = 10240
  constexpr int MAX_TOPK = 1024;
  constexpr size_t SHM_SIZE =
      sizeof(int) * MAX_TOPK + sizeof(float) * MAX_TOPK +
      sizeof(Bbox<T_BBOX>) * MAX_TOPK + sizeof(float) * 3;
  static_assert(SHM_SIZE <= 128 * 1024, "Exceeded shm size limit");

  __shared__ int s_indices[MAX_TOPK];
  __shared__ float s_scores[MAX_TOPK];
  __shared__ Bbox<T_BBOX> s_bboxes[MAX_TOPK];
  __shared__ float s_im_info[3]; // per-image

  const int tidx = threadIdx.x;
  const int bidx = blockIdx.x;
  if (tidx < 3) {
    s_im_info[tidx] = im_info[tidx];
  }

  int max_read_num = min(num_rois, topk);

  // TODO: Replace by TMA Load on Lauda
  for (int i = tidx; i < max_read_num; i += BS) {
    s_scores[i] = scores[bidx * num_rois + i];
    s_indices[i] = indices[bidx * num_rois + i];
  }
  __syncthreads();

  for (int i = tidx; i < MAX_TOPK; i += BS) {
    if (i < max_read_num) {
      // explicitly using registers, as MIDA compiler may not optimize well
      Bbox<T_BBOX> local_bbox;
      // TODO: consider use bulk_load in Lauda
      int bbox_idx = s_indices[i];
      int roi_idx = shareLocation ? bbox_idx : bbox_idx / num_classes;
      int class_idx = shareLocation ? 0 : bbox_idx % num_classes;
      float *roi_ptr = rois + roi_idx * 4;
      float *delta_ptr = bbox_delta + (roi_idx * num_classes + class_idx) * 4;
      bbox_decode_clip_xywh(roi_ptr, delta_ptr, s_im_info, local_bbox, flipXY);
      // TODO: consider use bulk_store in Lauda
      // for now we don't support roated bbox, so a bbox is represented as
      // (xmin, ymin, xmax, ymax) which is a `float4` in memory, thus we can use
      // bulk store/load in Lauda
      // if we need to support rotated bbox later, we may need to interleave the
      // bbox data in DSMEM so that we can still use bulk store/load, or hold
      // them in registers
      s_bboxes[i] = local_bbox;
    } else {
      s_indices[i] = -1;
    }
  }
  __syncthreads();

  int ref_item_idx = 0;
  int ref_bbox_idx = s_indices[ref_item_idx];
  while ((ref_bbox_idx != -1) && (ref_item_idx < max_read_num)) {
    Bbox<T_BBOX> ref_bbox = s_bboxes[ref_item_idx];

    for (int i = tidx; i < max_read_num; i += BS) {
      if (i > ref_item_idx && s_indices[i] != -1) {
        // TODO: consider use bulk_load in Lauda
        Bbox<T_BBOX> curr_bbox = s_bboxes[i];
        float iou =
            jaccardOverlap(ref_bbox, curr_bbox, isNormalized, T_BBOX(0));
        if (iou > iou_threshold) {
          s_scores[i] = 0.0f;
          s_indices[i] = -1;
        }
      }
    }
    __syncthreads();
    // find next reference bbox
    do {
      ref_item_idx++;
    } while ((ref_item_idx < max_read_num) &&
             ((ref_bbox_idx = s_indices[ref_item_idx]) == -1));
  }

  // TODO: Replace by TMA Store on Lauda
  for (int i = tidx; i < max_read_num; i += BS) {
    nmsed_scores[bidx * topk + i] = s_scores[i];
    nmsed_indices[bidx * topk + i] = s_indices[i];
  }
}

void per_class_nms(const nb::ndarray<> &nmsed_scores,
                   const nb::ndarray<> &nmsed_indices,
                   const nb::ndarray<> &bbox_delta, const nb::ndarray<> &rois,
                   const nb::ndarray<> &im_info, const nb::ndarray<> &scores,
                   const nb::ndarray<> &indices, int num_rois, int num_classes,
                   int topk, float iou_threshold) {
  k_per_class_nms<<<num_classes, BS>>>(
      reinterpret_cast<float *>(nmsed_scores.data()),
      reinterpret_cast<int *>(nmsed_indices.data()),
      reinterpret_cast<float *>(bbox_delta.data()),
      reinterpret_cast<float *>(rois.data()),
      reinterpret_cast<float *>(im_info.data()),
      reinterpret_cast<float *>(scores.data()),
      reinterpret_cast<int *>(indices.data()), num_rois, num_classes, topk,
      iou_threshold);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  CHECK_CUDA_ERROR(cudaGetLastError());
  return;
}