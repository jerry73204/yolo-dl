#include <ATen/ATen.h>

template <typename scalar_t>
at::Tensor nms_cpu_kernel(const at::Tensor &dets, const at::Tensor &scores,
                          const at::Tensor &groups,
                          const double iou_threshold) {
  TORCH_CHECK(!dets.is_cuda(), "dets must be a CPU tensor");
  TORCH_CHECK(!scores.is_cuda(), "scores must be a CPU tensor");
  TORCH_CHECK(!groups.is_cuda(), "scores must be a CPU tensor");
  TORCH_CHECK(dets.scalar_type() == scores.scalar_type(),
              "dets should have the same type as scores");
  TORCH_CHECK(groups.scalar_type() == at::kLong, "groups must be Long type");

  if (dets.numel() == 0)
    return at::empty({0}, dets.options().dtype(at::kLong));

  auto x1_t = dets.select(1, 0).contiguous();
  auto y1_t = dets.select(1, 1).contiguous();
  auto x2_t = dets.select(1, 2).contiguous();
  auto y2_t = dets.select(1, 3).contiguous();

  at::Tensor areas_t = (x2_t - x1_t) * (y2_t - y1_t);

  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

  auto ndets = dets.size(0);
  at::Tensor suppressed_t = at::zeros({ndets}, dets.options().dtype(at::kByte));
  at::Tensor keep_t = at::zeros({ndets}, dets.options().dtype(at::kLong));

  auto suppressed = suppressed_t.data_ptr<uint8_t>();
  auto keep = keep_t.data_ptr<int64_t>();
  auto order = order_t.data_ptr<int64_t>();
  auto x1 = x1_t.data_ptr<scalar_t>();
  auto y1 = y1_t.data_ptr<scalar_t>();
  auto x2 = x2_t.data_ptr<scalar_t>();
  auto y2 = y2_t.data_ptr<scalar_t>();
  auto areas = areas_t.data_ptr<scalar_t>();
  auto groups_ = groups.data_ptr<int64_t>();

  int64_t num_to_keep = 0;

  for (int64_t _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1)
      continue;
    keep[num_to_keep++] = i;
    auto ix1 = x1[i];
    auto iy1 = y1[i];
    auto ix2 = x2[i];
    auto iy2 = y2[i];
    auto iarea = areas[i];

    for (int64_t _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1)
        continue;
      if (groups_[i] != groups_[j])
        continue;

      auto xx1 = std::max(ix1, x1[j]);
      auto yy1 = std::max(iy1, y1[j]);
      auto xx2 = std::min(ix2, x2[j]);
      auto yy2 = std::min(iy2, y2[j]);

      auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1);
      auto h = std::max(static_cast<scalar_t>(0), yy2 - yy1);
      auto inter = w * h;
      auto ovr = inter / (iarea + areas[j] - inter);
      if (ovr > iou_threshold)
        suppressed[j] = 1;
    }
  }
  return keep_t.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep);
}

at::Tensor nms_cpu(const at::Tensor &dets, const at::Tensor &scores,
                   const at::Tensor &groups, const double iou_threshold) {
  TORCH_CHECK(dets.dim() == 2, "boxes should be a 2d tensor, got ", dets.dim(),
              "D");
  TORCH_CHECK(dets.size(1) == 4,
              "boxes should have 4 elements in dimension 1, got ",
              dets.size(1));
  TORCH_CHECK(scores.dim() == 1, "scores should be a 1d tensor, got ",
              scores.dim(), "D");
  TORCH_CHECK(groups.dim() == 1, "groups should be a 1d tensor, got ",
              groups.dim(), "D");
  TORCH_CHECK(dets.size(0) == scores.size(0),
              "boxes and scores should have same number of elements in ",
              "dimension 0, got ", dets.size(0), " and ", scores.size(0));
  TORCH_CHECK(dets.size(0) == groups.size(0),
              "boxes and groups should have same number of elements in ",
              "dimension 0, got ", dets.size(0), " and ", groups.size(0));

  auto result = at::empty({0}, dets.options());

  AT_DISPATCH_FLOATING_TYPES(dets.scalar_type(), "nms", [&] {
    result = nms_cpu_kernel<scalar_t>(dets, scores, groups, iou_threshold);
  });
  return result;
}

extern "C" {
void nms_cpu_ffi(at::Tensor **keep, at::Tensor *dets, at::Tensor *scores,
                 at::Tensor *groups, double iou_threshold) {
  at::Tensor keep_ = nms_cpu(*dets, *scores, *groups, iou_threshold);
  *keep = new at::Tensor(keep_);
}
}
