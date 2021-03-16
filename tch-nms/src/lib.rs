use libc::{c_char, c_void};
use static_assertions::const_assert_eq;
use std::{ffi::CStr, mem, ptr};
use tch::{Device, TchError, Tensor};

const_assert_eq!(mem::size_of::<Tensor>(), mem::size_of::<*mut c_void>());

macro_rules! unsafe_torch_err {
    ($e:expr) => {{
        let v = unsafe { $e };
        crate::read_and_clean_error()?;
        v
    }};
}

#[link(name = "nms_cpu", kind = "static")]
extern "C" {
    /// The raw FFI interface to the CUDA implementation.
    pub fn nms_cpu_ffi(
        keep: *mut *mut c_void,
        dets: *mut c_void,
        scores: *mut c_void,
        groups: *mut c_void,
        iou_threshold: f64,
    );

    pub(crate) fn get_and_reset_last_err() -> *mut c_char;
}

#[link(name = "nms_cuda", kind = "static")]
extern "C" {
    /// The raw FFI interface to the CUDA implementation.
    pub fn nms_cuda_ffi(
        keep: *mut *mut c_void,
        dets: *mut c_void,
        scores: *mut c_void,
        groups: *mut c_void,
        iou_threshold: f64,
    );
}

/// Run Non-Maximum Suppression algorithm on boxes with corresponding scores.
///
/// - **dets** - Nx4 shaped float tensor in left-top-right-bottom format.
/// - **scores** - N shaped float tensor of scores for each box.
/// - **iou_threshold** - The IoU threshold value if one box is considered overlapped with the other.
pub fn nms_by_scores(
    dets: &Tensor,
    scores: &Tensor,
    groups: &Tensor,
    iou_threshold: f64,
) -> Result<Tensor, TchError> {
    match (dets.device(), scores.device(), groups.device()) {
        (Device::Cpu, Device::Cpu, Device::Cpu) => {
            nms_by_scores_cpu(dets, scores, groups, iou_threshold)
        }
        (Device::Cuda(_), Device::Cuda(_), Device::Cuda(_)) => {
            nms_by_scores_cuda(dets, scores, groups, iou_threshold)
        }
        _ => Err(TchError::Torch(
            "dets, scores and groups tensors must be on identical device".to_string(),
        )),
    }
}

/// Run Non-Maximum Suppression algorithm on CUDA device.
///
/// Input tensors must be on CUDA device. Otherwise the function will panic.
pub fn nms_by_scores_cuda(
    dets: &Tensor,
    scores: &Tensor,
    groups: &Tensor,
    iou_threshold: f64,
) -> Result<Tensor, TchError> {
    // workaround to get the internal pointers
    let dets: *mut c_void = unsafe { mem::transmute(dets.shallow_clone()) };
    let scores: *mut c_void = unsafe { mem::transmute(scores.shallow_clone()) };
    let groups: *mut c_void = unsafe { mem::transmute(groups.shallow_clone()) };

    // create uninitialized output tensors
    let mut keep: *mut c_void = ptr::null_mut();

    unsafe_torch_err!(nms_cuda_ffi(
        &mut keep as *mut _,
        dets,
        scores,
        groups,
        iou_threshold,
    ));

    unsafe {
        let keep: Tensor = mem::transmute(keep);
        Ok(keep)
    }
}

/// Run Non-Maximum Suppression algorithm on CPU.
///
/// Input tensors must be on CPU. Otherwise the function will panic.
pub fn nms_by_scores_cpu(
    dets: &Tensor,
    scores: &Tensor,
    groups: &Tensor,
    iou_threshold: f64,
) -> Result<Tensor, TchError> {
    // workaround to get the internal pointers
    let dets: *mut c_void = unsafe { mem::transmute(dets.shallow_clone()) };
    let scores: *mut c_void = unsafe { mem::transmute(scores.shallow_clone()) };
    let groups: *mut c_void = unsafe { mem::transmute(groups.shallow_clone()) };

    // create uninitialized output tensors
    let mut keep: *mut c_void = ptr::null_mut();

    unsafe_torch_err!(nms_cpu_ffi(
        &mut keep as *mut _,
        dets,
        scores,
        groups,
        iou_threshold,
    ));

    unsafe {
        let keep: Tensor = mem::transmute(keep);
        Ok(keep)
    }
}

pub(crate) fn read_and_clean_error() -> Result<(), TchError> {
    unsafe {
        match ptr_to_string(crate::get_and_reset_last_err()) {
            None => Ok(()),
            Some(c_error) => Err(TchError::Torch(c_error)),
        }
    }
}

pub(crate) unsafe fn ptr_to_string(ptr: *mut c_char) -> Option<String> {
    if !ptr.is_null() {
        let text = CStr::from_ptr(ptr).to_string_lossy().into_owned();
        libc::free(ptr as *mut c_void);
        Some(text)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind};

    fn run_nms_test(device: Device) {
        const N_BOXES: i64 = 10000;

        let cy = Tensor::rand(&[N_BOXES, 1], (Kind::Float, device));
        let cx = Tensor::rand(&[N_BOXES, 1], (Kind::Float, device));
        let h = Tensor::rand(&[N_BOXES, 1], (Kind::Float, device));
        let w = Tensor::rand(&[N_BOXES, 1], (Kind::Float, device));

        let t = &cy - &h / 2.0;
        let b = &cy + &h / 2.0;
        let l = &cx - &w / 2.0;
        let r = &cx + &w / 2.0;

        let boxes = Tensor::cat(&[l, t, r, b], 1);
        let scores = Tensor::rand(&[N_BOXES], (Kind::Float, device));
        let groups = Tensor::zeros(&[N_BOXES], (Kind::Int64, device));

        let keep = nms_by_scores(&boxes, &scores, &groups, 0.50).unwrap();
        assert!(keep.size1().unwrap() <= N_BOXES);

        let keep = Vec::<i64>::from(&keep);
        assert!(keep.iter().all(|&val| val < N_BOXES));
    }

    #[test]
    fn nms_cpu_test() {
        run_nms_test(Device::Cpu);
    }

    #[test]
    fn nms_cuda_test() {
        match Device::cuda_if_available() {
            dev @ Device::Cuda(_) => {
                run_nms_test(dev);
            }
            _ => (),
        };
    }
}
