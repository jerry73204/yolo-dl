use libc::{c_char, c_void};
use static_assertions::const_assert_eq;
use std::{ffi::CStr, mem, ptr};
use tch::{TchError, Tensor};

const_assert_eq!(mem::size_of::<Tensor>(), mem::size_of::<*mut c_void>());

macro_rules! unsafe_torch_err {
    ($e:expr) => {{
        let v = unsafe { $e };
        crate::read_and_clean_error()?;
        v
    }};
}

#[link(name = "nms_kernel", kind = "static")]
extern "C" {
    pub fn nms_cuda_forward_ffi(
        keep: *mut *mut c_void,
        num_to_keep: *mut *mut c_void,
        parent_object_index: *mut *mut c_void,
        boxes: *mut c_void,
        idx: *mut c_void,
        nms_overlap_thresh: f64,
        top_k: u64,
    );

    pub fn get_and_reset_last_err() -> *mut c_char;
}

pub fn nms_cuda_forward(
    boxes: &Tensor,
    idx: &Tensor,
    nms_overlap_thresh: f64,
    top_k: u64,
) -> Result<(Tensor, Tensor, Tensor), TchError> {
    // workaround to get the internal pointers
    let boxes: *mut c_void = unsafe { mem::transmute(boxes.shallow_clone()) };
    let idx: *mut c_void = unsafe { mem::transmute(idx.shallow_clone()) };

    // create uninitialized output tensors
    let mut keep: *mut c_void = ptr::null_mut();
    let mut num_to_keep: *mut c_void = ptr::null_mut();
    let mut parent_object_index: *mut c_void = ptr::null_mut();

    unsafe_torch_err!(nms_cuda_forward_ffi(
        &mut keep as *mut _,
        &mut num_to_keep as *mut _,
        &mut parent_object_index as *mut _,
        boxes,
        idx,
        nms_overlap_thresh,
        top_k
    ));

    unsafe {
        let keep: Tensor = mem::transmute(keep);
        let num_to_keep: Tensor = mem::transmute(num_to_keep);
        let parent_object_index: Tensor = mem::transmute(parent_object_index);
        Ok((keep, num_to_keep, parent_object_index))
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

    #[test]
    fn nms_ffi_test() {
        const N_BOXES: i64 = 1000;
        let device = Device::Cuda(0);

        for _ in 0..10 {
            let cy = Tensor::rand(&[N_BOXES, 1], (Kind::Float, device));
            let cx = Tensor::rand(&[N_BOXES, 1], (Kind::Float, device));
            let h = Tensor::rand(&[N_BOXES, 1], (Kind::Float, device));
            let w = Tensor::rand(&[N_BOXES, 1], (Kind::Float, device));

            let t = &cy - &h / 2.0;
            let b = &cy + &h / 2.0;
            let l = &cx - &w / 2.0;
            let r = &cx + &w / 2.0;

            let boxes = Tensor::cat(&[l, t, r, b], 1);
            let idx = Tensor::full(&[N_BOXES], 0, (Kind::Float, device));

            let (_keep, _num_to_keep, _parent_object_index) =
                nms_cuda_forward(&boxes, &idx, 0.98, 1).unwrap();

            // dbg!(&keep, &num_to_keep, &parent_object_index);
        }
    }
}
