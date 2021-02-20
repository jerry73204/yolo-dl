use libc::{c_char, c_void};
use std::{
    ffi::CStr,
    mem::{self, MaybeUninit},
};
use tch::{TchError, Tensor};

macro_rules! unsafe_torch_err {
    ($e:expr) => {{
        let v = unsafe { $e };
        crate::read_and_clean_error()?;
        v
    }};
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
    let mut keep: MaybeUninit<*mut c_void> = MaybeUninit::uninit();
    let mut num_to_keep: MaybeUninit<*mut c_void> = MaybeUninit::uninit();
    let mut parent_object_index: MaybeUninit<*mut c_void> = MaybeUninit::uninit();

    unsafe_torch_err!(nms_cuda_forward_ffi(
        keep.as_mut_ptr(),
        num_to_keep.as_mut_ptr(),
        parent_object_index.as_mut_ptr(),
        boxes,
        idx,
        nms_overlap_thresh,
        top_k
    ));

    unsafe {
        let keep: Tensor = mem::transmute(keep.assume_init());
        let num_to_keep: Tensor = mem::transmute(num_to_keep.assume_init());
        let parent_object_index: Tensor = mem::transmute(parent_object_index.assume_init());
        Ok((keep, num_to_keep, parent_object_index))
    }
}
