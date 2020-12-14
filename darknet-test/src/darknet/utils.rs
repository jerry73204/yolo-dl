use crate::common::*;

#[cfg(unix)]
pub fn path_to_cstring<'a>(path: &'a Path) -> Option<CString> {
    use std::os::unix::ffi::OsStrExt;
    Some(CString::new(path.as_os_str().as_bytes()).unwrap())
}

#[cfg(not(unix))]
pub fn path_to_cstring<'a>(path: &'a Path) -> Option<CString> {
    path.to_str().map(|s| CString::new(s.as_bytes()).unwrap())
}
