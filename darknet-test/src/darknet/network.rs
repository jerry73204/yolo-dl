use super::{detections::Detections, error::Error, image::IntoCowImage, layer::Layer, utils};
use crate::{common::*, sys};

/// The network wrapper type for Darknet.
#[derive(Debug)]
pub struct Network {
    net: NonNull<sys::network>,
}

impl Network {
    /// Build the network instance from a configuration file and an optional weights file.
    pub fn load<C, W>(cfg: C, weights: Option<W>, clear: bool) -> Result<Network, Error>
    where
        C: AsRef<Path>,
        W: AsRef<Path>,
    {
        // convert paths to CString
        let weights_cstr = weights
            .map(|path| {
                utils::path_to_cstring(path.as_ref()).ok_or_else(|| Error::EncodingError {
                    reason: format!("the path {} is invalid", path.as_ref().display()),
                })
            })
            .transpose()?;
        let cfg_cstr =
            utils::path_to_cstring(cfg.as_ref()).ok_or_else(|| Error::EncodingError {
                reason: format!("the path {} is invalid", cfg.as_ref().display()),
            })?;

        let ptr = unsafe {
            let raw_weights = weights_cstr
                .as_ref()
                .map(|cstr| cstr.as_ptr() as *mut _)
                .unwrap_or(ptr::null_mut());
            sys::load_network(cfg_cstr.as_ptr() as *mut _, raw_weights, clear as c_int)
        };

        let net = NonNull::new(ptr).ok_or_else(|| Error::InternalError {
            reason: "failed to load model".into(),
        })?;

        // drop paths here to avoid early deallocation
        mem::drop(cfg_cstr);
        mem::drop(weights_cstr);

        Ok(Self { net })
    }

    /// Get network input width.
    pub fn input_width(&self) -> usize {
        unsafe { self.net.as_ref().w as usize }
    }

    /// Get network input height.
    pub fn input_height(&self) -> usize {
        unsafe { self.net.as_ref().h as usize }
    }

    /// Get network input channels.
    pub fn input_channels(&self) -> usize {
        unsafe { self.net.as_ref().c as usize }
    }

    /// Get network input shape tuple (channels, height, width).
    pub fn input_shape(&self) -> (usize, usize, usize) {
        (
            self.input_channels(),
            self.input_height(),
            self.input_width(),
        )
    }

    /// Get the number of layers.
    pub fn num_layers(&self) -> usize {
        unsafe { self.net.as_ref().n as usize }
    }

    /// Get network layers.
    pub fn layers(&self) -> &[Layer] {
        let layers = unsafe {
            slice::from_raw_parts(self.net.as_ref().layers as *const Layer, self.num_layers())
        };
        layers
    }

    /// Run inference on an image.
    pub fn predict<'a, M>(
        &mut self,
        image: M,
        thresh: f32,
        hier_thres: f32,
        nms: f32,
        use_letter_box: bool,
    ) -> Detections
    where
        M: IntoCowImage<'a>,
    {
        let cow = image.into_cow_image();

        unsafe {
            let output_layer = self
                .net
                .as_ref()
                .layers
                .add(self.num_layers() - 1)
                .as_ref()
                .unwrap();

            // run prediction
            if use_letter_box {
                sys::network_predict_image_letterbox(self.net.as_ptr(), cow.image);
            } else {
                sys::network_predict_image(self.net.as_ptr(), cow.image);
            }

            let mut nboxes: c_int = 0;
            let dets_ptr = sys::get_network_boxes(
                self.net.as_mut(),
                cow.width() as c_int,
                cow.height() as c_int,
                thresh,
                hier_thres,
                ptr::null_mut(),
                1,
                &mut nboxes,
                use_letter_box as c_int,
            );
            let dets = NonNull::new(dets_ptr).unwrap();

            // NMS sort
            if nms != 0.0 {
                if output_layer.nms_kind == sys::NMS_KIND_DEFAULT_NMS {
                    sys::do_nms_sort(dets.as_ptr(), nboxes, output_layer.classes, nms);
                } else {
                    sys::diounms_sort(
                        dets.as_ptr(),
                        nboxes,
                        output_layer.classes,
                        nms,
                        output_layer.nms_kind,
                        output_layer.beta_nms,
                    );
                }
            }

            Detections {
                detections: dets,
                n_detections: nboxes as usize,
            }
        }
    }

    // pub fn into_state(self, train: bool) -> NetworkState {
    //     NetworkState::new(self, train)
    // }

    pub fn into_raw(self) -> *mut sys::network {
        let net_ptr = self.net.as_ptr();
        mem::forget(self);
        net_ptr as *mut _
    }

    pub unsafe fn from_raw(net_ptr: *mut sys::network) -> Self {
        Self {
            net: NonNull::new(net_ptr).unwrap(),
        }
    }
}

impl Drop for Network {
    fn drop(&mut self) {
        unsafe {
            let ptr = self.net.as_ptr();
            sys::free_network(*ptr);

            // The network* pointer was allocated by calloc
            // We have to deallocate it manually
            libc::free(ptr as *mut c_void);
        }
    }
}

unsafe impl Send for Network {}

// pub struct NetworkState {
//     net: *mut sys::network,
//     state: sys::network_state,
// }

// impl NetworkState {
//     pub fn new(network: Network, train: bool) -> Self {
//         unsafe {
//             let net = network.into_raw();

//             Self {
//                 net,
//                 state: sys::network_state {
//                     net: *net,
//                     input: ptr::null_mut(),
//                     truth: ptr::null_mut(),
//                     delta: ptr::null_mut(),
//                     workspace: ptr::null_mut(),
//                     train: if train { 1 } else { 0 },
//                     index: 0,
//                 },
//             }
//         }
//     }

//     pub fn into_forward_iterator<'a>(self, input: impl Into<Cow<'a, [f32]>>) -> Forward {
//         unsafe {
//             let (input, input_len_cap) = {
//                 let (input, len, cap) = input.into().into_owned().into_raw_parts();
//                 (input, (len, cap))
//             };
//             let (net, mut state) = self.into_raw_parts();
//             state.workspace = (*net).workspace;
//             state.input = input;
//             Forward {
//                 net,
//                 state,
//                 input_len_cap,
//             }
//         }
//     }

//     pub fn predict<'a>(&self, input: impl Into<Cow<'a, [f32]>>) -> Result<()> {
//         unsafe {
//             let (input, input_len_cap) = {
//                 let (input, len, cap) = input.into().into_owned().into_raw_parts();
//                 (input, (len, cap))
//             };
//             todo!();
//             Ok(())
//         }
//     }

//     pub fn train(&self, input: &[f32], truth: Option<&[f32]>) -> Result<()> {
//         Ok(())
//     }

//     pub fn into_network(self) -> Network {
//         Network {
//             net: NonNull::new(self.net).unwrap(),
//         }
//     }

//     pub fn into_raw_parts(self) -> (*mut sys::network, sys::network_state) {
//         let net = self.net;
//         let state = self.state;
//         mem::forget(self);
//         (net, state)
//     }
// }

// impl Drop for NetworkState {
//     fn drop(&mut self) {
//         unsafe {
//             // free network
//             let net = self.net;
//             sys::free_network(*net);

//             // The network* pointer was allocated by calloc
//             // We have to deallocate it manually
//             libc::free(net as *mut c_void);
//         }
//     }
// }

// unsafe impl Send for NetworkState {}

// forward

// #[derive(Debug)]
// pub struct Forward {
//     input_len_cap: (usize, usize),
//     net: *mut sys::network,
//     state: sys::network_state,
// }

// impl Iterator for Forward {
//     type Item = ();

//     fn next(&mut self) -> Option<Self::Item> {
//         unsafe {
//             let net = self.net.as_ref().unwrap();

//             if self.state.index < net.n {
//                 let layer_ptr = net.layers.add(self.state.index as usize);
//                 let layer = LayerRef::new(layer_ptr.as_ref().unwrap());

//                 if !layer.delta.is_null() && self.state.train != 0 {
//                     todo!();
//                 }

//                 (layer.forward.unwrap())(*layer, self.state);

//                 self.state.input = layer.output;
//                 self.state.index += 1;

//                 todo!();
//             } else {
//                 None
//             }
//         }
//     }
// }

// impl Drop for Forward {
//     fn drop(&mut self) {
//         unsafe {
//             // free input
//             if self.state.input != ptr::null_mut() {
//                 let (len, cap) = self.input_len_cap;
//                 Vec::from_raw_parts(self.state.input, len, cap);
//             }

//             // free network
//             let net = self.net;
//             sys::free_network(*net);

//             // The network* pointer was allocated by calloc
//             // We have to deallocate it manually
//             libc::free(net as *mut c_void);
//         }
//     }
// }
