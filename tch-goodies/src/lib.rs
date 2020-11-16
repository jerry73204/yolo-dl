use tch::Tensor;

pub trait TensorExt {
    fn swish(&self) -> Tensor;
    fn hard_swish(&self) -> Tensor;
    fn mish(&self) -> Tensor;
    fn hard_mish(&self) -> Tensor;
    // fn normalize_channels(&self) -> Tensor;
    // fn normalize_channels_softmax(&self) -> Tensor;
}

impl TensorExt for Tensor {
    fn swish(&self) -> Tensor {
        self * self.sigmoid()
    }

    fn hard_swish(&self) -> Tensor {
        self * (self + 3.0).clamp(0.0, 6.0) / 6.0
    }

    fn mish(&self) -> Tensor {
        self * &self.softplus().tanh()
    }

    fn hard_mish(&self) -> Tensor {
        let case1 = self.clamp(-2.0, 0.0);
        let case2 = self.clamp_min(0.0);
        (case1.pow(2.0) / 2.0 + &case1) + case2
    }

    // fn normalize_channels(&self) -> Tensor {
    //     todo!();
    // }

    // fn normalize_channels_softmax(&self) -> Tensor {
    //     todo!();
    // }
}
