use crate::common::*;

#[derive(Debug, Clone)]
pub struct GroupNormInit {
    pub eps: R64,
    pub cudnn_enabled: bool,
    pub ws_init: Option<nn::Init>,
    pub bs_init: Option<nn::Init>,
}

impl GroupNormInit {
    pub fn build<'a>(
        self,
        path: impl Borrow<nn::Path<'a>>,
        out_dim: i64,
        num_groups: i64,
    ) -> GroupNorm {
        let path = path.borrow();
        let Self {
            eps,
            cudnn_enabled,
            ws_init,
            bs_init,
        } = self;

        let ws = ws_init.map(|init| path.var("weight", &[out_dim], init));
        let bs = bs_init.map(|init| path.var("bias", &[out_dim], init));

        GroupNorm {
            ws,
            bs,
            num_groups,
            cudnn_enabled,
            eps: eps.raw(),
        }
    }
}

impl Default for GroupNormInit {
    fn default() -> Self {
        Self {
            eps: r64(1e-5),
            cudnn_enabled: true,
            ws_init: Some(nn::Init::Const(1.0)),
            bs_init: Some(nn::Init::Const(0.0)),
        }
    }
}

#[derive(Debug)]
pub struct GroupNorm {
    ws: Option<Tensor>,
    bs: Option<Tensor>,
    cudnn_enabled: bool,
    eps: f64,
    num_groups: i64,
}

impl nn::ModuleT for GroupNorm {
    fn forward_t(&self, input: &Tensor, train: bool) -> Tensor {
        let Self {
            ref ws,
            ref bs,
            eps,
            cudnn_enabled,
            num_groups,
            ..
        } = *self;

        let output = Tensor::group_norm(
            input,
            num_groups,
            ws.as_ref(),
            bs.as_ref(),
            eps,
            cudnn_enabled,
        );

        output
    }
}

impl GroupNorm {
    pub fn grad(&self) -> GroupNormGrad {
        let Self { ws, bs, .. } = self;

        GroupNormGrad {
            ws: ws.as_ref().map(Tensor::grad),
            bs: bs.as_ref().map(Tensor::grad),
        }
    }
}

#[derive(Debug, TensorLike)]
pub struct GroupNormGrad {
    pub ws: Option<Tensor>,
    pub bs: Option<Tensor>,
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use tch::kind::FLOAT_CPU;

//     #[test]
//     fn instance_norm_nan_test() {
//         const CHANNELS: i64 = 16;

//         let vs = nn::VarStore::new(Device::Cpu);
//         let root = vs.root();

//         let norm = GroupNormInit {
//             var_min: Some(1e-3),
//             ..Default::default()
//         }
//         .build(&root, CHANNELS);

//         for _ in 0..100 {
//             norm.clamp_running_var();
//             let input = Tensor::zeros(&[8, CHANNELS, 1, 1], FLOAT_CPU);
//             let output = norm.forward_t(&input, true);
//             assert!(!norm.has_nan());
//         }
//     }
// }
