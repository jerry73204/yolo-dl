use crate::common::*;
use tch_goodies::TensorExt as _;

#[cfg(debug_assertions)]
static SMALL_SCALING_WARN: Once = Once::new();

#[derive(Debug, Clone)]
pub struct DarkBatchNormInit {
    pub cudnn_enabled: bool,
    pub eps: R64,
    pub momentum: R64,
    pub ws_init: Option<nn::Init>,
    pub bs_init: Option<nn::Init>,
    pub var_min: Option<f64>,
    pub var_max: Option<f64>,
}

#[derive(Debug)]
pub struct DarkBatchNorm {
    running_mean: Tensor,
    running_var: Tensor,
    ws: Option<Tensor>,
    bs: Option<Tensor>,
    cudnn_enabled: bool,
    eps: f64,
    momentum: f64,
    var_min: Option<f64>,
    var_max: Option<f64>,
}

impl Default for DarkBatchNormInit {
    fn default() -> Self {
        Self {
            cudnn_enabled: true,
            eps: r64(1e-4),
            momentum: r64(0.03),
            ws_init: Some(nn::Init::Const(1.0)),
            bs_init: Some(nn::Init::Const(0.0)),
            var_min: None,
            var_max: None,
        }
    }
}

impl DarkBatchNormInit {
    pub fn build<'a>(self, path: impl Borrow<nn::Path<'a>>, out_dim: i64) -> DarkBatchNorm {
        let path = path.borrow();
        let Self {
            cudnn_enabled,
            eps,
            momentum,
            ws_init,
            bs_init,
            var_min,
            var_max,
        } = self;

        let ws = ws_init.map(|init| path.var("weight", &[out_dim], init));
        let bs = bs_init.map(|init| path.var("bias", &[out_dim], init));

        DarkBatchNorm {
            running_mean: path.zeros_no_train("running_mean", &[out_dim]),
            running_var: path.ones_no_train("running_var", &[out_dim]),
            ws,
            bs,
            cudnn_enabled,
            eps: eps.raw(),
            momentum: momentum.raw(),
            var_min,
            var_max,
        }
    }
}

impl nn::ModuleT for DarkBatchNorm {
    fn forward_t(&self, input: &Tensor, train: bool) -> Tensor {
        let Self {
            ref running_mean,
            ref running_var,
            ref ws,
            ref bs,
            momentum,
            eps,
            cudnn_enabled,
            ..
        } = *self;

        let output = Tensor::batch_norm(
            input,
            ws.as_ref(),
            bs.as_ref(),
            Some(running_mean),
            Some(running_var),
            train,
            momentum,
            eps,
            cudnn_enabled,
        );

        #[cfg(debug_assertions)]
        {
            let has_small_var = bool::from(running_var.abs().le(1e-15).any());

            let has_small_ws = ws
                .as_ref()
                .map(|ws| bool::from(ws.abs().le(1e-15).any()))
                .unwrap_or(false);

            if has_small_var {
                SMALL_SCALING_WARN.call_once(|| {
                    warn!(
                        "runing variance {} is too small",
                        f64::from(running_var.abs().min())
                    );
                });
            }

            if has_small_ws {
                SMALL_SCALING_WARN.call_once(|| {
                    warn!(
                        "scaling factor {} is too small",
                        f64::from(ws.as_ref().unwrap().abs().min())
                    );
                });
            }
        }

        output
    }
}

impl DarkBatchNorm {
    pub fn has_nan(&self) -> bool {
        let Self {
            ws,
            bs,
            running_mean,
            running_var,
            ..
        } = self;

        ws.as_ref().map(|ws| ws.has_nan()).unwrap_or(false)
            || bs.as_ref().map(|bs| bs.has_nan()).unwrap_or(false)
            || running_mean.has_nan()
            || running_var.has_nan()
    }

    pub fn clamp_running_var(&self) {
        tch::no_grad(|| {
            let Self {
                ref running_var,
                var_min,
                var_max,
                ..
            } = *self;
            let mut running_var = running_var.shallow_clone();

            // clip running_var
            match (var_min, var_max) {
                (Some(min), Some(max)) => {
                    let _ = running_var.clamp_(min, max);
                }
                (None, Some(max)) => {
                    let _ = running_var.clamp_max_(max);
                }
                (Some(min), None) => {
                    let _ = running_var.clamp_min_(min);
                }
                (None, None) => {}
            }
        });
    }

    pub fn denormalize(&self) {
        tch::no_grad(|| {
            let Self {
                ws, running_var, ..
            } = self;
            let ws = ws.shallow_clone();
            let mut running_var = running_var.shallow_clone();

            if let Some(mut ws) = ws {
                ws.copy_(&(&ws / &running_var));
                let _ = running_var.fill_(1.0);
            }
        });
    }

    pub fn grad(&self) -> DarkBatchNormGrad {
        let Self { ws, bs, .. } = self;

        DarkBatchNormGrad {
            ws: ws.as_ref().map(Tensor::grad),
            bs: bs.as_ref().map(Tensor::grad),
        }
    }
}

#[derive(Debug, TensorLike)]
pub struct DarkBatchNormGrad {
    pub ws: Option<Tensor>,
    pub bs: Option<Tensor>,
}

#[cfg(test)]
mod tests {
    // use super::*;
    // use tch::kind::FLOAT_CPU;

    // #[test]
    // fn batch_norm_clip_running_var() {
    //     let vs = nn::VarStore::new(Device::Cpu);
    //     let root = vs.root();

    //     let bn_init = DarkBatchNormInit {
    //         var_min: Some(1e-3),
    //         var_max: Some(1e3),
    //         ..Default::default()
    //     };

    //     let seq = nn::seq_t()
    //         .add(nn::conv2d(
    //             &root / "conv1",
    //             3,
    //             8,
    //             3,
    //             nn::ConvConfig {
    //                 padding: 1,
    //                 stride: 2,
    //                 ..Default::default()
    //             },
    //         ))
    //         .add_fn(|xs| xs.leaky_relu())
    //         .add(bn_init.clone().build(&root / "bn1", 8))
    //         .add_fn(|xs| xs.leaky_relu())
    //         .add(nn::conv2d(
    //             &root / "conv2",
    //             8,
    //             8,
    //             3,
    //             nn::ConvConfig {
    //                 padding: 1,
    //                 stride: 2,
    //                 ..Default::default()
    //             },
    //         ))
    //         .add(bn_init.clone().build(&root / "bn2", 8))
    //         .add(nn::conv2d(
    //             &root / "conv3",
    //             8,
    //             1,
    //             3,
    //             nn::ConvConfig {
    //                 padding: 1,
    //                 stride: 2,
    //                 ..Default::default()
    //             },
    //         ))
    //         .add(bn_init.build(&root / "bn3", 1))
    //         .add_fn(|xs| {
    //             let bs = xs.size()[0];
    //             xs.view([bs])
    //         });

    //     const BATCH_SIZE: i64 = 7;
    //     let mut opt = nn::adam(0.5, 0.999, 0.).build(&vs, 1e-3).unwrap();

    //     for _ in 0..100 {
    //         let input = Tensor::rand(&[BATCH_SIZE, 3, 8, 8], FLOAT_CPU);
    //         let output = seq.forward_t(&input, true);
    //         let target = Tensor::zeros(&[BATCH_SIZE], FLOAT_CPU);
    //         let loss = output.binary_cross_entropy_with_logits::<Tensor>(
    //             &target,
    //             None,
    //             None,
    //             Reduction::Mean,
    //         );
    //         opt.backward_step(&loss);
    //     }
    // }

    // #[test]
    // fn batch_norm_nan_test() {
    //     const CHANNELS: i64 = 16;

    //     let vs = nn::VarStore::new(Device::Cpu);
    //     let root = vs.root();

    //     let norm = DarkBatchNormInit {
    //         var_min: Some(1e-3),
    //         ..Default::default()
    //     }
    //     .build(&root, CHANNELS);

    //     for _ in 0..100 {
    //         norm.clamp_running_var();
    //         let input = Tensor::zeros(&[8, CHANNELS, 1, 1], FLOAT_CPU);
    //         let output = norm.forward_t(&input, true);
    //         assert!(!norm.has_nan());
    //     }
    // }
}
