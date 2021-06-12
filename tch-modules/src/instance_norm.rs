use crate::common::*;

#[derive(Debug, Clone)]
pub struct InstanceNormInit {
    pub eps: R64,
    pub momentum: R64,
    pub cudnn_enabled: bool,
    pub track_running_stats: bool,
    pub ws_init: Option<nn::Init>,
    pub bs_init: Option<nn::Init>,
    pub var_min: Option<f64>,
    pub var_max: Option<f64>,
}

impl InstanceNormInit {
    pub fn build<'a>(self, path: impl Borrow<nn::Path<'a>>, out_dim: i64) -> InstanceNorm {
        let path = path.borrow();
        let Self {
            eps,
            momentum,
            track_running_stats,
            cudnn_enabled,
            ws_init,
            bs_init,
            var_min,
            var_max,
        } = self;

        let ws = ws_init.map(|init| path.var("weight", &[out_dim], init));
        let bs = bs_init.map(|init| path.var("bias", &[out_dim], init));

        InstanceNorm {
            running_mean: path.zeros_no_train("running_mean", &[out_dim]),
            running_var: path.ones_no_train("running_var", &[out_dim]),
            ws,
            bs,
            cudnn_enabled,
            track_running_stats,
            eps: eps.raw(),
            momentum: momentum.raw(),
            var_min,
            var_max,
        }
    }
}

impl Default for InstanceNormInit {
    fn default() -> Self {
        Self {
            eps: r64(1e-5),
            momentum: r64(0.1),
            track_running_stats: false,
            cudnn_enabled: true,
            ws_init: None,
            bs_init: None,
            var_min: None,
            var_max: None,
        }
    }
}

#[derive(Debug)]
pub struct InstanceNorm {
    running_mean: Tensor,
    running_var: Tensor,
    ws: Option<Tensor>,
    bs: Option<Tensor>,
    cudnn_enabled: bool,
    track_running_stats: bool,
    eps: f64,
    momentum: f64,
    var_min: Option<f64>,
    var_max: Option<f64>,
}

impl InstanceNorm {
    pub fn forward_t(&self, input: &Tensor, train: bool) -> Result<Tensor> {
        let Self {
            ref running_mean,
            ref running_var,
            ref ws,
            ref bs,
            momentum,
            eps,
            cudnn_enabled,
            track_running_stats,
            ..
        } = *self;

        let output = Tensor::instance_norm(
            input,
            ws.as_ref(),
            bs.as_ref(),
            Some(running_mean),
            Some(running_var),
            !track_running_stats || train,
            momentum,
            eps,
            cudnn_enabled,
        );

        #[cfg(debug_assertions)]
        {
            static SMALL_SCALING_WARN: Once = Once::new();
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

        Ok(output)
    }

    pub fn clamp_bn_var(&mut self) {
        tch::no_grad(|| {
            let Self {
                ref mut running_var,
                var_min,
                var_max,
                ..
            } = *self;

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

    pub fn denormalize_bn(&mut self) {
        tch::no_grad(|| {
            let Self {
                ws, running_var, ..
            } = self;

            if let Some(ws) = ws {
                ws.copy_(&(&*ws / &*running_var));
                let _ = running_var.fill_(1.0);
            }
        });
    }

    pub fn grad(&self) -> InstanceNormGrad {
        let Self { ws, bs, .. } = self;

        InstanceNormGrad {
            ws: ws.as_ref().map(Tensor::grad),
            bs: bs.as_ref().map(Tensor::grad),
        }
    }
}

#[derive(Debug, TensorLike)]
pub struct InstanceNormGrad {
    pub ws: Option<Tensor>,
    pub bs: Option<Tensor>,
}
