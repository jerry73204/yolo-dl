use crate::common::*;

#[cfg(debug_assertions)]
static SMALL_SCALING_WARN: Once = Once::new();

#[derive(Debug, Clone)]
pub struct DarkBatchNormConfig {
    pub cudnn_enabled: bool,
    pub eps: R64,
    pub momentum: R64,
    pub ws_init: Option<nn::Init>,
    pub bs_init: Option<nn::Init>,
    pub var_min: Option<R64>,
    pub var_max: Option<R64>,
}

#[derive(Debug)]
pub struct DarkBatchNorm {
    running_mean: Tensor,
    running_var: Tensor,
    ws: Option<Tensor>,
    bs: Option<Tensor>,
    nd: usize,
    cudnn_enabled: bool,
    eps: f64,
    momentum: f64,
    var_min: Option<f64>,
    var_max: Option<f64>,
}

impl Default for DarkBatchNormConfig {
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

impl DarkBatchNorm {
    pub fn new<'a>(
        path: impl Borrow<nn::Path<'a>>,
        nd: usize,
        out_dim: i64,
        config: DarkBatchNormConfig,
    ) -> Self {
        let path = path.borrow();
        let DarkBatchNormConfig {
            cudnn_enabled,
            eps,
            momentum,
            ws_init,
            bs_init,
            var_min,
            var_max,
        } = config;

        let ws = ws_init.map(|init| path.var("weight", &[out_dim], init));
        let bs = bs_init.map(|init| path.var("bias", &[out_dim], init));

        Self {
            running_mean: path.zeros_no_train("running_mean", &[out_dim]),
            running_var: path.ones_no_train("running_var", &[out_dim]),
            ws,
            bs,
            nd,
            cudnn_enabled,
            eps: eps.raw(),
            momentum: momentum.raw(),
            var_min: var_min.map(|min| min.raw()),
            var_max: var_max.map(|max| max.raw()),
        }
    }

    pub fn new_2d<'a>(
        path: impl Borrow<nn::Path<'a>>,
        out_dim: i64,
        config: DarkBatchNormConfig,
    ) -> Self {
        Self::new(path, 2, out_dim, config)
    }

    pub fn forward_t(&mut self, input: &Tensor, train: bool) -> Result<Tensor> {
        let Self {
            ref running_mean,
            ref mut running_var,
            ref ws,
            ref bs,
            nd,
            momentum,
            eps,
            cudnn_enabled,
            var_min,
            var_max,
        } = *self;

        ensure!(
            input.dim() == nd + 2,
            "expected an input tensor with {} dims, got {:?}",
            nd + 2,
            input.dim()
        );

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

        Ok(output)
    }
}
