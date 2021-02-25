use crate::common::*;

static SMALL_SCALING_WARN: Once = Once::new();

#[derive(Debug, Clone)]
pub struct DarkBatchNormConfig {
    pub cudnn_enabled: bool,
    pub eps: R64,
    pub momentum: R64,
    pub ws_init: Option<nn::Init>,
    pub bs_init: Option<nn::Init>,
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
}

impl Default for DarkBatchNormConfig {
    fn default() -> Self {
        Self {
            cudnn_enabled: true,
            eps: r64(1e-4),
            momentum: r64(0.03),
            ws_init: Some(nn::Init::Const(1.0)),
            bs_init: Some(nn::Init::Const(0.0)),
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
        }
    }

    pub fn new_2d<'a>(
        path: impl Borrow<nn::Path<'a>>,
        out_dim: i64,
        config: DarkBatchNormConfig,
    ) -> Self {
        Self::new(path, 2, out_dim, config)
    }

    pub fn forward_t(&self, input: &Tensor, train: bool) -> Result<Tensor> {
        let Self {
            ref running_mean,
            ref running_var,
            ref ws,
            ref bs,
            nd,
            momentum,
            eps,
            cudnn_enabled,
            ..
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

        assert!(
            ws.as_ref()
                .map(|ws| !bool::from(ws.le(1e-6).any()))
                .unwrap_or(false),
            "scaling factor {} is too small",
            f64::from(ws.as_ref().unwrap().min())
        );

        Ok(output)
    }
}
