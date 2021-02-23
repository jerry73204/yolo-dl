use crate::common::*;

#[derive(Debug, Clone)]
pub struct DarkBatchNormConfig {
    pub cudnn_enabled: bool,
    pub eps: f64,
    pub momentum: f64,
    pub affine_init: Option<AffineInit>,
}

#[derive(Debug, Clone)]
pub struct AffineInit {
    pub ws_init: nn::Init,
    pub bs_init: nn::Init,
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
            eps: 1e-4,
            momentum: 0.03,
            affine_init: Some(Default::default()),
        }
    }
}

impl Default for AffineInit {
    fn default() -> Self {
        Self {
            ws_init: nn::Init::Const(1.0),
            bs_init: nn::Init::Const(0.0),
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
            affine_init,
        } = config;

        let ws = affine_init
            .as_ref()
            .map(|init| path.var("weight", &[out_dim], init.ws_init));
        let bs = affine_init
            .as_ref()
            .map(|init| path.var("bias", &[out_dim], init.bs_init));

        Self {
            running_mean: path.zeros_no_train("running_mean", &[out_dim]),
            running_var: path.ones_no_train("running_var", &[out_dim]),
            ws,
            bs,
            nd,
            cudnn_enabled,
            eps,
            momentum,
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

        Ok(Tensor::batch_norm(
            input,
            ws.as_ref(),
            bs.as_ref(),
            Some(running_mean),
            Some(running_var),
            train,
            momentum,
            eps,
            cudnn_enabled,
        ))
    }
}
