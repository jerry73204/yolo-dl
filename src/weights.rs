use crate::common::*;

pub trait WeightsInit
where
    Self: Sized,
{
    type Config;
    type InShape;
    type OutShape;

    fn init(
        layer_index: usize,
        config: &Self::Config,
        input_shape: Self::InShape,
        output_shape: Self::OutShape,
    ) -> Result<Self>;
}

impl WeightsInit for () {
    type Config = ();
    type InShape = ();
    type OutShape = ();

    fn init(
        _layer_index: usize,
        _config: &Self::Config,
        _input_shape: Self::InShape,
        _output_shape: Self::OutShape,
    ) -> Result<Self> {
        Ok(())
    }
}
