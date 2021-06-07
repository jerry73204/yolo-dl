use crate::common::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, TensorLike)]
pub struct InstanceIndex {
    pub batch_index: i64,
    pub layer_index: i64,
    pub anchor_index: i64,
    pub grid_row: i64,
    pub grid_col: i64,
}

#[derive(Debug, TensorLike)]
pub struct InstanceIndexTensor {
    pub batches: Tensor,
    pub layers: Tensor,
    pub anchors: Tensor,
    pub grid_rows: Tensor,
    pub grid_cols: Tensor,
}

impl FromIterator<InstanceIndex> for InstanceIndexTensor {
    fn from_iter<T: IntoIterator<Item = InstanceIndex>>(iter: T) -> Self {
        let (batches, layers, anchors, grid_rows, grid_cols) = iter
            .into_iter()
            .map(|index| {
                let InstanceIndex {
                    batch_index,
                    layer_index,
                    anchor_index,
                    grid_row,
                    grid_col,
                } = index;
                (batch_index, layer_index, anchor_index, grid_row, grid_col)
            })
            .unzip_n_vec();
        Self {
            batches: Tensor::of_slice(&batches),
            layers: Tensor::of_slice(&layers),
            anchors: Tensor::of_slice(&anchors),
            grid_rows: Tensor::of_slice(&grid_rows),
            grid_cols: Tensor::of_slice(&grid_cols),
        }
    }
}

impl<'a> FromIterator<&'a InstanceIndex> for InstanceIndexTensor {
    fn from_iter<T: IntoIterator<Item = &'a InstanceIndex>>(iter: T) -> Self {
        let (batches, layers, anchors, grid_rows, grid_cols) = iter
            .into_iter()
            .map(|index| {
                let InstanceIndex {
                    batch_index,
                    layer_index,
                    anchor_index,
                    grid_row,
                    grid_col,
                } = *index;
                (batch_index, layer_index, anchor_index, grid_row, grid_col)
            })
            .unzip_n_vec();
        Self {
            batches: Tensor::of_slice(&batches),
            layers: Tensor::of_slice(&layers),
            anchors: Tensor::of_slice(&anchors),
            grid_rows: Tensor::of_slice(&grid_rows),
            grid_cols: Tensor::of_slice(&grid_cols),
        }
    }
}
