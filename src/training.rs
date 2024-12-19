use burn::{prelude::*, train::ClassificationOutput};
use nn::loss::CrossEntropyLossConfig;

use crate::model::Model;

impl<B: Backend> Model<B> {
    pub fn forward_classifition(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());
        ClassificationOutput::new(loss, output, targets)
    }
}