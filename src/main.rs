mod data;
mod inference;
mod model;
mod training;

#[cfg(test)]
mod books;

use crate::{model::ModelConfig, training::TrainingConfig};
use burn::{
    backend::{Autodiff, Wgpu},
    data::dataset::Dataset,
    optim::AdamConfig,
};

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "/tmp/guide";
    // crate::training::train::<MyAutodiffBackend>(
    //     artifact_dir,
    //     TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
    //     device.clone(),
    // );
    crate::inference::infer::<MyBackend>(
        artifact_dir,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(39)
            .unwrap(),
    );
}
