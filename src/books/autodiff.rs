use burn::prelude::Backend;
use burn::tensor::{backend::AutodiffBackend, Tensor};
use burn::backend::Autodiff;

fn calculate_gradients<B: AutodiffBackend>(tensor: Tensor<B, 2>) -> B::Gradients {
    let mut gradients = tensor.clone().backward();
    let tensor_grad = tensor.grad(&gradients);
    let tensor_grad = tensor.grad_remove(&mut gradients);

    gradients
}

/// Use `B: AutodiffBackend`
fn example_validation<B: AutodiffBackend>(tensor: Tensor<B, 2>) {
    let inner_tensor: Tensor<B::InnerBackend, 2> = tensor.inner();
    let _ = inner_tensor + 5;
    todo!()
}

/// Use `B: Backend`
fn example_inference<B: Backend>(tensor: Tensor<B, 2>) {
    let _ = tensor + 5;
    todo!()
}