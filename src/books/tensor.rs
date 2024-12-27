use burn::{backend::Wgpu, tensor::{check_closeness, Int, Tensor, TensorData}};

type Backend = Wgpu;

#[test]
fn test_tensor() {
    let device = Default::default();
    let tensor_1 = Tensor::<Wgpu, 1>::from_data([1., 2., 3.], &device);
    let tensor_2 = Tensor::<Backend, 1>::from_data(TensorData::from([1., 2., 3.]), &device);
    let tensor_3 = Tensor::<Backend, 1>::from_floats([1., 2., 3.], &device);
    let arr: [i32; 6] = [1, 2, 3, 4, 5, 6];
    let tensor_4 = Tensor::<Backend, 1, Int>::from_data(TensorData::from(&arr[0..3]), &device);

    struct BodyMetrics {
        age: i8,
        height: i16,
        weight: f32,
    }

    let bmi = BodyMetrics{
        age: 25,
        height: 180,
        weight: 70.0,
    };
    let data = TensorData::from([bmi.age as f32, bmi.height as f32, bmi.weight]);
    let tensor_5 = Tensor::<Backend, 1>::from_data(data, &device);
}

#[test]
fn test_ownership() {
    let device = Default::default();
    let input = Tensor::<Wgpu, 1>::from_floats([1., 2., 3., 4.], &device);
    let min = input.clone().min();
    let max = input.clone().max();
    let input = (input - min.clone()).div(max - min);
    println!("{}", input.to_data());
}

#[test]
fn test_tensor_display() {
    let tensor = Tensor::<Backend, 2>::full([2, 3], 0.123456789, &Default::default());
    println!("{}", tensor);
    println!("{:.2}", tensor);
}

type B = burn::backend::NdArray;

#[test]
fn test_check_closeness() {
    let device = Default::default();
    let tensor1 = Tensor::<B, 1>::from_floats(
        [1., 2., 3., 4., 5., 6.001, 7.002, 8.003, 9.004, 10.1],
        &device,);
    let tensor2 = Tensor::<B, 1>::from_floats(
        [1., 2., 3., 4., 5., 6.0, 7.001, 8.002, 9.003, 10.004],
        &device,);
    check_closeness(&tensor1, &tensor2);
}