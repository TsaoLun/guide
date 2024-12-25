use burn::{backend::Wgpu, tensor::{Int, Tensor, TensorData}};

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
