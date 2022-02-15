use unit_wrapper::unit_wrapper;

#[test]
fn f64_wrapper_test() {
    unit_wrapper!(Pixel);

    let zero: Pixel<f64> = Pixel(2.0) * Pixel(3.0) - Pixel(6.0);
    assert!(zero.0.abs() <= 1e-10);
}
