pub fn nearest_multiple(value: usize, multiple: usize) -> usize {
    (value + multiple - 1) & !(multiple - 1)
}
