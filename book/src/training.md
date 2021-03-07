# Training

## Run Training Program

Run the command to train a model. It reads the configuration file `train.json5` in current directory by default.

```sh
# Start training
cargo run --release --bin train
```

If you wish to specify a custom configuration file,


```sh
# Start training
cargo run --release --bin train -- --config /path/to/your/config.json5
```

## Profile the Training

Enable `profiling` feature to  measure the timing of each stage in the pipeline. It is useful to investigate the performance bottleneck.

```sh
cargo run --release --bin train --features profiling
```

It shows every timing profile by default. To show only specific profile, read the terminal to find available profile names.

```
Feb 17 19:32:02.184  INFO yolo_dl::profiling:registered timing profile 'pipeline'
Feb 17 19:32:02.185  INFO yolo_dl::profiling:registered timing profile 'cache loader'
```

Set `YOLODL_PROFILING_WHITELIST` variable to show only your interested profile.

```sh
env YOLODL_PROFILING_WHITELIST='pipeline' \
    cargo run --release --bin train --features profiling
```

## Performance Tuning

The default Rust toolchain has _debug_ and _release_ profiles. You can add `--release` option to cargo commands to switch from debug to release mode.

```sh
cargo run [--release]
cargo test [--release]
```

The _release_ profile settings can be found in `Cargo.toml`. In most cases, we enable verbose debug messages and debug assertions.

```toml
[profile.release]
debug = true
debug-assertions = true
overflow-checks = true
lto = false
```

You can disable debugging features to get better performance. Use it with caution that it disables several numeral checks, for example, NaN detection.

```toml
[profile.release]
debug = false
debug-assertions = false
overflow-checks = false
lto = true
```

More profile settings can be found in Cargo reference.

https://doc.rust-lang.org/cargo/reference/profiles.html

## Show Statistics in TensorBoard

The program reads The logging directory in `train.json` and write checkpoint files and TensorBoard logs in that directory. For example, to save the logs in `logs/` directory,

```json
"logging": {
    "dir": "logs",
}

```

When the training program is running, you can open a TensorBoard server to show the real-time statistics.

```sh
tensorboard --bind_all --logdir logs-coco/
```
