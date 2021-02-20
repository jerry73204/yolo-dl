# tch-nms

It is PyTorch's CUDA implementation of Non-Maximum Suppression ported to tch.

The CUDA source code was taken from gdlg's pytorch_nms ([link](https://github.com/gdlg/pytorch_nms)).

## Build

Make sure the CUDA >= 10 and the CUDA compiler (nvcc) are present on your system.

The build script requires the environment variable `LIBTORCH` to be set to the directory of libtorch library.

```sh
cargo build
```

## License

See the [LICENSE.txt](LICENSE.txt) for MIT license. The license of original CUDA implementation can be foudn at [licenses/LICENSE-pytorch_nms.txt](licenses/LICENSE-pytorch_nms.txt);
