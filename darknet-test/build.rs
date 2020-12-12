use anyhow::{format_err, Result};
use fs_extra::dir::CopyOptions;
use std::{
    env, fs,
    path::{Path, PathBuf},
};

const DARKNET_SRC_ENV: &'static str = "DARKNET_SRC";
// const DARKNET_INCLUDE_PATH_ENV: &'static str = "DARKNET_INCLUDE_PATH";

lazy_static::lazy_static! {
    static ref BINDINGS_SRC_PATH: PathBuf = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("Failed to get CARGO_MANIFEST_DIR")).join("src").join("bindings.rs");
    static ref BINDINGS_TARGET_PATH: PathBuf = PathBuf::from(env::var("OUT_DIR").expect("Failed to get OUT_DIR")).join("bindings.rs");
    static ref LIBRARY_PATH: PathBuf = PathBuf::from(env::var("OUT_DIR").expect("Failed to get OUT_DIR")).join("darknet");
}

// Guess the cmake profile using the rule defined in the link.
// https://docs.rs/cmake/0.1.42/src/cmake/lib.rs.html#475-536
fn guess_cmake_profile() -> &'static str {
    // Determine Rust's profile, optimization level, and debug info:
    #[derive(PartialEq)]
    enum RustProfile {
        Debug,
        Release,
    }
    #[derive(PartialEq, Debug)]
    enum OptLevel {
        Debug,
        Release,
        Size,
    }

    let rust_profile = match env::var("PROFILE").unwrap().as_str() {
        "debug" => RustProfile::Debug,
        "release" | "bench" => RustProfile::Release,
        unknown => {
            eprintln!(
                "Warning: unknown Rust profile={}; defaulting to a release build.",
                unknown
            );
            RustProfile::Release
        }
    };

    let opt_level = match env::var("OPT_LEVEL").unwrap().as_str() {
        "0" => OptLevel::Debug,
        "1" | "2" | "3" => OptLevel::Release,
        "s" | "z" => OptLevel::Size,
        unknown => {
            let default_opt_level = match rust_profile {
                RustProfile::Debug => OptLevel::Debug,
                RustProfile::Release => OptLevel::Release,
            };
            eprintln!(
                "Warning: unknown opt-level={}; defaulting to a {:?} build.",
                unknown, default_opt_level
            );
            default_opt_level
        }
    };

    let debug_info: bool = match env::var("DEBUG").unwrap().as_str() {
        "false" => false,
        "true" => true,
        unknown => {
            eprintln!("Warning: unknown debug={}; defaulting to `true`.", unknown);
            true
        }
    };

    match (opt_level, debug_info) {
        (OptLevel::Debug, _) => "Debug",
        (OptLevel::Release, false) => "Release",
        (OptLevel::Release, true) => "RelWithDebInfo",
        (OptLevel::Size, _) => "MinSizeRel",
    }
}

fn is_dynamic() -> bool {
    return cfg!(feature = "dylib");
}

fn is_cuda_enabled() -> bool {
    cfg!(feature = "enable-cuda")
}

fn is_cudnn_enabled() -> bool {
    cfg!(feature = "enable-cudnn")
}

fn is_opencv_enabled() -> bool {
    cfg!(feature = "enable-opencv")
}

fn build_with_cmake<P>(src_path: P) -> Result<()>
where
    P: AsRef<Path>,
{
    let link = if is_dynamic() { "dylib" } else { "static" };
    let src_path = src_path.as_ref();
    let dst_path = LIBRARY_PATH.as_path();

    if dst_path.exists() {
        fs::remove_dir_all(dst_path)?;
    }

    fs_extra::dir::copy(
        src_path,
        dst_path,
        &CopyOptions {
            content_only: true,
            ..Default::default()
        },
    )?;

    let dst = cmake::Config::new(dst_path)
        .uses_cxx11()
        .define("BUILD_SHARED_LIBS", if is_dynamic() { "ON" } else { "OFF" })
        .define("ENABLE_CUDA", if is_cuda_enabled() { "ON" } else { "OFF" })
        .define(
            "ENABLE_CUDNN",
            if is_cudnn_enabled() { "ON" } else { "OFF" },
        )
        .define(
            "ENABLE_OPENCV",
            if is_opencv_enabled() { "ON" } else { "OFF" },
        )
        .build();
    println!("cargo:rustc-link-search={}", dst.join("build").display());

    // link to different target under distinct profiles
    match guess_cmake_profile() {
        "Debug" => println!("cargo:rustc-link-lib={}=darknetd", link),
        _ => println!("cargo:rustc-link-lib={}=darknet", link),
    }
    if !is_dynamic() {
        println!("cargo:rustc-link-lib=gomp");
        println!("cargo:rustc-link-lib=stdc++");
        if is_cuda_enabled() {
            println!("cargo:rustc-link-lib=cudart");
            println!("cargo:rustc-link-lib=cudnn");
            println!("cargo:rustc-link-lib=cublas");
            println!("cargo:rustc-link-lib=curand");
        }
    }

    // find header files

    let src_dir = dst_path.join("src");
    let include_files = vec![
        dst_path.join("include").join("darknet.h"),
        src_dir.join("network.h"),
        src_dir.join("activation_layer.h"),
        src_dir.join("avgpool_layer.h"),
        src_dir.join("batchnorm_layer.h"),
        src_dir.join("connected_layer.h"),
        src_dir.join("conv_lstm_layer.h"),
        src_dir.join("convolutional_layer.h"),
        src_dir.join("cost_layer.h"),
        src_dir.join("crnn_layer.h"),
        src_dir.join("crop_layer.h"),
        src_dir.join("deconvolutional_layer.h"),
        src_dir.join("detection_layer.h"),
        src_dir.join("dropout_layer.h"),
        src_dir.join("gaussian_yolo_layer.h"),
        src_dir.join("gru_layer.h"),
        src_dir.join("local_layer.h"),
        src_dir.join("lstm_layer.h"),
        src_dir.join("maxpool_layer.h"),
        src_dir.join("normalization_layer.h"),
        src_dir.join("region_layer.h"),
        src_dir.join("reorg_layer.h"),
        src_dir.join("reorg_old_layer.h"),
        src_dir.join("rnn_layer.h"),
        src_dir.join("route_layer.h"),
        src_dir.join("sam_layer.h"),
        src_dir.join("scale_channels_layer.h"),
        src_dir.join("shortcut_layer.h"),
        src_dir.join("softmax_layer.h"),
        src_dir.join("upsample_layer.h"),
        src_dir.join("yolo_layer.h"),
    ];

    // generate bindings
    let builder = bindgen::Builder::default()
        .clang_arg(format!("-I{}", dst_path.join("include").to_str().unwrap()));

    let builder = include_files
        .iter()
        .try_fold(builder, |builder, path| -> Result<_> {
            // let path = path.as_ref();
            let builder = builder.header(
                path.to_str()
                    .ok_or_else(|| format_err!("cannot create path to {}", path.display()))?,
            );
            Ok(builder)
        })?;

    builder
        .whitelist_function("make_network")
        .whitelist_function("forward_network")
        .whitelist_function("update_network")
        .whitelist_function("backward_output")
        .whitelist_function("train_network.*")
        .whitelist_function("get_network_.*")
        .whitelist_function("network_*")
        .whitelist_function("forward_.*_layer")
        .whitelist_function("backward_.*_layer")
        .whitelist_function("resize_.*_layer")
        .generate()
        .map_err(|_| format_err!("failed to generate bindings"))?
        .write_to_file(&*BINDINGS_TARGET_PATH)?;

    Ok(())
}

fn build_from_source() -> Result<()> {
    let src_dir: PathBuf = match env::var_os(DARKNET_SRC_ENV) {
        Some(src) => src.into(),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("darknet"),
    };
    build_with_cmake(src_dir)?;

    Ok(())
}

fn main() -> Result<()> {
    println!("cargo:rerun-if-env-changed={}", DARKNET_SRC_ENV);
    // println!("cargo:rerun-if-env-changed={}", DARKNET_INCLUDE_PATH_ENV);
    // println!(
    //     "cargo:rerun-if-env-changed={}",
    //     BINDINGS_TARGET_PATH.display()
    // );
    if cfg!(feature = "docs-rs") {
        return Ok(());
    }
    // build from source by default
    build_from_source()?;
    Ok(())
}
