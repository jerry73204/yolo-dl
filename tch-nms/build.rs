use serde::{de::Error as _, Deserialize, Deserializer};
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize)]
struct Env {
    pub libtorch: PathBuf,
    pub cargo_manifest_dir: PathBuf,
    #[serde(deserialize_with = "deserialize_zero_one_bool")]
    pub libtorch_cxx11_abi: bool,
}

fn main() {
    let Env {
        libtorch,
        cargo_manifest_dir,
        libtorch_cxx11_abi,
    } = envy::from_env().unwrap();

    // find python3
    let python3_lib = pkg_config::Config::new().probe("python3").unwrap();

    let mut build = cc::Build::new();
    build
        .cuda(true)
        .pic(true)
        .warnings(false)
        .flag("-std=c++14")
        .flag("-cudart=shared")
        .flag("-gencode")
        .flag("arch=compute_61,code=sm_61")
        .flag("-Xlinker")
        .flag(&format!("-rpath,{}", libtorch.join("lib").display()))
        .flag(&format!("-D_GLIBCXX_USE_CXX11_ABI={}", libtorch_cxx11_abi))
        .include(libtorch.join("include"))
        .include(
            libtorch
                .join("include")
                .join("torch")
                .join("csrc")
                .join("api")
                .join("include"),
        );
    python3_lib.include_paths.iter().for_each(|path| {
        build.include(path);
    });
    build
        .file(cargo_manifest_dir.join("nms_kernel.cu"))
        .compile("libnms_kernel.a");

    // re-compile if CUDA source changed
    println!("cargo:rerun-if-changed=nms_kernel.cu");

    // link CUDA
    println!("cargo:rustc-link-lib=cudart");

    // link PyTorch
    println!(
        "cargo:rustc-link-search=native={}",
        libtorch.join("lib").display()
    );
    println!("cargo:rustc-link-lib=torch_cuda");
    println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=torch_cpu");
    println!("cargo:rustc-link-lib=c10");
}

fn deserialize_zero_one_bool<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    let output = match Option::<usize>::deserialize(deserializer)? {
        Some(0) => false,
        Some(1) | None => true,
        Some(value) => {
            return Err(D::Error::custom(format!(
                "expect 0 or 1, but get {}",
                value
            )))
        }
    };
    Ok(output)
}
