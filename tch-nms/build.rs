use serde::{de::Error as _, Deserialize, Deserializer};
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize)]
struct Env {
    pub libtorch: PathBuf,
    pub cargo_manifest_dir: PathBuf,
    #[serde(
        deserialize_with = "deserialize_zero_one_bool",
        default = "default_libtorch_cxx11_abi"
    )]
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

    // compile nms_cpu.cpp
    {
        let mut build = cc::Build::new();
        build
            .cpp(true)
            .pic(true)
            .warnings(false)
            .flag("-std=c++14")
            .flag(&format!("-Wl,-rpath,{}", libtorch.join("lib").display()))
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
            .file(cargo_manifest_dir.join("csrc/nms_cpu.cpp"))
            .compile("libnms_cpu.a");
    }

    // compile nms_cuda.cu
    {
        let mut build = cc::Build::new();
        build
            .cuda(true)
            .pic(true)
            .warnings(false)
            .flag("-std=c++14")
            .flag("-cudart=shared")
            .flag("-arch=sm_52")
            .flag("-gencode=arch=compute_52,code=sm_52")
            .flag("-gencode=arch=compute_60,code=sm_60")
            .flag("-gencode=arch=compute_61,code=sm_61")
            .flag("-gencode=arch=compute_75,code=sm_75")
            .flag("-gencode=arch=compute_80,code=sm_80")
            .flag("-gencode=arch=compute_86,code=sm_86")
            .flag("-gencode=arch=compute_86,code=compute_86")
            .flag("-Xlinker")
            .flag(&format!("-rpath,{}", libtorch.join("lib").display()))
            .flag("-DWITH_CUDA=1")
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
            .file(cargo_manifest_dir.join("csrc/nms_cuda.cu"))
            .compile("libnms_cuda.a");
    }

    // re-compile if C++/CUDA source changed
    println!("cargo:rerun-if-changed=csrc/nms_cpu.cpp");
    println!("cargo:rerun-if-changed=csrc/nms_cuda.cu");

    // link CUDA
    println!("cargo:rustc-link-lib=cudart");

    // link PyTorch
    println!(
        "cargo:rustc-link-search=native={}",
        libtorch.join("lib").display()
    );
    println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=torch_cpu");
    println!("cargo:rustc-link-lib=torch_cuda");
    println!("cargo:rustc-link-lib=c10");
    println!("cargo:rustc-link-lib=c10_cuda");
}

fn deserialize_zero_one_bool<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    let output = match usize::deserialize(deserializer)? {
        0 => false,
        1 => true,
        value => {
            return Err(D::Error::custom(format!(
                "expect 0 or 1, but get {}",
                value
            )))
        }
    };
    Ok(output)
}

fn default_libtorch_cxx11_abi() -> bool {
    true
}
