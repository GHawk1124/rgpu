fn main() {
    println!("cargo::rerun-if-env-changed=LBM_NVVM");
    println!("cargo::rerun-if-env-changed=LBM_OXIDE");
    rgpu_build::Builder {
        nvvm: std::env::var_os("LBM_NVVM").is_some(),
        oxide: std::env::var_os("LBM_OXIDE").is_some(),
        amdgpu: vec![],
        ..Default::default()
    }
    .build()
}
