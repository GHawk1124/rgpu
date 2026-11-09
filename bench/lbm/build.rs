//! Compiles the CUDA C++ and handwritten PTX references to cubins for the GPU in this machine.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo::rerun-if-changed=cuda");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let caps = Command::new("nvidia-smi").args(["--query-gpu=compute_cap", "--format=csv,noheader"]).output().unwrap();
    let sm = format!("sm_{}", String::from_utf8(caps.stdout).unwrap().lines().next().unwrap().trim().replace('.', ""));
    println!("cargo::rustc-env=LBM_ARCH={sm}");
    let arch = format!("-arch={sm}");

    let mut nvcc = Command::new("nvcc");
    if let Some(cl) = cc::windows_registry::find_tool("x86_64-pc-windows-msvc", "cl.exe") {
        nvcc.arg("-ccbin").arg(cl.path().parent().unwrap()).envs(cl.env().iter().cloned());
    }
    run(nvcc.args(["-cubin", "-O3", &arch, "cuda/lbm.cu", "-o"]).arg(out.join("cpp.cubin")));
    run(Command::new("ptxas").args([&arch, "cuda/lbm.ptx", "-o"]).arg(out.join("hand.cubin")));
}

fn run(cmd: &mut Command) {
    assert!(cmd.status().unwrap().success(), "{cmd:?} failed");
}
