//! Compiles a kernel crate for every GPU backend and embeds the binaries in it.
//!
//! In the kernel crate's `build.rs`: `fn main() { rgpu_build::build() }`

use std::env;
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const NIGHTLY: &str = "nightly-2026-08-28";
const SPIRV_NIGHTLY: &str = "nightly-2026-07-03";
const SPIRV_VERSION: &str = "0.10.0";
const NVVM_NIGHTLY: &str = "nightly-2026-04-02";
const RUST_CUDA_REV: &str = "3a82c35a7196731fb60eebdac2a40d2b1bf0ba95";
const CUDA_OXIDE_REV: &str = "a0cc6cc0bc18722ec5cdfbea136c118a7c051977";

const SPIRV_TARGET: &str = "spirv-unknown-vulkan1.2";
const SPIRV_SPEC: &str = r#"{
  "arch": "spirv",
  "crt-objects-fallback": "false",
  "crt-static-allows-dylibs": true,
  "crt-static-respected": true,
  "data-layout": "e-m:e-p:32:32:32-i64:64-n8:16:32:64",
  "dll-prefix": "",
  "dll-suffix": ".spv.json",
  "dynamic-linking": true,
  "emit-debug-gdb-scripts": false,
  "env": "vulkan1.2",
  "linker-flavor": "unix",
  "linker-is-gnu": false,
  "llvm-target": "spirv-unknown-vulkan1.2",
  "main-needs-argc-argv": false,
  "metadata": { "description": null, "host_tools": null, "std": null, "tier": null },
  "panic-strategy": "abort",
  "simd-types-indirect": false,
  "target-pointer-width": 32
}"#;

pub struct Builder {
    /// PTX from rustc's nvptx backend.
    pub ptx: bool,
    /// PTX from Rust-CUDA's NVVM backend.
    pub nvvm: bool,
    /// PTX from NVIDIA's cuda-oxide backend.
    pub oxide: bool,
    /// SPIR-V from rust-gpu, run through Vulkan.
    pub spirv: bool,
    /// AMDGPU targets. The runtime tries them in this order so list exact ones before generic ones.
    pub amdgpu: Vec<String>,
    /// Lowest compute capability the PTX supports, 75 is `sm_75`.
    pub ptx_arch: u32,
    /// Compute capabilities to precompile cubins for with `ptxas`. Defaults to the GPUs in this machine.
    pub cubins: Vec<u32>,
    /// Fuse multiplies and adds into fma like nvcc does by default.
    pub fma: bool,
}

impl Default for Builder {
    fn default() -> Builder {
        let generic = ["gfx9-generic", "gfx9-4-generic", "gfx10-1-generic", "gfx10-3-generic", "gfx11-generic", "gfx12-generic"];
        Builder {
            ptx: true,
            nvvm: false,
            oxide: false,
            spirv: true,
            amdgpu: generic.map(String::from).into(),
            ptx_arch: 75,
            cubins: native_cubins(),
            fma: true,
        }
    }
}

pub fn build() {
    Builder::default().build()
}

struct Image {
    codegen: &'static str,
    format: &'static str,
    arch: String,
    path: PathBuf,
}

impl Builder {
    pub fn build(&self) {
        println!(r#"cargo::rustc-check-cfg=cfg(rgpu, values(none(), "ptx", "nvvm", "oxide", "spirv", "amdgpu"))"#);
        if env::var_os("RGPU_DEVICE").is_some() {
            return;
        }
        let out = PathBuf::from(env::var("OUT_DIR").unwrap());
        let k = Krate::new(&out);
        let mut images = Vec::new();

        if self.ptx {
            let tc = toolchain(NIGHTLY);
            let mut flags = vec![
                format!("-Ctarget-cpu=sm_{}", self.ptx_arch),
                "-Zunstable-options".into(),
                "-Clinker-flavor=llbc".into(),
                "-Zcrate-attr=feature(abi_gpu_kernel)".into(),
                // LLVM's NVPTX cost model does not credit unrolling for moving local arrays into registers the way
                // its AMDGPU one does, so raise the threshold to keep kernel arrays out of local memory like nvcc.
                "-Cllvm-args=-unroll-threshold=1000".into(),
            ];
            if self.fma {
                flags.push("-Cllvm-args=-nvptx-fma-level=1".into());
            }
            // Fat LTO makes rustc run the final codegen itself, so the PTX is emitted directly and `-Cllvm-args` apply.
            let ptx = out.join("ptx.s");
            let emit = format!("--emit=asm={},link", ptx.display());
            k.build(
                &tc,
                "ptx",
                "ptx",
                "nvptx64-nvidia-cuda",
                &flags,
                &["--crate-type", "cdylib", "--", &emit],
                &[("CARGO_PROFILE_RELEASE_LTO", "fat".into())],
            );
            images.extend(self.cuda("ptx", &ptx, &out));
        }

        if self.nvvm {
            let tc = toolchain(NVVM_NIGHTLY);
            let dep = format!(r#"rustc_codegen_nvvm = {{ git = "https://github.com/Rust-GPU/rust-cuda", rev = "{RUST_CUDA_REV}" }}"#);
            let mut llvm = format!("-arch=compute_{}", self.ptx_arch);
            if !self.fma {
                llvm.push_str(" -fma=0");
            }
            let flags = [
                backend("rustc_codegen_nvvm", &RUST_CUDA_REV[..12], &tc, &dep),
                "-Zcrate-attr=feature(register_tool)".into(),
                "-Zcrate-attr=register_tool(nvvm_internal)".into(),
                "-Zsaturating_float_casts=false".into(),
                format!("-Cllvm-args={llvm}"),
            ];
            let env = [("CARGO_FEATURE_NO_F16_F128", "1".into()), ("PATH", nvvm_path())];
            let ptx = k.build(&tc, "nvvm", "nvvm", "nvptx64-nvidia-cuda", &flags, &["--crate-type", "cdylib"], &env).with_extension("ptx");
            images.extend(self.cuda("nvvm", &ptx, &out));
        }

        if self.oxide {
            let tc = toolchain(NIGHTLY);
            let dep = format!(r#"rustc_codegen_cuda = {{ git = "https://github.com/NVIDIA/cuda-rust", rev = "{CUDA_OXIDE_REV}" }}"#);
            let flags = [
                backend("rustc_codegen_cuda", &CUDA_OXIDE_REV[..12], &tc, &dep),
                "-Copt-level=3".into(),
                "-Cdebug-assertions=off".into(),
                "-Zmir-enable-passes=-JumpThreading".into(),
                "-Zalways-encode-mir".into(),
                "-Csymbol-mangling-version=v0".into(),
            ];
            // cuda-oxide compiles the crate for a CPU target and writes the PTX of its kernels on the side.
            // Linux is used everywhere since it embeds a copy of the PTX into the object file, which it only supports for ELF.
            let ptx = out.join("oxide");
            let env = [
                ("CUDA_OXIDE_PTX_DIR", ptx.display().to_string()),
                ("CUDA_OXIDE_TARGET", format!("sm_{}", self.ptx_arch)),
                ("CUDA_OXIDE_DEVICE_CODEGEN_CRATE", k.name.clone()),
            ];
            let target = format!("{}-unknown-linux-gnu", host().split('-').next().unwrap());
            k.build(&tc, "oxide", "oxide", &target, &flags, &[], &env);
            images.extend(self.cuda("oxide", &ptx.join(&k.name).with_extension("ptx"), &out));
        }

        if self.spirv {
            let tc = toolchain(SPIRV_NIGHTLY);
            let spec = k.target.join(format!("{SPIRV_TARGET}.json"));
            fs::create_dir_all(&k.target).unwrap();
            fs::write(&spec, SPIRV_SPEC).unwrap();
            let flags = [
                backend("rustc_codegen_spirv", SPIRV_VERSION, &tc, &format!(r#"rustc_codegen_spirv = "={SPIRV_VERSION}""#)),
                "-Zbinary-dep-depinfo".into(),
                "-Csymbol-mangling-version=v0".into(),
                "-Zcrate-attr=feature(register_tool)".into(),
                "-Zcrate-attr=register_tool(rust_gpu)".into(),
                "-Coverflow-checks=off".into(),
                "-Cdebug-assertions=off".into(),
                "-Zinline-mir=off".into(),
                "-Zmir-enable-passes=-GVN".into(),
                "-Zshare-generics=off".into(),
            ];
            // rust-gpu finds entry points among all public items, which only a Rust dylib keeps.
            let args = ["--crate-type", "dylib", "-Zjson-target-spec", "-Zbuild-std-features=compiler-builtins-mem"];
            let spv = k.build(&tc, "spirv", "spirv", spec.to_str().unwrap(), &flags, &args, &[]);
            let spv = spv.parent().unwrap().join("deps").join(spv.file_name().unwrap()).with_extension("spv");
            images.push(Image { codegen: "spirv", format: "Spirv", arch: "vulkan1.2".into(), path: spv });
        }

        let tc = if self.amdgpu.is_empty() { String::new() } else { toolchain(NIGHTLY) };
        for arch in &self.amdgpu {
            let flags = [format!("-Ctarget-cpu={arch}"), "-Zcrate-attr=feature(abi_gpu_kernel)".into()];
            let elf = k.build(&tc, "amdgpu", &format!("amdgpu-{arch}"), "amdgcn-amd-amdhsa", &flags, &["--crate-type", "cdylib"], &[]);
            let elf = elf.with_extension("elf");
            images.push(Image { codegen: "amdgpu", format: "Amdgpu", arch: arch.clone(), path: elf });
        }

        let mut list = String::from("&[\n");
        for Image { codegen, format, arch, path } in images {
            let path = path.display();
            writeln!(list, r#"    ::rgpu::Image {{ codegen: "{codegen}", format: ::rgpu::Format::{format}, arch: "{arch}", data: include_bytes!(r"{path}") }},"#).unwrap();
        }
        list.push(']');
        fs::write(out.join("rgpu.rs"), list).unwrap();
    }

    fn cuda(&self, codegen: &'static str, ptx: &Path, out: &Path) -> Vec<Image> {
        let path = out.join(format!("{codegen}.ptx"));
        let mut text = fs::read(ptx).unwrap();
        // The driver needs PTX NUL terminated.
        text.push(0);
        fs::write(&path, text).unwrap();
        let mut images = vec![Image { codegen, format: "Ptx", arch: format!("sm_{}", self.ptx_arch), path }];
        let Some(ptxas) = ptxas() else { return images };
        for sm in &self.cubins {
            let path = out.join(format!("{codegen}-sm_{sm}.cubin"));
            run(Command::new(&ptxas).arg(format!("-arch=sm_{sm}")).arg(ptx).arg("-o").arg(&path));
            images.push(Image { codegen, format: "Cubin", arch: format!("sm_{sm}"), path });
        }
        images
    }
}

struct Krate {
    manifest: PathBuf,
    name: String,
    target: PathBuf,
}

impl Krate {
    fn new(out: &Path) -> Krate {
        // Cargo marks its target directory with CACHEDIR.TAG.
        let root = out.ancestors().find(|p| p.join("CACHEDIR.TAG").exists()).unwrap_or(out);
        Krate {
            manifest: PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("Cargo.toml"),
            name: env::var("CARGO_PKG_NAME").unwrap().replace('-', "_"),
            target: root.join("rgpu"),
        }
    }

    /// Builds this crate as a device cdylib and returns the artifact path without its extension.
    #[allow(clippy::too_many_arguments)]
    fn build(&self, tc: &str, backend: &str, dir: &str, target: &str, flags: &[String], args: &[&str], env: &[(&str, String)]) -> PathBuf {
        let dir = self.target.join(dir);
        let mut flags = flags.to_vec();
        flags.extend(["-Aunused_features".into(), "--cfg".into(), "rgpu".into(), "--cfg".into(), format!(r#"rgpu="{backend}""#)]);
        let mut cmd = cargo(tc);
        cmd.args(["rustc", "--lib", "--release", "-Zbuild-std=core", "--target", target])
            .arg("--manifest-path")
            .arg(&self.manifest)
            .arg("--target-dir")
            .arg(&dir)
            .args(args)
            .env("CARGO_ENCODED_RUSTFLAGS", flags.join("\x1f"))
            .env("CARGO_PROFILE_RELEASE_CODEGEN_UNITS", "1")
            .env("CARGO_PROFILE_RELEASE_DEBUG", "0")
            .env("RGPU_DEVICE", backend)
            .envs(env.iter().cloned());
        run(&mut cmd);

        let name = Path::new(target).file_stem().unwrap();
        let artifact = dir.join(name).join("release").join(&self.name);
        let release = artifact.parent().unwrap();
        let depfile = [format!("{}.d", self.name), format!("{}.spv.d", self.name), format!("lib{}.d", self.name)]
            .map(|d| release.join(d))
            .into_iter()
            .find(|d| d.exists())
            .unwrap();
        rerun_if_changed(&depfile, &self.target);
        artifact
    }
}

/// A cargo command for a pinned toolchain with the outer build's environment removed.
fn cargo(tc: &str) -> Command {
    let mut cmd = Command::new("rustup");
    cmd.args(["run", tc, "cargo"]).stdout(std::io::stderr());
    let inherited = [
        "CARGO",
        "CARGO_MAKEFLAGS",
        "CARGO_PRIMARY_PACKAGE",
        "CARGO_CRATE_NAME",
        "CARGO_TARGET_DIR",
        "CARGO_BUILD_TARGET",
        "CARGO_ENCODED_RUSTFLAGS",
        "RUSTC",
        "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER",
        "RUSTC_LINKER",
        "RUSTDOC",
        "RUSTFLAGS",
        "RUSTUP_TOOLCHAIN",
        "OUT_DIR",
        "TARGET",
        "HOST",
        "PROFILE",
        "OPT_LEVEL",
        "DEBUG",
        "NUM_JOBS",
        "MAKEFLAGS",
        "MFLAGS",
    ];
    let prefixes = ["CARGO_FEATURE_", "CARGO_CFG_", "CARGO_PKG_", "CARGO_MANIFEST_", "__CARGO"];
    for (k, _) in env::vars_os() {
        let k = k.to_string_lossy();
        if inherited.contains(&&*k) || prefixes.iter().any(|p| k.starts_with(p)) {
            cmd.env_remove(&*k);
        }
    }
    cmd
}

fn toolchain(date: &str) -> String {
    let tc = format!("{date}-{}", host());
    let components = ["rust-src", "rustc-dev", "llvm-tools", "llvm-bitcode-linker"];
    // `rustup toolchain install` checks the network even when nothing is missing, which costs seconds per build.
    let installed = Command::new("rustup").args(["component", "list", "--installed", "--toolchain", &tc]).output().unwrap();
    let installed = String::from_utf8_lossy(&installed.stdout);
    if components.iter().all(|c| installed.lines().any(|l| l.starts_with(c))) {
        return tc;
    }
    let mut install = Command::new("rustup");
    install.args(["toolchain", "install", "--no-self-update", "--profile", "minimal", &tc]);
    for c in components {
        install.args(["--component", c]);
    }
    run(&mut install);
    tc
}

/// The build host, with MSVC instead of GNU on Windows since prebuilt LLVM and spirv-tools link against it.
fn host() -> String {
    let host = env::var("HOST").unwrap();
    match host.split_once("-pc-windows-") {
        Some((arch, _)) => format!("{arch}-pc-windows-msvc"),
        None => host,
    }
}

/// Builds a rustc codegen backend dylib once, cached per user, and returns the flag that loads it.
fn backend(name: &str, version: &str, tc: &str, dependency: &str) -> String {
    let dir = cache().join(format!("{name}-{version}"));
    let file = format!("{}{name}{}", env::consts::DLL_PREFIX, env::consts::DLL_SUFFIX);
    let dylib = dir.join(&file);
    let flag = format!("-Zcodegen-backend={}", dylib.display());
    if dylib.exists() {
        return flag;
    }
    fs::create_dir_all(dir.join("src")).unwrap();
    fs::write(dir.join("src/lib.rs"), "").unwrap();
    let manifest = format!(
        "[package]\nname = \"rgpu-backend\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\n[dependencies]\n{dependency}\n\n[workspace]\n"
    );
    fs::write(dir.join("Cargo.toml"), manifest).unwrap();
    run(cargo(tc).current_dir(&dir).args(["build", "--release"]).env("PATH", nvvm_path()));
    fs::copy(dir.join("target/release").join(&file), &dylib).unwrap();
    flag
}

fn run(cmd: &mut Command) {
    assert!(cmd.status().unwrap().success(), "{cmd:?} failed");
}

fn cache() -> PathBuf {
    if let Some(dir) = env::var_os("RGPU_CACHE") {
        return dir.into();
    }
    let base = env::var_os("LOCALAPPDATA")
        .or_else(|| env::var_os("XDG_CACHE_HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env::var_os("HOME").unwrap()).join(".cache"));
    base.join("rgpu")
}

fn cuda_path() -> Option<PathBuf> {
    env::var_os("CUDA_PATH").or_else(|| env::var_os("CUDA_HOME")).map(PathBuf::from).or_else(|| {
        let p = PathBuf::from("/usr/local/cuda");
        p.exists().then_some(p)
    })
}

/// PATH with libnvvm added so the NVVM backend can load it.
fn nvvm_path() -> String {
    let path = env::var("PATH").unwrap_or_default();
    match cuda_path() {
        Some(cuda) => {
            let sep = if cfg!(windows) { ';' } else { ':' };
            let bin = cuda.join("nvvm").join("bin");
            format!("{}{sep}{}{sep}{path}", bin.join("x64").display(), bin.display())
        }
        None => path,
    }
}

fn ptxas() -> Option<PathBuf> {
    let exe = format!("ptxas{}", env::consts::EXE_SUFFIX);
    let candidate = cuda_path().map(|c| c.join("bin").join(&exe)).filter(|p| p.exists());
    candidate.or_else(|| Command::new(&exe).arg("--version").output().ok().map(|_| exe.into()))
}

fn native_cubins() -> Vec<u32> {
    // Device builds rerun this build script, so skip the query there.
    if env::var_os("RGPU_DEVICE").is_some() {
        return Vec::new();
    }
    let Ok(out) = Command::new("nvidia-smi").args(["--query-gpu=compute_cap", "--format=csv,noheader"]).output() else {
        return Vec::new();
    };
    let mut caps: Vec<u32> = String::from_utf8_lossy(&out.stdout).lines().filter_map(|l| l.trim().replace('.', "").parse().ok()).collect();
    caps.sort();
    caps.dedup();
    caps
}

/// Reruns the build script when any source the device build read changes, ignoring toolchain and registry sources.
fn rerun_if_changed(depfile: &Path, target: &Path) {
    let text = fs::read_to_string(depfile).unwrap();
    let line = text.lines().next().unwrap();
    let deps = &line[line.find(": ").unwrap() + 2..];
    let mut path = String::new();
    let mut chars = deps.chars().peekable();
    let mut paths = Vec::new();
    while let Some(c) = chars.next() {
        match c {
            '\\' if chars.peek() == Some(&' ') => path.push(chars.next().unwrap()),
            ' ' => paths.push(std::mem::take(&mut path)),
            c => path.push(c),
        }
    }
    paths.push(path);
    let skip = [".cargo", ".rustup", "rustlib"];
    let generated = [target, &cache()];
    for p in paths
        .iter()
        .filter(|p| !p.is_empty() && !skip.iter().any(|s| p.contains(s)) && !generated.iter().any(|g| Path::new(p).starts_with(g)))
    {
        println!("cargo::rerun-if-changed={p}");
    }
}
