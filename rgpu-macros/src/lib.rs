use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Attribute, Error, Expr, ExprLit, FnArg, Ident, Item, ItemFn, ItemMod, Lit, Type, parse_macro_input, parse_quote};

/// Turns `#[kernel]` functions in a module into GPU entry points and typed host launchers.
///
/// A kernel takes its index first (`usize`, `[usize; 2]` or `[usize; 3]`) followed by
/// `&[T]`, `&mut [T]` or `Copy` values. `#[kernel(block = 128)]` or `#[kernel(block = [16, 16])]`
/// sets the block size.
#[proc_macro_attribute]
pub fn kernels(_: proc_macro::TokenStream, item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let module = parse_macro_input!(item as ItemMod);
    expand(module).unwrap_or_else(Error::into_compile_error).into()
}

fn expand(mut module: ItemMod) -> syn::Result<TokenStream> {
    let Some((_, items)) = &mut module.content else {
        return Err(Error::new_spanned(&module.ident, "#[rgpu::kernels] needs a module body"));
    };
    let mut kernels = Vec::new();
    for item in items.iter_mut() {
        let Item::Fn(f) = item else { continue };
        let Some(i) = f.attrs.iter().position(|a| a.path().is_ident("kernel")) else { continue };
        let attr = f.attrs.remove(i);
        kernels.push(Kernel::new(f, &attr)?);
        f.attrs.push(parse_quote!(#[inline(always)]));
        f.attrs.push(parse_quote!(#[allow(clippy::too_many_arguments)]));
    }
    let device = kernels.iter().map(Kernel::device);
    items.push(parse_quote! {
        #[allow(unused_imports)]
        use ::rgpu::prelude::*;
    });
    items.push(parse_quote! {
        #[cfg(rgpu)]
        pub mod __device {
            use super::*;
            #(#device)*
        }
    });
    items.push(Item::Verbatim(host(&kernels)));
    Ok(quote!(#module))
}

struct Kernel {
    name: Ident,
    index: Type,
    dims: usize,
    block: [u32; 3],
    params: Vec<(Ident, Param)>,
}

enum Param {
    Buf(Type),
    BufMut(Type),
    Val(Type),
}

impl Kernel {
    fn new(f: &ItemFn, attr: &Attribute) -> syn::Result<Kernel> {
        let mut types = Vec::new();
        for input in &f.sig.inputs {
            match input {
                FnArg::Typed(t) => types.push(&*t.ty),
                FnArg::Receiver(r) => return Err(Error::new_spanned(r, "kernels cannot take self")),
            }
        }
        let Some((&index, params)) = types.split_first() else {
            return Err(Error::new_spanned(&f.sig, "a kernel takes its index as the first parameter"));
        };
        let dims = match index {
            ty if is_usize(ty) => Some(1),
            Type::Array(a) if is_usize(&a.elem) => int(&a.len).ok().filter(|n| matches!(n, 2 | 3)).map(|n| n as usize),
            _ => None,
        }
        .ok_or_else(|| Error::new_spanned(index, "a kernel index is usize, [usize; 2] or [usize; 3]"))?;

        let mut block = [[256, 1, 1], [16, 16, 1], [8, 8, 4]][dims - 1];
        if !matches!(attr.meta, syn::Meta::Path(_)) {
            attr.parse_nested_meta(|m| {
                if !m.path.is_ident("block") {
                    return Err(m.error("the only kernel option is `block`"));
                }
                let value: Expr = m.value()?.parse()?;
                let sizes: Vec<&Expr> = match &value {
                    Expr::Array(a) => a.elems.iter().collect(),
                    e => vec![e],
                };
                // Threads past the index dimensions would not be masked and would repeat work.
                if sizes.len() > dims {
                    return Err(Error::new_spanned(&value, "the block has more dimensions than the index"));
                }
                for (b, e) in block.iter_mut().zip(sizes) {
                    *b = int(e)?;
                }
                Ok(())
            })?;
        }

        let params = params
            .iter()
            .enumerate()
            .map(|(i, &ty)| {
                let param = match ty {
                    Type::Reference(r) => match &*r.elem {
                        Type::Slice(s) if r.mutability.is_some() => Param::BufMut((*s.elem).clone()),
                        Type::Slice(s) => Param::Buf((*s.elem).clone()),
                        _ => return Err(Error::new_spanned(ty, "kernel references must be slices")),
                    },
                    ty => Param::Val(ty.clone()),
                };
                Ok((format_ident!("a{i}"), param))
            })
            .collect::<syn::Result<_>>()?;
        Ok(Kernel { name: f.sig.ident.clone(), index: index.clone(), dims, block, params })
    }

    fn index(&self) -> TokenStream {
        let d = (0..self.dims).map(syn::Index::from);
        match self.dims {
            1 => quote!(g[0] as usize),
            _ => quote!([#(g[#d] as usize),*]),
        }
    }

    fn device(&self) -> TokenStream {
        let name = &self.name;
        let [bx, by, bz] = self.block;
        let n: Vec<_> = (0..self.dims).map(|d| format_ident!("n{d}")).collect();
        let d = (0..self.dims).map(syn::Index::from);
        let inside = quote!(#(g[#d] < #n)&&*);
        let index = self.index();

        let mut ptx_params = Vec::new();
        let mut ptx_args = Vec::new();
        let mut push = Vec::new();
        let mut spirv_params = Vec::new();
        let mut spirv_args = Vec::new();
        let mut binding = 0u32;
        // Pointer and length come from a live `Buffer` the launcher borrows, so the slices below are valid.
        for (a, p) in &self.params {
            let len = format_ident!("{a}_len");
            match p {
                Param::Val(t) => {
                    ptx_params.push(quote!(#a: #t));
                    ptx_args.push(quote!(#a));
                    push.push(quote!(#a: #t));
                    spirv_args.push(quote!(p.#a));
                }
                Param::Buf(t) => {
                    ptx_params.push(quote!(#a: *const #t, #len: usize));
                    ptx_args.push(quote!(unsafe { ::core::slice::from_raw_parts(#a, #len) }));
                    spirv_params.push(quote!(#[spirv(storage_buffer, descriptor_set = 0, binding = #binding)] #a: &[#t]));
                    spirv_args.push(quote!(#a));
                    binding += 1;
                }
                Param::BufMut(t) => {
                    ptx_params.push(quote!(#a: *mut #t, #len: usize));
                    ptx_args.push(quote!(unsafe { ::core::slice::from_raw_parts_mut(#a, #len) }));
                    spirv_params.push(quote!(#[spirv(storage_buffer, descriptor_set = 0, binding = #binding)] #a: &mut [#t]));
                    spirv_args.push(quote!(#a));
                    binding += 1;
                }
            }
        }
        let ptx_body = quote! {
            let g = ::rgpu::device::global_id::<#bx, #by, #bz>();
            if #inside {
                super::#name(#index, #(#ptx_args),*)
            }
        };
        let push_ty = format_ident!("{name}_push");
        let entry = name.to_string();
        quote! {
            #[cfg(any(rgpu = "ptx", rgpu = "amdgpu"))]
            #[unsafe(no_mangle)]
            pub extern "gpu-kernel" fn #name(#(#n: u32,)* #(#ptx_params),*) {
                #ptx_body
            }

            #[cfg(rgpu = "nvvm")]
            #[unsafe(no_mangle)]
            #[nvvm_internal::kernel()]
            pub extern "C" fn #name(#(#n: u32,)* #(#ptx_params),*) {
                #ptx_body
            }

            #[cfg(rgpu = "oxide")]
            #[::rgpu::device::kernel]
            pub fn #name(#(#n: u32,)* #(#ptx_params),*) {
                #ptx_body
            }

            #[cfg(rgpu = "spirv")]
            #[allow(non_camel_case_types)]
            #[repr(C)]
            pub struct #push_ty {
                #(#n: u32,)*
                #(#push),*
            }

            #[cfg(rgpu = "spirv")]
            #[::rgpu::device::spirv(compute(threads(#bx, #by, #bz), entry_point_name = #entry))]
            pub fn #name(
                #[spirv(global_invocation_id)] g: ::rgpu::device::UVec3,
                #[spirv(push_constant)] p: &#push_ty,
                #(#spirv_params),*
            ) {
                let g = [g.x, g.y, g.z];
                #(let #n = p.#n;)*
                if #inside {
                    super::#name(#index, #(#spirv_args),*)
                }
            }
        }
    }

    fn launcher(&self) -> TokenStream {
        let name = &self.name;
        let index_ty = &self.index;
        let d = (0..self.dims).map(syn::Index::from);
        let cpu_index = self.index();
        let mut params = Vec::new();
        let mut views = Vec::new();
        let mut cpu_args = Vec::new();
        let mut args = Vec::new();
        for (a, p) in &self.params {
            match p {
                Param::Val(t) => {
                    params.push(quote!(#a: #t));
                    cpu_args.push(quote!(#a));
                    args.push(quote!(::rgpu::Arg::val(&#a)));
                }
                Param::Buf(t) => {
                    params.push(quote!(#a: &::rgpu::Buffer<#t>));
                    views.push(quote!(let #a = #a.view();));
                    cpu_args.push(quote!(#a.get()));
                    args.push(quote!(#a.arg()));
                }
                Param::BufMut(t) => {
                    params.push(quote!(#a: &mut ::rgpu::Buffer<#t>));
                    views.push(quote!(let #a = #a.view();));
                    cpu_args.push(quote!(#a.get_mut()));
                    args.push(quote!(#a.arg()));
                }
            }
        }
        quote! {
            pub fn #name(&self, n: #index_ty, #(#params),*) {
                let n = ::rgpu::Dims::dims(n);
                if self.module.is_cpu() {
                    #(#views)*
                    ::rgpu::cpu::launch(n, |g| #name(#cpu_index, #(#cpu_args),*));
                } else {
                    self.module.launch(&self.#name, n, &[#(::rgpu::Arg::val(&n[#d]),)* #(#args),*]);
                }
            }
        }
    }
}

fn host(kernels: &[Kernel]) -> TokenStream {
    let names: Vec<_> = kernels.iter().map(|k| &k.name).collect();
    let blocks = kernels.iter().map(|k| {
        let [x, y, z] = k.block;
        quote!([#x, #y, #z])
    });
    let launchers = kernels.iter().map(Kernel::launcher);
    quote! {
        #[cfg(not(rgpu))]
        pub static IMAGES: &[::rgpu::Image] = include!(concat!(env!("OUT_DIR"), "/rgpu.rs"));

        #[cfg(not(rgpu))]
        pub struct Module {
            module: ::rgpu::Module,
            #(#names: ::rgpu::Kernel,)*
        }

        #[cfg(not(rgpu))]
        pub fn load(gpu: &::rgpu::Gpu) -> Module {
            Module::new(gpu, IMAGES)
        }

        #[cfg(not(rgpu))]
        #[allow(clippy::too_many_arguments)]
        impl Module {
            pub fn new(gpu: &::rgpu::Gpu, images: &[::rgpu::Image]) -> Module {
                let module = gpu.load(images);
                Module {
                    #(#names: module.kernel(stringify!(#names), #blocks),)*
                    module,
                }
            }

            #(#launchers)*
        }
    }
}

fn int(e: &Expr) -> syn::Result<u32> {
    match e {
        Expr::Lit(ExprLit { lit: Lit::Int(i), .. }) => i.base10_parse(),
        e => Err(Error::new_spanned(e, "expected an integer literal")),
    }
}

fn is_usize(ty: &Type) -> bool {
    matches!(ty, Type::Path(p) if p.path.is_ident("usize"))
}

/// `unroll!(q in 0..19 { .. })` repeats the body with `q: usize` bound to each value, like `#pragma unroll`.
#[proc_macro]
pub fn unroll(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let Unroll { var, start, end, body } = parse_macro_input!(input as Unroll);
    let copies = (start..end).map(|i| quote!({ let #var: usize = #i; #body }));
    quote!({ #(#copies)* }).into()
}

struct Unroll {
    var: Ident,
    start: usize,
    end: usize,
    body: syn::Block,
}

impl syn::parse::Parse for Unroll {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Unroll> {
        let var = input.parse()?;
        input.parse::<syn::Token![in]>()?;
        let start = input.parse::<syn::LitInt>()?.base10_parse()?;
        input.parse::<syn::Token![..]>()?;
        let end = input.parse::<syn::LitInt>()?.base10_parse()?;
        Ok(Unroll { var, start, end, body: input.parse()? })
    }
}
