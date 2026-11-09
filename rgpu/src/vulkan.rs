use std::ffi::{CStr, CString};
use std::mem::MaybeUninit;
use std::sync::{Mutex, OnceLock};

use ash::vk;

use crate::gpu::{Arg, ArgKind, Format, Image, Raw};

pub struct Device {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    push: ash::khr::push_descriptor::Device,
    memory: vk::PhysicalDeviceMemoryProperties,
    queue: vk::Queue,
    pool: vk::CommandPool,
    name: String,
    state: Mutex<State>,
}

struct State {
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    recording: bool,
}

pub struct Buf {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    bytes: usize,
}

pub struct Module(vk::ShaderModule);

pub struct Kernel {
    name: CString,
    pipeline: OnceLock<(vk::Pipeline, vk::PipelineLayout, vk::DescriptorSetLayout)>,
}

impl Kernel {
    pub fn new(name: &str) -> Kernel {
        Kernel { name: CString::new(name).unwrap(), pipeline: OnceLock::new() }
    }
}

impl Device {
    pub fn new(index: usize) -> Option<Device> {
        unsafe {
            let entry = ash::Entry::load().ok()?;
            let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
            let instance = entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None).ok()?;
            let mut physicals = instance.enumerate_physical_devices().ok()?;
            physicals.sort_by_key(|&p| instance.get_physical_device_properties(p).device_type != vk::PhysicalDeviceType::DISCRETE_GPU);
            let physical = *physicals.get(index)?;
            let properties = instance.get_physical_device_properties(physical);
            let family = instance
                .get_physical_device_queue_family_properties(physical)
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))? as u32;

            // Enable everything the device supports so any shader capability works, except bounds checked buffers.
            let mut v11 = vk::PhysicalDeviceVulkan11Features::default();
            let mut v12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut v11).push_next(&mut v12);
            instance.get_physical_device_features2(physical, &mut features);
            features.features.robust_buffer_access = vk::FALSE;

            let priority = [1.0];
            let queues = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priority)];
            let extensions = [ash::khr::push_descriptor::NAME.as_ptr()];
            let info =
                vk::DeviceCreateInfo::default().queue_create_infos(&queues).enabled_extension_names(&extensions).push_next(&mut features);
            let device = instance.create_device(physical, &info, None).ok()?;

            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .unwrap();
            let cmd = device
                .allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1))
                .unwrap()[0];
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();

            Some(Device {
                push: ash::khr::push_descriptor::Device::new(&instance, &device),
                memory: instance.get_physical_device_memory_properties(physical),
                queue: device.get_device_queue(family, 0),
                name: properties.device_name_as_c_str().unwrap().to_string_lossy().into(),
                state: Mutex::new(State { cmd, fence, recording: false }),
                pool,
                device,
                instance,
                _entry: entry,
            })
        }
    }

    pub fn name(&self) -> String {
        format!("{} (vulkan)", self.name)
    }

    fn record(&self, f: impl FnOnce(vk::CommandBuffer)) {
        let mut s = self.state.lock().unwrap();
        unsafe {
            if !s.recording {
                let begin = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                self.device.begin_command_buffer(s.cmd, &begin).unwrap();
                s.recording = true;
            }
            f(s.cmd);
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
            self.device.cmd_pipeline_barrier(
                s.cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }

    pub fn sync(&self) {
        let mut s = self.state.lock().unwrap();
        if !s.recording {
            return;
        }
        unsafe {
            self.device.end_command_buffer(s.cmd).unwrap();
            let cmds = [s.cmd];
            self.device.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], s.fence).unwrap();
            self.device.wait_for_fences(&[s.fence], true, u64::MAX).unwrap();
            self.device.reset_fences(&[s.fence]).unwrap();
        }
        s.recording = false;
    }

    fn buffer(&self, bytes: usize, flags: vk::MemoryPropertyFlags) -> Buf {
        // Fills work in whole words, so the buffer is rounded up to one for `zero` to reach every byte.
        let size = bytes.next_multiple_of(4) as u64;
        unsafe {
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
            let buffer = self.device.create_buffer(&vk::BufferCreateInfo::default().size(size).usage(usage), None).unwrap();
            let req = self.device.get_buffer_memory_requirements(buffer);
            let kind = (0..self.memory.memory_type_count)
                .find(|&i| req.memory_type_bits & (1 << i) != 0 && self.memory.memory_types[i as usize].property_flags.contains(flags))
                .expect("no suitable Vulkan memory type");
            let alloc = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(kind);
            let memory = self.device.allocate_memory(&alloc, None).unwrap();
            self.device.bind_buffer_memory(buffer, memory, 0).unwrap();
            Buf { buffer, memory, bytes }
        }
    }

    pub fn alloc(&self, bytes: usize) -> Buf {
        self.buffer(bytes, vk::MemoryPropertyFlags::DEVICE_LOCAL)
    }

    pub fn free(&self, b: &Buf) {
        self.sync();
        unsafe {
            self.device.destroy_buffer(b.buffer, None);
            self.device.free_memory(b.memory, None);
        }
    }

    pub fn zero(&self, b: &Buf) {
        self.record(|cmd| unsafe { self.device.cmd_fill_buffer(cmd, b.buffer, 0, vk::WHOLE_SIZE, 0) });
    }

    fn staged(&self, bytes: usize, f: impl FnOnce(&Buf, *mut MaybeUninit<u8>)) {
        let staging = self.buffer(bytes, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT);
        unsafe {
            let ptr = self.device.map_memory(staging.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).unwrap();
            f(&staging, ptr.cast());
            self.device.unmap_memory(staging.memory);
        }
        self.free(&staging);
    }

    pub fn write(&self, b: &Buf, data: &[MaybeUninit<u8>]) {
        self.staged(data.len(), |staging, ptr| unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            let region = [vk::BufferCopy::default().size(data.len() as u64)];
            self.record(|cmd| self.device.cmd_copy_buffer(cmd, staging.buffer, b.buffer, &region));
            self.sync();
        });
    }

    pub fn read(&self, b: &Buf, out: &mut [MaybeUninit<u8>]) {
        self.staged(out.len(), |staging, ptr| unsafe {
            let region = [vk::BufferCopy::default().size(out.len() as u64)];
            self.record(|cmd| self.device.cmd_copy_buffer(cmd, b.buffer, staging.buffer, &region));
            self.sync();
            std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), out.len());
        });
    }

    pub fn load(&self, images: &[Image]) -> Module {
        let image = images.iter().find(|i| i.format == Format::Spirv).expect("no SPIR-V image");
        let words: Vec<u32> = image.data.as_chunks().0.iter().map(|w| u32::from_le_bytes(*w)).collect();
        let info = vk::ShaderModuleCreateInfo::default().code(&words);
        Module(unsafe { self.device.create_shader_module(&info, None) }.unwrap())
    }

    pub fn unload(&self, m: &Module) {
        unsafe { self.device.destroy_shader_module(m.0, None) };
    }

    fn pipeline(&self, m: &Module, name: &CStr, buffers: u32, push: u32) -> (vk::Pipeline, vk::PipelineLayout, vk::DescriptorSetLayout) {
        unsafe {
            let bindings: Vec<_> = (0..buffers)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let set = vk::DescriptorSetLayoutCreateInfo::default()
                .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
                .bindings(&bindings);
            let set = self.device.create_descriptor_set_layout(&set, None).unwrap();
            let sets = [set];
            let ranges = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: push }];
            let layout = vk::PipelineLayoutCreateInfo::default().set_layouts(&sets).push_constant_ranges(&ranges);
            let layout = self.device.create_pipeline_layout(&layout, None).unwrap();
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(m.0).name(name);
            let info = [vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)];
            let pipeline = self.device.create_compute_pipelines(vk::PipelineCache::null(), &info, None).unwrap()[0];
            (pipeline, layout, set)
        }
    }

    pub fn destroy(&self, k: &Kernel) {
        if let Some(&(pipeline, layout, set)) = k.pipeline.get() {
            self.sync();
            unsafe {
                self.device.destroy_pipeline(pipeline, None);
                self.device.destroy_pipeline_layout(layout, None);
                self.device.destroy_descriptor_set_layout(set, None);
            }
        }
    }

    pub fn launch(&self, m: &Module, k: &Kernel, groups: [u32; 3], args: &[Arg]) {
        let mut infos = Vec::with_capacity(args.len());
        let mut push = [0u8; 256];
        let mut len = 0usize;
        let mut align = 4;
        for arg in args {
            match arg.0 {
                ArgKind::Val(p, size, a) => {
                    len = len.next_multiple_of(a);
                    unsafe { std::ptr::copy_nonoverlapping(p, push[len..len + size].as_mut_ptr(), size) };
                    len += size;
                    align = align.max(a);
                }
                // The range is the slice length kernels see.
                ArgKind::Buf(Raw::Vk(b), _) => infos.push([vk::DescriptorBufferInfo::default().buffer(b.buffer).range(b.bytes as u64)]),
                ArgKind::Buf(..) => unreachable!(),
            }
        }
        let len = len.next_multiple_of(align);
        let &(pipeline, layout, _) = k.pipeline.get_or_init(|| self.pipeline(m, &k.name, infos.len() as u32, len as u32));
        let writes: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            })
            .collect();
        self.record(|cmd| unsafe {
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.push.cmd_push_descriptor_set(cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &writes);
            self.device.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, &push[..len]);
            self.device.cmd_dispatch(cmd, groups[0], groups[1], groups[2]);
        });
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        self.sync();
        unsafe {
            let s = self.state.get_mut().unwrap();
            self.device.destroy_fence(s.fence, None);
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
