use cgmath::{Vector3, Quaternion, Rotation3, Deg, Zero, InnerSpace};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    vertex_attr_array, BindGroup, BindGroupDescriptor, BindGroupLayoutDescriptor, BlendState,
    Buffer, BufferAddress, BufferUsages, Color, ColorTargetState, ColorWrites,
    CommandEncoderDescriptor, Device, DepthStencilState, FragmentState, FrontFace, IndexFormat,
    InstanceDescriptor, MultisampleState, PipelineLayoutDescriptor, PolygonMode, PrimitiveState,
    PrimitiveTopology, Queue, RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline,
    RenderPipelineDescriptor, ShaderModuleDescriptor, Surface, SurfaceConfiguration, SurfaceError,
    VertexAttribute, VertexBufferLayout, VertexState, VertexStepMode, BindGroupLayoutEntry, BindGroupLayout,
};
use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

use camera::{Camera, CameraController};
use std::{f64::consts::PI, mem::size_of};
use util::color_from_hsva;

use crate::camera::CameraUniform;

mod camera;
mod texture;
mod util;

const CLEAR_COLOUR: Color = Color {
    r: 0.1,
    g: 0.2,
    b: 0.3,
    a: 1.0,
};

const CAMERA_SPEED: f32 = 0.2;
const INSTANCES_PER_ROW: u32 = 10;
const INSTANCE_DISPLACEMENT: Vector3<f32> = Vector3::new(
    INSTANCES_PER_ROW as f32 * 0.5,
    0.0,
    INSTANCES_PER_ROW as f32 * 0.5,
);

const VERTICES: &[Vertex] = &[
    // Changed
    Vertex {
        position: [-0.0868241, 0.49240386, 0.0],
        tex_coords: [0.4131759, 0.00759614],
    }, // A
    Vertex {
        position: [-0.49513406, 0.06958647, 0.0],
        tex_coords: [0.0048659444, 0.43041354],
    }, // B
    Vertex {
        position: [-0.21918549, -0.44939706, 0.0],
        tex_coords: [0.28081453, 0.949397],
    }, // C
    Vertex {
        position: [0.35966998, -0.3473291, 0.0],
        tex_coords: [0.85967, 0.84732914],
    }, // D
    Vertex {
        position: [0.44147372, 0.2347359, 0.0],
        tex_coords: [0.9414737, 0.2652641],
    }, // E
];

const SCREEN: &[Vertex] = &[
    Vertex {
        position: [-1.0, 0.0, 0.0],
        tex_coords: [0.0, 1.0],
    },
    Vertex {
        position: [0.0, 0.0, 0.0],
        tex_coords: [1.0, 1.0],
    },
    Vertex {
        position: [-1.0, 1.0, 0.0],
        tex_coords: [0.0, 0.0],
    },
    Vertex {
        position: [0.0, 1.0, 0.0],
        tex_coords: [1.0, 0.0],
    },
];

const SCREEN_INDICES: &[u16] = &[0, 1, 2, 1, 3, 2];

const INDICES: &[u16] = &[0, 1, 4, 1, 2, 4, 2, 3, 4];

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

struct State {
    surface: Surface,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,
    size: PhysicalSize<u32>,
    window: Window,
    render_pipeline: RenderPipeline,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    diffuse_bind_group: BindGroup,
    diffuse_texture: texture::Texture,
    depth_texture: texture::Texture,
    depth_bind_group_layout: BindGroupLayout,
    depth_bind_group: BindGroup,
    depth_vertex_buffer: Buffer,
    depth_index_buffer: Buffer,
    depth_render_pipeline: RenderPipeline,
    camera: Camera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: Buffer,
    camera_bind_group: BindGroup,
    num_indices: u32,
    instances: Vec<Instance>,
    instance_buffer: Buffer,

    // Surface challenge
    clear_colour: Color,
    changing_clear_colour: bool,
    cursor_position: Option<PhysicalPosition<f64>>,

    // Texture challenge
    freud_texture: texture::Texture,
    freud_bind_group: wgpu::BindGroup,
    freud: bool,
}

// Struct that represents the vertices of objects in our scene
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
}

struct Instance {
    position: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    matrix: [[f32; 4]; 4],
}

impl Instance {
    // A mat4x4 is really just 4 vec4's. So we need these 4 vectors.
    const ATTRS: [VertexAttribute; 4] = vertex_attr_array![5 => Float32x4, 6 => Float32x4, 7 => Float32x4, 8 => Float32x4];

    fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            matrix: (cgmath::Matrix4::from_translation(self.position)
                * cgmath::Matrix4::from(self.rotation))
            .into(),
        }
    }

    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        VertexBufferLayout { 
            array_stride: size_of::<InstanceRaw>() as BufferAddress, 
            step_mode: VertexStepMode::Instance, 
            attributes: &Self::ATTRS,
        }
    }
}

impl Vertex {
    const ATTRS: [VertexAttribute; 2] = vertex_attr_array![0 => Float32x3, 1 => Float32x2];

    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        VertexBufferLayout {
            array_stride: size_of::<Vertex>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            /*
            attributes: &[
                VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: VertexFormat::Float32x3,
                },

                VertexAttribute {
                    offset: size_of::<[f32; 3]>() as BufferAddress,
                    shader_location: 1,
                    format: VertexFormat::Float32x3,
                }
            ]
            */
            attributes: &Self::ATTRS,
        }
    }
}


fn create_instances() -> Vec<Instance> {
    (0..INSTANCES_PER_ROW).flat_map(|z|
        (0..INSTANCES_PER_ROW).map(move |x| {
            let position = Vector3::new(x as f32, 0.0, z as f32) - INSTANCE_DISPLACEMENT;

            let rotation = if position.is_zero() { 
                Quaternion::from_axis_angle(Vector3::unit_z(), Deg(45.0))
            } else {
                // put the following line in for an odd bug
                //Quaternion::from_axis_angle(position, Deg(45.0))
                Quaternion::from_axis_angle(position.normalize(), Deg(45.0))
            };

            Instance {
                position,
                rotation,
            }
        })).collect::<Vec<_>>()
}

pub async fn run() {
    // To build a window with winit, we need to build an event loop first
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("太鼓の達人")
        .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT))
        .build(&event_loop)
        .expect("Couldn't build the window");

    let mut state = State::new(window).await;

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            window_id,
            ref event,
        } if window_id == state.window().id() => {
            if !state.input(event) {
                match event {
                    // Closing the window
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => control_flow.set_exit(),

                    // Resizing the window
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }

                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        state.resize(**new_inner_size);
                    }

                    _ => {}
                }
            }
        }

        Event::RedrawRequested(window_id) if window_id == state.window().id() => {
            state.update();

            match state.render() {
                Ok(_) => {}
                Err(SurfaceError::Lost) => state.resize(state.size),
                Err(SurfaceError::OutOfMemory) => control_flow.set_exit(),
                Err(e) => log::error!("{e:?}"),
            }
        }

        Event::MainEventsCleared => state.window().request_redraw(),

        _ => {}
    });
}

impl State {
    async fn new(window: Window) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });

        // # Safety
        //
        // The surface needs to live as long as the window that
        // created it. State owns the window so this should
        // be safe.
        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: Default::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        // Just curious
        let backend = adapter.get_info().backend;
        dbg!(backend);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    // If we were targeting wasm32 we would use
                    // wgpu::Limits::downlevel_webgl12_defaults()
                    limits: Default::default(),
                    label: None,
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);

        // Shader code in this tutorial assumes an sRGB surface
        // texture. Using a different one will result in all the
        // colours coming out darker. If you want to support
        // non sRGB surfaces, you'll need to account for that
        // when drawing to the frame.
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .filter(|f| f.describe().srgb)
            .next()
            .unwrap_or(surface_caps.formats[0]);

        if size.width == 0 || size.height == 0 {
            panic!("SurfaceTexture width or height set to 0");
        }

        let config = SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let diffuse_bytes = include_bytes!("../assets/dababy.jpg");
        let diffuse_texture =
            texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "dababy").unwrap();

        let freud_bytes = include_bytes!("../assets/sigmundfreud256.png");
        let freud_texture =
            texture::Texture::from_bytes(&device, &queue, freud_bytes, "freud").unwrap();

        // A BindGroup describes a set of resources and how they can be
        // accessed by the shader. We start by defining the layout of it.
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // This should match the filterable field
                        // of the corresponding Texture entry above
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
            ],
            label: Some("diffuse_bind_group"),
        });

        let freud_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&freud_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&freud_texture.sampler),
                },
            ],
            label: Some("freud_bind_group"),
        });

        let camera = Camera {
            // We want to be 1 unit up and 2 back.
            // +z is out of the screen
            eye: (0.0, 1.0, 2.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };

        let camera_controller = CameraController::new(CAMERA_SPEED);

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_projection(&camera);

        let camera_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("camera"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("camera_bind_group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let d3_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/instances_shader.wgsl").into()),
        });

        let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&texture_bind_group_layout, &camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: VertexState {
                module: &d3_shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc(), Instance::desc()],
            },

            fragment: Some(FragmentState {
                module: &d3_shader,
                entry_point: "fs_main",
                targets: &[Some(ColorTargetState {
                    format: config.format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
            }),

            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                //cull_mode: Some(Face::Back), // enable when rendering 3d models
                cull_mode: None,
                polygon_mode: PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },

            depth_stencil: Some(DepthStencilState {
                format: texture::Texture::DEPTH_TEXTURE_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),

            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },

            multiview: None,
        });

        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: BufferUsages::INDEX,
        });

        let instances = create_instances();
        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();

        // For some reason, the instance buffer needs to be a vertex buffer.
        let instance_buffer = device.create_buffer_init(&BufferInitDescriptor { 
            label: Some("Instance Buffer"), 
            contents: bytemuck::cast_slice(&instance_data), 
            usage: BufferUsages::VERTEX,
        });

        let depth_texture = texture::Texture::depth_texture(&device, &config, "depth_texture");

        // We need a new bind group layout for the depth texture because it needs a comparison sampler
        // whatever that is
        let depth_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor { 
            label: Some("depth_bind_group_layout"), 
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture { 
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2, 
                    multisampled: false,
                },
                count: None,
            },

            BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            }]
        });

        let depth_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("depth_bind_group"),
            layout: &depth_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&depth_texture.view),
            },
            
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&depth_texture.sampler),
            }],
        });

        let d2_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("2d shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/depth_texture_shader.wgsl").into()),
        });

        let depth_render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Depth Render Pipeline Layout"),
            bind_group_layouts: &[&depth_bind_group_layout],
            push_constant_ranges: &[],
        });

        let depth_render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Depth Render Pipeline"),
            layout: Some(&depth_render_pipeline_layout),
            vertex: VertexState {
                module: &d2_shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },

            fragment: Some(FragmentState {
                module: &d2_shader,
                entry_point: "fs_main",
                targets: &[Some(ColorTargetState {
                    format: config.format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
            }),

            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                //cull_mode: Some(Face::Back), // enable when rendering 3d models
                cull_mode: None,
                polygon_mode: PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },

            depth_stencil: None,

            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },

            multiview: None,
        });

        let depth_vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Depth vertex buffer"),
            contents: bytemuck::cast_slice(SCREEN),
            usage: BufferUsages::VERTEX,
        });

        let depth_index_buffer = device.create_buffer_init(&BufferInitDescriptor { 
            label: Some("Depth index buffer"), 
            contents: bytemuck::cast_slice(SCREEN_INDICES), 
            usage: BufferUsages::INDEX,
        });

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            diffuse_bind_group,
            diffuse_texture,
            depth_texture,
            depth_bind_group,
            depth_bind_group_layout,
            depth_vertex_buffer,
            depth_index_buffer,
            depth_render_pipeline,
            camera,
            camera_controller,
            camera_uniform,
            camera_bind_group,
            camera_buffer,
            instances,
            instance_buffer,

            num_indices: INDICES.len() as u32,
            clear_colour: CLEAR_COLOUR,
            changing_clear_colour: false,
            cursor_position: None,

            freud_texture,
            freud_bind_group,
            freud: false,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture = texture::Texture::depth_texture(&self.device, &self.config, "depth_texture");
            self.depth_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.depth_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.depth_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.depth_texture.sampler),
                    },
                ],
                label: Some("depth_pass.bind_group"),
            });
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        let camera_res = self.camera_controller.process_events(event);

        let res = match *event {
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_position = Some(position);
                true
            }

            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(key),
                        ..
                    },
                ..
            } => {
                match key {
                    VirtualKeyCode::Tab => self.changing_clear_colour = !self.changing_clear_colour,
                    VirtualKeyCode::Space => self.freud = !self.freud,
                    _ => {}
                }

                true
            }

            _ => false,
        };

        // This isn't a great way of doing it but i dunno
        res || camera_res
    }

    fn update(&mut self) {
        // Not very efficient but I don't really mind
        if self.changing_clear_colour {
            if let Some(colour) = self.cursor_position.and_then(|position| {
                color_from_hsva(
                    2.0 * PI * position.x / self.size.width as f64,
                    1.0,
                    position.y / self.size.height as f64,
                    1.0,
                )
            }) {
                self.clear_colour = colour;
            }
        }

        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_projection(&self.camera);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
    }

    fn render(&mut self) -> Result<(), SurfaceError> {
        let output = self.surface.get_current_texture()?;

        let view = output.texture.create_view(&Default::default());

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Create the render pass. We need to drop this afterwards
        // so that we can call encoder.finish(), as encoder
        // is borrowed mutably here.
        let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(self.clear_colour),
                    store: true,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_texture.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: true,
                }),
                stencil_ops: None,
            }),
        });

        render_pass.set_pipeline(&self.render_pipeline);
        let bind_group = if self.freud {
            &self.freud_bind_group
        } else {
            &self.diffuse_bind_group
        };

        // diffuse texture
        render_pass.set_bind_group(0, bind_group, &[]);
        // camera
        render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
        // vertices
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        // instances
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        // indices
        render_pass.set_index_buffer(self.index_buffer.slice(..), IndexFormat::Uint16);
        // draw!
        render_pass.draw_indexed(0..self.num_indices, 0, 0..self.instances.len() as _);

        drop(render_pass);

        // Now for the depth texture
        let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Depth Render Pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
        });

        render_pass.set_pipeline(&self.depth_render_pipeline);

        render_pass.set_bind_group(0, &self.depth_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.depth_vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.depth_index_buffer.slice(..), IndexFormat::Uint16);
        render_pass.draw_indexed(0..SCREEN_INDICES.len() as _, 0, 0..1);

        drop(render_pass);

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
