use egui::FontDefinitions;
use egui_demo_lib::DemoWindows;
use egui_wgpu::renderer::ScreenDescriptor;
use egui_winit_platform::{Platform, PlatformDescriptor};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    vertex_attr_array, BlendState, Buffer, BufferAddress, BufferUsages, Color, ColorTargetState,
    ColorWrites, CommandEncoderDescriptor, Device, Face, FragmentState, FrontFace, IndexFormat,
    Instance, InstanceDescriptor, MultisampleState, PipelineLayoutDescriptor, PolygonMode,
    PrimitiveState, PrimitiveTopology, Queue, RenderPassColorAttachment, RenderPassDescriptor,
    RenderPipeline, RenderPipelineDescriptor, ShaderModuleDescriptor, Surface,
    SurfaceConfiguration, SurfaceError, VertexAttribute, VertexBufferLayout, VertexState,
    VertexStepMode, TextureFormat,
};
use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

use std::{f64::consts::PI, mem::size_of, time::Instant};
use util::color_from_hsva;

pub mod texture;
pub mod util;

const CLEAR_COLOUR: Color = Color {
    r: 0.1,
    g: 0.2,
    b: 0.3,
    a: 1.0,
};

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
    diffuse_bind_group: wgpu::BindGroup,
    diffuse_texture: texture::Texture,
    num_indices: u32,

    // Surface challenge
    clear_colour: Color,
    changing_clear_colour: bool,
    cursor_position: Option<PhysicalPosition<f64>>,

    // Texture challenge
    freud_texture: texture::Texture,
    freud_bind_group: wgpu::BindGroup,
    freud: bool,

    // Egui
    egui_platform: Platform,
    egui_app: DemoWindows,
    egui_renderer: egui_wgpu::Renderer,
    start_time: Instant,
}

// Struct that represents the vertices of objects in our scene
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
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

const INDICES: &[u16] = &[0, 1, 4, 1, 2, 4, 2, 3, 4];

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

pub async fn run() {
    // To build a window with winit, we need to build an event loop first
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("太鼓の達人")
        .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT))
        .build(&event_loop)
        .expect("Couldn't build the window");

    let mut state = State::new(window).await;

    event_loop.run(move |event, _, control_flow| {
        state.egui_platform.handle_event(&event);
        match event {
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
        }
    });
}

impl State {
    async fn new(window: Window) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = Instance::new(InstanceDescriptor {
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

        let diffuse_bytes = include_bytes!("../assets/dababy-256.jpg");
        let diffuse_texture =
            texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "dababy").unwrap();

        let freud_bytes = include_bytes!("../assets/sigmundfreud3.png");
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

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader3.wgsl").into()),
        });

        let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },

            fragment: Some(FragmentState {
                module: &shader,
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
                cull_mode: Some(Face::Back),
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

        // Egui stuff
        let platform = Platform::new(PlatformDescriptor {
            physical_width: size.width,
            physical_height: size.height,
            scale_factor: window.scale_factor(),
            font_definitions: FontDefinitions::default(),
            style: egui::Style::default(),
        });

        let egui_renderer = egui_wgpu::Renderer::new(&device, TextureFormat::Bgra8UnormSrgb, None, 1);

        let demo = egui_demo_lib::DemoWindows::default();
        let start_time = Instant::now();

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

            num_indices: INDICES.len() as u32,
            clear_colour: CLEAR_COLOUR,
            changing_clear_colour: false,
            cursor_position: None,

            freud_texture,
            freud_bind_group,
            freud: false,

            egui_platform: platform,
            egui_app: demo,
            egui_renderer,
            start_time,
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
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        match *event {
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
                    _ => {},
                }

                true
            }

            _ => false,
        }
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
    }

    fn render(&mut self) -> Result<(), SurfaceError> {
        let output = self.surface.get_current_texture()?;

        let view = output.texture.create_view(&Default::default());

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: self.window.scale_factor() as f32,
        };

        self.egui_platform.update_time(self.start_time.elapsed().as_secs_f64());
        self.egui_platform.begin_frame();
        self.egui_app.ui(&self.egui_platform.context());

        let full_output = self.egui_platform.end_frame(Some(&self.window));
        let paint_jobs = self.egui_platform.context().tessellate(full_output.shapes);
        let textures_delta = full_output.textures_delta;

        for texture in textures_delta.free.iter() {
            self.egui_renderer.free_texture(texture);
        }

        for (id, image_delta) in textures_delta.set {
            self.egui_renderer.update_texture(&self.device, &self.queue, id, &image_delta);
        }

        self.egui_renderer.update_buffers(
            &self.device, 
            &self.queue, 
            &mut encoder, 
            &paint_jobs, 
            &screen_descriptor
        );
        
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
            depth_stencil_attachment: None,
        });

        render_pass.set_pipeline(&self.render_pipeline);
        let bind_group = if self.freud {
            &self.freud_bind_group
        } else {
            &self.diffuse_bind_group
        };

        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), IndexFormat::Uint16);
        render_pass.draw_indexed(0..self.num_indices, 0, 0..1);

        self.egui_renderer.render(&mut render_pass, &paint_jobs, &screen_descriptor);

        drop(render_pass);

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
