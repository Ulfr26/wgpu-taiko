use wgpu::{Device, Queue, Surface, SurfaceConfiguration, Instance, InstanceDescriptor, Color, CommandEncoderDescriptor, RenderPassDescriptor, RenderPassColorAttachment, SurfaceError};
use winit::{dpi::PhysicalSize, window::{Window, WindowBuilder}, event::{WindowEvent, Event, KeyboardInput, ElementState, VirtualKeyCode}, event_loop::EventLoop};

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
}

pub async fn run() {
    // Start the error logger as winit uses this extensively
    env_logger::init();

    // To build a window with winit, we need to build an event loop first
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .build(&event_loop)
        .expect("Couldn't build the window");

    let mut state = State::new(window).await;

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            window_id,
            ref event,
        } if window_id == state.window().id() => if !state.input(event) {
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
                },

                WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    state.resize(**new_inner_size);
                },

                _ => {}
            }
        }

        Event::RedrawRequested(window_id) if window_id == state.window().id() => {
            state.update();

            match state.render() {
                Ok(_) => {},
                Err(SurfaceError::Lost) => state.resize(state.size),
                Err(SurfaceError::OutOfMemory) => control_flow.set_exit(),
                Err(e) => eprintln!("{e:?}"),
            }
        }

        Event::MainEventsCleared => {
            state.window().request_redraw()
        }

        _ => {}
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
        let surface = unsafe {
            instance.create_surface(&window)
        }.unwrap();

        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: Default::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            }
        ).await.unwrap();

        // Just curious
        println!("{:?}", adapter.get_info().backend);

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                // If we were targeting wasm32 we would use
                // wgpu::Limits::downlevel_webgl12_defaults()
                limits: Default::default(),
                label: None,

            },
            None,
        ).await.unwrap();

        let surface_caps = surface.get_capabilities(&adapter);

        // Shader code in this tutorial assumes an sRGB surface
        // texture. Using a different one will result in all the
        // colours coming out darker. If you want to support
        // non sRGB surfaces, you'll need to account for that
        // when drawing to the frame.
        let surface_format = surface_caps.formats.iter()
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

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size
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

    fn input(&mut self, _event: &WindowEvent) -> bool {
        // We don't have any events we want to capture,
        // for now.
        false
    }

    fn update(&mut self) {
        // We don't have anything to update yet
    }

    fn render(&mut self) -> Result<(), SurfaceError> {
        let output = self.surface.get_current_texture()?;

        let view = output.texture.create_view(&Default::default());

        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor { 
            label: Some("Render Encoder"), 
        });

        // Create the render pass. We need to drop this afterwards
        // so that we can call encoder.finish(), as encoder
        // is borrowed mutably here.
        let render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(CLEAR_COLOUR),
                    store: true,
                }
            })],
            depth_stencil_attachment: None,
        });

        drop(render_pass);

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}