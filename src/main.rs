use winit::{event_loop::{EventLoop, ControlFlow}, window::WindowBuilder, event::{Event, WindowEvent, KeyboardInput, ElementState, VirtualKeyCode}};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Start the error logger as winit uses this extensively
    env_logger::init();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .build(&event_loop)?;

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent { 
                window_id, 
                ref event 
            } if window_id == window.id() => match event {
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
                _ => {}
            },
            _ => {},
        }
    });
    
    Ok(())
}
