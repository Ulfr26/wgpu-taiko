use wgpu_taiko::run;

fn main() {
    // Start the error logger as winit uses this extensively
    env_logger::init();

    pollster::block_on(run())
}
