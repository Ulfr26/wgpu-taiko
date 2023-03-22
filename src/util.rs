use std::f64::consts::PI;
use wgpu::Color;

pub fn color_from_hsva(h: f64, s: f64, v: f64, a: f64) -> Option<Color> {
    let chroma = v * s;
    let hprime = h / (PI / 3.0);
    let x = chroma * (1.0 - (hprime % 2.0 - 1.0).abs());
    let (r, g, b) = if 0.0 <= hprime && hprime < 1.0 {
        (chroma, x, 0.0)
    } else if 1.0 <= hprime && hprime < 2.0 {
        (x, chroma, 0.0)
    } else if 2.0 <= hprime && hprime < 3.0 {
        (0.0, chroma, x)
    } else if 3.0 <= hprime && hprime < 4.0 {
        (0.0, x, chroma)
    } else if 4.0 <= hprime && hprime < 5.0 {
        (x, 0.0, chroma)
    } else if 5.0 <= hprime && hprime < 6.0 {
        (chroma, 0.0, x)
    } else {
        return None;
    };

    let m = v - chroma;

    Some(Color {
        r: r + m,
        g: g + m,
        b: b + m,
        a,
    })
}
