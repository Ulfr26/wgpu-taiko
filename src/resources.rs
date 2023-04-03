use std::path::Path;

use crate::model::{self, Model};
use crate::texture;
use anyhow::Result;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

pub fn load_texture(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<texture::Texture> {
    let data = std::fs::read(file_name)?;
    texture::Texture::from_bytes(device, queue, &data, file_name)
}

pub fn load_model(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> Result<Model> {
    let (models, obj_materials) = tobj::load_obj(
        file_name,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
    )?;

    let path = Path::new(file_name);

    let materials = obj_materials?
        .into_iter()
        .map(|m| {
            let mut texpath = path.parent().unwrap().clone().to_path_buf();
            texpath.push(&m.diffuse_texture);
            let texfilename = texpath.as_os_str().to_str().unwrap();

            let tex = load_texture(texfilename, device, queue)?;
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&tex.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&tex.sampler),
                    },
                ],
                label: None,
            });

            Ok(model::Material {
                name: m.name,
                diffuse_texture: tex,
                bind_group,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let meshes = models
        .into_iter()
        .map(|m| {
            let vertices = (0..m.mesh.positions.len() / 3)
                .map(|i| model::ModelVertex {
                    position: [
                        m.mesh.positions[i * 3],
                        m.mesh.positions[i * 3 + 1],
                        m.mesh.positions[i * 3 + 2],
                    ],
                    tex_coords: [m.mesh.texcoords[i * 2], 1.0 - m.mesh.texcoords[i * 2 + 1]],
                    normal: [
                        m.mesh.normals[i * 3],
                        m.mesh.normals[i * 3 + 1],
                        m.mesh.normals[i * 3 + 2],
                    ],
                })
                .collect::<Vec<_>>();

            let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
                label: Some(&format!("{file_name} Vertex Buffer")),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

            let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
                label: Some(&format!("{file_name} Index Buffer")),
                contents: bytemuck::cast_slice(&m.mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });

            model::Mesh {
                name: file_name.to_string(),
                vertex_buffer,
                index_buffer,
                num_elements: m.mesh.indices.len() as u32,
                material: m.mesh.material_id.unwrap_or(0),
            }
        })
        .collect::<Vec<_>>();

    println!("meshes: {}", meshes.len());
    println!("materials: {}", materials.len());
    Ok(model::Model { meshes, materials })
}
