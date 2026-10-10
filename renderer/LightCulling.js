import { replaceArray } from './utils/stringReplaceArray.js';
import { vec3, vec4, mat4 } from 'gl-matrix';


const tmpVec3 = vec3.create();
const tmpVec4 = vec4.create();
const tmpMat4 = mat4.create();

/* =========================
    OPTIMIZED COMPUTE SHADER
========================= */
const computeShader = `
struct LightData {
  position : vec4<f32>,
  color : vec3<f32>,
  radius : f32,
  direction : vec3<f32>,
  coneAngle : f32, // cos(angle)
};

struct LightsBuffer {
  lights : array<LightData>,
};

@group(0) @binding(0)
var<storage, read_write> lightsBuffer : LightsBuffer;

struct TileLightIdData {
  count : atomic<u32>,
  lightId : array<u32, NUM_TILE_LIGHT_SLOT>,
};

struct Tiles {
  data : array<TileLightIdData>,
};

@group(1) @binding(0)
var<storage, read_write> tileLightId : Tiles;

struct Config {
  numLights : u32,
  numTiles : u32,
  tileCountX : u32,
  tileCountY : u32,
  numTileLightSlot : u32,
  tileSize : u32,
  padding1 : u32,
  padding2 : u32,
};

@group(2) @binding(0)
var<uniform> config : Config;

struct Uniforms {
  min : vec4<f32>,
  max : vec4<f32>,
  viewMatrix : mat4x4<f32>,
  projectionMatrix : mat4x4<f32>,
  invProjectionMatrix : mat4x4<f32>,
  fullScreenSize : vec4<f32>,
};

@group(3) @binding(0)
var<uniform> uniforms : Uniforms;

fn get_view_pos(screen_pos : vec2<f32>) -> vec3<f32> {
  let clip_pos = vec4<f32>(
    screen_pos.x * 2.0 / uniforms.fullScreenSize.x - 1.0,
    1.0 - screen_pos.y * 2.0 / uniforms.fullScreenSize.y,
    0.0,
    1.0
  );
  let view_pos_h = uniforms.invProjectionMatrix * clip_pos;
  return view_pos_h.xyz / view_pos_h.w;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let tileX = gid.x;
  let tileY = gid.y;

  if (tileX >= config.tileCountX || tileY >= config.tileCountY) {
    return;
  }

  let tileId = tileX + tileY * config.tileCountX;
  let TILE_SIZE = f32(config.tileSize);
  
  // Get view space corners of the tile frustum
  let p0 = get_view_pos(vec2<f32>(f32(tileX) * TILE_SIZE, f32(tileY) * TILE_SIZE));
  let p1 = get_view_pos(vec2<f32>(f32(tileX + 1u) * TILE_SIZE, f32(tileY) * TILE_SIZE));
  let p2 = get_view_pos(vec2<f32>(f32(tileX) * TILE_SIZE, f32(tileY + 1u) * TILE_SIZE));
  let p3 = get_view_pos(vec2<f32>(f32(tileX + 1u) * TILE_SIZE, f32(tileY + 1u) * TILE_SIZE));

  var planes : array<vec4<f32>, 4>;
  planes[0] = vec4<f32>(normalize(cross(p2, p0)), 0.0); // Left
  planes[1] = vec4<f32>(normalize(cross(p1, p3)), 0.0); // Right
  planes[2] = vec4<f32>(normalize(cross(p0, p1)), 0.0); // Top
  planes[3] = vec4<f32>(normalize(cross(p3, p2)), 0.0); // Bottom

  for (var i : u32 = 0u; i < config.numLights; i++) {
    let light = lightsBuffer.lights[i];
    if (light.radius <= 0.0) { continue; }

    let lightPos = (uniforms.viewMatrix * light.position).xyz;
    let r = light.radius;

    var inFrustum = true;
    for (var p = 0u; p < 4u; p++) {
      if (dot(lightPos, planes[p].xyz) < -r) {
        inFrustum = false;
        break;
      }
    }

    if (inFrustum) {
      let off = atomicAdd(&tileLightId.data[tileId].count, 1u);
      if (off < config.numTileLightSlot) {
        tileLightId.data[tileId].lightId[off] = i;
      }
    }
  }
}
`;

export default class LightCulling {
  constructor(device) {
    this.device = device;
  }

  async init(canvas, camera) {
    this.camera = camera;
    this.lightsActive = false;

    /* SAFE LIMITS */
    this.numLights = 5000;
    this.tileSize = 16;
    this.tileLightSlot = 256;

    this.extentMin = vec3.fromValues(-20, -1, -20);
    this.extentMax = vec3.fromValues(20, 20, 20);

    const extent = vec3.create();
    vec3.sub(extent, this.extentMax, this.extentMin);

    this.lightDataStride = 12; // 4 (pos) + 4 (color+radius) + 4 (dir+cone)
    this.lightDataGPUBuffer = this.device.createBuffer({
      size: this.numLights * this.lightDataStride * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const lightData = new Float32Array(this.numLights * this.lightDataStride);
    for (let i = 0; i < this.numLights; i++) {
      const o = i * this.lightDataStride;
      lightData[o]     = (Math.random() - 0.5) * extent[0] * 2.0;
      lightData[o + 1] = (Math.random() - 0.5) * extent[1] * 2.0;
      lightData[o + 2] = (Math.random() - 0.5) * extent[2] * 2.0;
      lightData[o + 3] = 1.0;
      
      const hue = Math.random();
      const rgb = this.hsvToRgb(hue, 0.8, 1.0);
      lightData[o + 4] = rgb[0];
      lightData[o + 5] = rgb[1];
      lightData[o + 6] = rgb[2];
      lightData[o + 7] = 2.0 + Math.random() * 5.0;

      // Random direction for omni lights (won't affect them if cone is 0)
      lightData[o + 8] = 0; 
      lightData[o + 9] = -1;
      lightData[o + 10] = 0;
      lightData[o + 11] = 0.0; // 0 means Omni
    }
    this.device.queue.writeBuffer(this.lightDataGPUBuffer, 0, lightData);

    this.tileCount = [
      Math.ceil(canvas.width / this.tileSize),
      Math.ceil(canvas.height / this.tileSize)
    ];
    this.numTiles = this.tileCount[0] * this.tileCount[1];

    this.tileLightIdGPUBuffer = this.device.createBuffer({
      size: this.numTiles * (1 + this.tileLightSlot) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.tileLightIdZeroData =
      new Uint32Array(this.numTiles * (1 + this.tileLightSlot));

    this.configBuffer = this.device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.uniformsBuffer = this.device.createBuffer({
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.lightCullingComputePipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: this.device.createShaderModule({
          code: replaceArray(
            computeShader,
            ['NUM_TILE_LIGHT_SLOT', 'NUM_TILES'],
            [this.tileLightSlot, this.numTiles]
          )
        }),
        entryPoint: 'main',
      },
    });

    this.setupBindGroups();
  }

  regenerateLights() {
    const extent = vec3.create();
    vec3.sub(extent, this.extentMax, this.extentMin);

    const lightData = new Float32Array(this.numLights * this.lightDataStride);
    for (let i = 0; i < this.numLights; i++) {
      const o = i * this.lightDataStride;
      
      // Position relative to center point of bounds
      const center = vec3.create();
      vec3.add(center, this.extentMin, this.extentMax);
      vec3.scale(center, center, 0.5);

      lightData[o]     = center[0] + (Math.random() - 0.5) * extent[0];
      lightData[o + 1] = center[1] + (Math.random() - 0.5) * extent[1];
      lightData[o + 2] = center[2] + (Math.random() - 0.5) * extent[2];
      lightData[o + 3] = 1.0;
      
      const hue = Math.random();
      const sat = 0.8;
      const val = 1.0 + Math.random() * 2.0;
      const rgb = this.hsvToRgb(hue, sat, val);
      
      lightData[o + 4] = rgb[0];
      lightData[o + 5] = rgb[1];
      lightData[o + 6] = rgb[2];
      
      // Scale radius based on model size
      const sceneScale = vec3.length(extent);
      lightData[o + 7] = (0.01 + Math.random() * 0.05) * sceneScale; 
    }
    this.device.queue.writeBuffer(this.lightDataGPUBuffer, 0, lightData);
  }

  hsvToRgb(h, s, v) {
    let r, g, b;
    const i = Math.floor(h * 6);
    const f = h * 6 - i;
    const p = v * (1 - s);
    const q = v * (1 - f * s);
    const t = v * (1 - (1 - f) * s);
    switch (i % 6) {
      case 0: r = v; g = t; b = p; break;
      case 1: r = q; g = v; b = p; break;
      case 2: r = p; g = v; b = t; break;
      case 3: r = p; g = q; b = v; break;
      case 4: r = t; g = p; b = v; break;
      case 5: r = v; g = p; b = q; break;
    }
    return [r, g, b];
  }

  setupBindGroups() {
    this.bindGroup0 = this.device.createBindGroup({
      layout: this.lightCullingComputePipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.lightDataGPUBuffer } }]
    });
    this.bindGroup1 = this.device.createBindGroup({
      layout: this.lightCullingComputePipeline.getBindGroupLayout(1),
      entries: [{ binding: 0, resource: { buffer: this.tileLightIdGPUBuffer } }]
    });
    this.bindGroup2 = this.device.createBindGroup({
      layout: this.lightCullingComputePipeline.getBindGroupLayout(2),
      entries: [{ binding: 0, resource: { buffer: this.configBuffer } }]
    });
    this.bindGroup3 = this.device.createBindGroup({
      layout: this.lightCullingComputePipeline.getBindGroupLayout(3),
      entries: [{ binding: 0, resource: { buffer: this.uniformsBuffer } }]
    });
  }

  updateBounds(min, max) {
    vec3.copy(this.extentMin, min);
    vec3.copy(this.extentMax, max);
    // Don't regenerate random lights if manual lights are expected
  }

  setManualLights(lights) {
    const lightData = new Float32Array(this.numLights * this.lightDataStride);
    this.manualLightCount = Math.min(lights.length, this.numLights);
    
    for (let i = 0; i < this.numLights; i++) {
        const o = i * this.lightDataStride;
        if (i < this.manualLightCount) {
            const l = lights[i];
            lightData[o]     = l.position[0];
            lightData[o + 1] = l.position[1];
            lightData[o + 2] = l.position[2];
            lightData[o + 3] = 1.0;
            
            lightData[o + 4] = l.color[0];
            lightData[o + 5] = l.color[1];
            lightData[o + 6] = l.color[2];
            lightData[o + 7] = l.radius;

            if (l.direction) {
                lightData[o + 8] = l.direction[0];
                lightData[o + 9] = l.direction[1];
                lightData[o + 10] = l.direction[2];
                lightData[o + 11] = Math.cos(l.angle || 0.7); // Default cone
            } else {
                lightData[o + 8] = 0;
                lightData[o + 9] = 0;
                lightData[o + 10] = 0;
                lightData[o + 11] = 0.0; // Omni
            }
        } else {
            lightData[o + 7] = 0.0;
        }
    }
    this.device.queue.writeBuffer(this.lightDataGPUBuffer, 0, lightData);
  }

  toggleLights(active) {
    this.lightsActive = active;
    // We can handle this by zeroing out radius or intensity in a quick buffer write
    // For now, DeferredRenderer will call setManualLights with appropriate values
  }

  update(externalCommandEncoder = null) {
    this.device.queue.writeBuffer(
      this.tileLightIdGPUBuffer,
      0,
      this.tileLightIdZeroData
    );

    this.device.queue.writeBuffer(
      this.configBuffer,
      0,
      new Uint32Array([
        this.numLights,
        this.numTiles,
        this.tileCount[0],
        this.tileCount[1],
        this.tileLightSlot,
        this.tileSize,
        0,
        0
      ])
    );

    const u = new Float32Array(80); // Increased size for invProjection
    u.set([...this.extentMin, 0], 0);
    u.set([...this.extentMax, 0], 4);
    u.set(this.camera.viewMatrix, 8);
    u.set(this.camera.projectionMatrix, 24);
    
    const invProj = mat4.create();
    mat4.invert(invProj, this.camera.projectionMatrix);
    u.set(invProj, 40);
    
    u.set([this.camera.canvas.width, this.camera.canvas.height, 0, 0], 56);
    this.device.queue.writeBuffer(this.uniformsBuffer, 0, u);

    const encoder = externalCommandEncoder || this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.lightCullingComputePipeline);
    pass.setBindGroup(0, this.bindGroup0);
    pass.setBindGroup(1, this.bindGroup1);
    pass.setBindGroup(2, this.bindGroup2);
    pass.setBindGroup(3, this.bindGroup3);

    pass.dispatchWorkgroups(
      Math.ceil(this.tileCount[0] / 16),
      Math.ceil(this.tileCount[1] / 16)
    );
    pass.end();

    if (!externalCommandEncoder) {
      this.device.queue.submit([encoder.finish()]);
    }
  }
}
