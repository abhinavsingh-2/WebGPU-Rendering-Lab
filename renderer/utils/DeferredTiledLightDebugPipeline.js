import Camera from './Camera.js';
import WriteGBufferMaterial from './WriteGBufferMaterial.js';
import LightCulling from './LightCulling.js';
import Geometry from './Geometry.js';
import Drawable from './Drawable.js';

const tmpMat4 = new Float32Array(16);

const vertexShaderFullScreenQuad = `
struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) fragUV : vec2<f32>,
};
@vertex
fn main(@location(0) position : vec4<f32>, @location(1) uv : vec2<f32>) -> VertexOutput {
    var output : VertexOutput;
    output.position = position;
    output.fragUV = uv;
    return output;
}
`;

export default class DeferredRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.drawableLists = [];
        this.camera = new Camera(canvas);
        this.renderFuncs = {
            'debugView': this.renderGBufferDebugView.bind(this),
            'deferredTiledLightDebug': this.renderDeferredTiledLightDebug.bind(this),
        };
        this.renderMode = 'debugView';
        this.curRenderModeFunc = this.renderFuncs[this.renderMode];
    }

    async init() {
        const adapter = await navigator.gpu.requestAdapter();
        const device = await adapter.requestDevice({
            requiredLimits: { maxColorAttachmentBytesPerSample: 128 }
        });
        this.device = device;
        const context = this.context = this.canvas.getContext('webgpu');
        context.configure({
            device,
            format: navigator.gpu.getPreferredCanvasFormat(),
            alphaMode: 'opaque'
        });

        WriteGBufferMaterial.setup(device);
        this.uniformBuffer = device.createBuffer({
            size: 128,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.lightCulling = new LightCulling(device);
        await this.lightCulling.init(this.canvas, this.camera);

        this.setupGBuffer();
        this.setupQuadPipeline();
    }

    setupGBuffer() {
        this.depthTexture = this.device.createTexture({
            size: [this.canvas.width, this.canvas.height],
            format: "depth24plus-stencil8",
            usage: GPUTextureUsage.RENDER_ATTACHMENT
        });

        this.gbufferTextures = [
            this.device.createTexture({ size: [this.canvas.width, this.canvas.height], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING }),
            this.device.createTexture({ size: [this.canvas.width, this.canvas.height], format: "rgba32float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING }),
            this.device.createTexture({ size: [this.canvas.width, this.canvas.height], format: navigator.gpu.getPreferredCanvasFormat(), usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING }),
        ];

        this.renderPassDescriptor = {
            colorAttachments: this.gbufferTextures.map(tex => ({
                view: tex.createView(),
                clearValue: [0, 0, 0, 1],
                loadOp: 'clear',
                storeOp: 'store'
            })),
            depthStencilAttachment: {
                view: this.depthTexture.createView(),
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            }
        };
    }

    setupQuadPipeline() {
        this.quadVerticesBuffer = this.device.createBuffer({
            size: 6 * 4 * 6,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        const fullScreenQuadArray = new Float32Array([-1,-1,0.5,1,0,0, 1,-1,0.5,1,1,0, 1,1,0.5,1,1,1, -1,-1,0.5,1,0,0, 1,1,0.5,1,1,1, -1,1,0.5,1,0,1]);
        this.device.queue.writeBuffer(this.quadVerticesBuffer, 0, fullScreenQuadArray);

        // 1. Setup GBuffer Debug Pipeline
        this.setupGBufferDebugViewPipeline();
        
        // 2. Setup Tiled Light Debug Pipeline (THE FIX IS HERE)
        this.setupDeferredTiledLightDebugPipeline();
    }

    setupGBufferDebugViewPipeline() {
        const shader = `
        @group(0) @binding(0) var pos : texture_2d<f32>;
        @group(0) @binding(1) var norm : texture_2d<f32>;
        @group(0) @binding(2) var alb : texture_2d<f32>;
        @fragment
        fn main(@builtin(position) coord : vec4<f32>, @location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
            if (uv.x < 0.33) { return textureLoad(pos, vec2<i32>(coord.xy), 0); }
            if (uv.x < 0.66) { return vec4(textureLoad(norm, vec2<i32>(coord.xy), 0).xyz * 0.5 + 0.5, 1.0); }
            return textureLoad(alb, vec2<i32>(coord.xy), 0);
        }`;

        this.gbufferDebugViewPipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: this.device.createShaderModule({ code: vertexShaderFullScreenQuad }),
                entryPoint: "main",
                buffers: [{ arrayStride: 24, attributes: [{shaderLocation:0, offset:0, format:"float32x4"}, {shaderLocation:1, offset:16, format:"float32x2"}]}]
            },
            fragment: {
                module: this.device.createShaderModule({ code: shader }),
                entryPoint: "main",
                targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
            },
            primitive: { topology: 'triangle-list' },
        });

        this.quadBindGroup = this.device.createBindGroup({
            layout: this.gbufferDebugViewPipeline.getBindGroupLayout(0),
            entries: this.gbufferTextures.map((t, i) => ({ binding: i, resource: t.createView() }))
        });
    }

    setupDeferredTiledLightDebugPipeline() {
        // We inject the values DIRECTLY into the string. This makes "replaceArray" obsolete.
        const slot = this.lightCulling.tileLightSlot;
        const tiles = this.lightCulling.numTiles;
        const tx = this.lightCulling.tileCount[0];
        const ts = this.lightCulling.tileSize;

        const shaderCode = `
        struct TileData { count: atomic<u32>, ids: array<u32, ${slot}> };
        struct Tiles { data: array<TileData, ${tiles}> };
        @group(0) @binding(0) var<storage, read_write> tileLightId : Tiles;

        @fragment
        fn main(@builtin(position) coord : vec4<f32>) -> @location(0) vec4<f32> {
            let tileCoord = vec2<u32>(coord.xy) / ${ts}u;
            let tileId = tileCoord.x + tileCoord.y * ${tx}u;
            let c = atomicLoad(&tileLightId.data[tileId].count);
            let t = f32(c) / f32(${slot});
            return vec4<f32>(4.0 * t - 2.0, select(4.0 * t, 4.0 - 4.0 * t, t >= 0.5), 2.0 - 4.0 * t, 1.0);
        }`;

        this.deferredTiledLightDebugPipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: this.device.createShaderModule({ code: vertexShaderFullScreenQuad }),
                entryPoint: "main",
                buffers: [{ arrayStride: 24, attributes: [{shaderLocation:0, offset:0, format:"float32x4"}, {shaderLocation:1, offset:16, format:"float32x2"}]}]
            },
            fragment: {
                module: this.device.createShaderModule({ code: shaderCode }),
                entryPoint: "main",
                targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
            },
            primitive: { topology: 'triangle-list' },
        });

        this.tileBindGroup = this.device.createBindGroup({
            layout: this.deferredTiledLightDebugPipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: this.lightCulling.tileLightIdGPUBuffer } }],
        });
    }

    renderGBufferDebugView(commandEncoder) {
        const pass = commandEncoder.beginRenderPass({
            colorAttachments: [{ view: this.context.getCurrentTexture().createView(), clearValue: [0,0,0,1], loadOp: 'clear', storeOp: 'store' }]
        });
        pass.setPipeline(this.gbufferDebugViewPipeline);
        pass.setVertexBuffer(0, this.quadVerticesBuffer);
        pass.setBindGroup(0, this.quadBindGroup);
        pass.draw(6);
        pass.end();
    }

    renderDeferredTiledLightDebug(commandEncoder) {
        this.lightCulling.update();
        const pass = commandEncoder.beginRenderPass({
            colorAttachments: [{ view: this.context.getCurrentTexture().createView(), clearValue: [0,0,0,1], loadOp: 'clear', storeOp: 'store' }]
        });
        pass.setPipeline(this.deferredTiledLightDebugPipeline);
        pass.setVertexBuffer(0, this.quadVerticesBuffer);
        pass.setBindGroup(0, this.tileBindGroup);
        pass.draw(6);
        pass.end();
    }

    frame() {
        const commandEncoder = this.device.createCommandEncoder();
        // 1. GBuffer Pass
        const pass = commandEncoder.beginRenderPass(this.renderPassDescriptor);
        // ... drawing logic ...
        pass.end();

        // 2. Post process / Debug Pass
        this.curRenderModeFunc(commandEncoder);
        this.device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(() => this.frame());
    }
}