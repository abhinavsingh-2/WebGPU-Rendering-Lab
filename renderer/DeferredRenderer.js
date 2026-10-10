import { mat4, vec3 } from 'gl-matrix';
import Camera from './Camera.js';
import WriteGBufferMaterial from './WriteGBufferMaterial.js';
import Geometry from './Geometry.js';
import Drawable from './Drawable.js';
import OBJ from 'webgl-obj-loader';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader.js';
import Transform from './Transform.js';
import LightCulling from './LightCulling.js';

export default class DeferredRenderer {
    constructor(device, canvas) {
        this.device = device;
        this.canvas = canvas;
        this.context = canvas.getContext('webgpu');
        this.format = navigator.gpu.getPreferredCanvasFormat();

        this.context.configure({
            device: this.device,
            format: this.format,
            alphaMode: 'opaque'
        });

        this.camera = new Camera(canvas);
        this.drawables = [];
        this.geometryCache = new Map();
        this.textureCache = new Map();
        this.materialBindGroupCache = new Map();
        this.sceneCenter = vec3.create();
        this.hasSceneCenter = false;

        this.lightCulling = new LightCulling(this.device);

        // Buffers for global and per-object data
        this.frameUniformBuffer = null;
        this.modelMatrixStorageBuffer = null;
        this.frameBindGroup = null;
        this.modelBindGroup = null;
        
        this.maxInstances = 100000; // Reduced to 100k instances for better compatibility
        this.instanceData = new Float32Array(this.maxInstances * 24); 
        this.instanceDataCentered = new Float32Array(this.maxInstances * 24);
        this.drawablesSorted = []; 
        this.totalInstances = 0;
        this.needsBufferUpdate = true;

        this.renderPassDescriptor = {
            colorAttachments: [{
                view: undefined,
                resolveTarget: undefined,
                clearValue: { r: 0.01, g: 0.01, b: 0.03, a: 1.0 }, // Deep navy studio
                loadOp: 'clear',
                storeOp: 'store'
            }],
            depthStencilAttachment: {
                view: undefined,
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            }
        };

        this.initialized = false;
        this.groundEnabled = true; 
        
        this.studioLightColor = [1.0, 1.2, 2.0]; 
        this.studioLightRadius = 25.0;
    }

    async init() {
        if (this.initialized) return;

        // Ensure canvas size is set with devicePixelRatio for "4K" like sharpness
        const pixelRatio = window.devicePixelRatio || 1;
        this.canvas.width = this.canvas.clientWidth * pixelRatio;
        this.canvas.height = this.canvas.clientHeight * pixelRatio;

        // Create buffers
        this.frameUniformBuffer = this.device.createBuffer({
            size: 256, // view (64) + projection (64) + cameraPos (16) + sceneCenter (16) + padding
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        const maxStorageSize = this.device.limits.maxStorageBufferBindingSize;
        const requestedSize = this.instanceData.byteLength;
        const actualBufferSize = Math.min(requestedSize, maxStorageSize);

        this.modelMatrixStorageBuffer = this.device.createBuffer({
            size: actualBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });

        // Setup G-Buffer material (static part)
        WriteGBufferMaterial.setup(this.device);

        // Create Default Sampler
        this.defaultSampler = this.device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
            mipmapFilter: 'linear',
            addressModeU: 'repeat',
            addressModeV: 'repeat',
        });

        // Create Default White Texture (1x1)
        this.defaultWhiteTexture = this.device.createTexture({
            size: [1, 1],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        this.device.queue.writeTexture(
            { texture: this.defaultWhiteTexture },
            new Uint8Array([255, 255, 255, 255]),
            { bytesPerRow: 4 },
            { width: 1, height: 1 }
        );
        this.defaultWhiteTextureView = this.defaultWhiteTexture.createView();

        // Create Depth Texture
        this.depthTexture = this.device.createTexture({
            size: [this.canvas.width, this.canvas.height],
            format: 'depth24plus',
            sampleCount: 4,
            usage: GPUTextureUsage.RENDER_ATTACHMENT
        });

        this.msaaColorTexture = this.device.createTexture({
            size: [this.canvas.width, this.canvas.height],
            format: navigator.gpu.getPreferredCanvasFormat(),
            sampleCount: 4,
            usage: GPUTextureUsage.RENDER_ATTACHMENT
        });

        this.renderPassDescriptor.colorAttachments[0].view = this.msaaColorTexture.createView();
        this.renderPassDescriptor.depthStencilAttachment.view = this.depthTexture.createView();

        await this.lightCulling.init(this.canvas, this.camera);

        const shaderModule = this.device.createShaderModule({
            code: `
                struct FrameUniforms {
                    viewMatrix : mat4x4<f32>,
                    projectionMatrix : mat4x4<f32>,
                    cameraPos : vec3<f32>,
                    sceneCenter : vec3<f32>,
                };
                @group(0) @binding(0) var<uniform> frame : FrameUniforms;
                
                struct InstanceData {
                    modelMatrix : mat4x4<f32>,
                    color : vec4<f32>,
                    extra : vec4<f32>, // x: glow power
                };
                @group(1) @binding(0) var<storage, read> instances : array<InstanceData>;

                struct LightData {
                    position : vec4<f32>,
                    color : vec3<f32>,
                    radius : f32,
                    direction : vec3<f32>,
                    coneAngle : f32,
                };

                struct LightsBuffer {
                    lights : array<LightData>,
                };
                @group(3) @binding(0) var<storage, read> lightsBuffer : LightsBuffer;

                struct TileLightIdData {
                    count : u32,
                    lightId : array<u32, ${this.lightCulling.tileLightSlot}>,
                };
                struct Tiles {
                    data : array<TileLightIdData>,
                };
                @group(3) @binding(1) var<storage, read> tileLightId : Tiles;

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
                @group(3) @binding(2) var<uniform> config : Config;

                struct VertexOutput {
                    @builtin(position) position : vec4<f32>,
                    @location(0) worldPos : vec3<f32>,
                    @location(1) normal : vec3<f32>,
                    @location(2) color : vec3<f32>,
                    @location(3) uv : vec2<f32>,
                    @location(4) glow : f32,
                };

                @vertex
                fn vs_main(
                    @location(0) pos : vec3<f32>, 
                    @location(1) norm : vec3<f32>,
                    @location(2) uv : vec2<f32>,
                    @location(3) vertColor : vec3<f32>,
                    @builtin(instance_index) instanceIdx : u32
                ) -> VertexOutput {
                    var out : VertexOutput;
                    let instance = instances[instanceIdx];
                    
                    let worldPos = instance.modelMatrix * vec4<f32>(pos, 1.0);
                    out.position = frame.projectionMatrix * frame.viewMatrix * worldPos;
                    out.worldPos = worldPos.xyz;
                    out.normal = (instance.modelMatrix * vec4<f32>(norm, 0.0)).xyz;
                    out.color = vertColor * instance.color.rgb;
                    out.uv = uv;
                    out.glow = instance.extra.x;
                    return out;
                }

                @group(2) @binding(0) var baseColorSampler : sampler;
                @group(2) @binding(1) var baseColorTexture : texture_2d<f32>;

                fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
                    let a = 2.51;
                    let b = 0.03;
                    let c = 2.43;
                    let d = 0.59;
                    let e = 0.14;
                    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
                }

                fn saturate_color(c: vec3<f32>, adjustment: f32) -> vec3<f32> {
                    let luminance = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
                    return mix(vec3<f32>(luminance), c, adjustment);
                }

                @fragment
                fn fs_main(
                    @builtin(position) fragCoord : vec4<f32>,
                    @location(0) worldPos : vec3<f32>, 
                    @location(1) normal : vec3<f32>,
                    @location(2) color : vec3<f32>,
                    @location(3) uv : vec2<f32>,
                    @location(4) glow : f32
                ) -> @location(0) vec4<f32> {
                    let texColor = textureSample(baseColorTexture, baseColorSampler, uv);
                    let baseColor = color * texColor.rgb;
                    
                    var N = normalize(normal);
                    if (length(N) < 0.1) {
                        N = vec3(0.0, 1.0, 0.0);
                    }
                    
                    let V = normalize(frame.cameraPos - worldPos);
                    
                    // --- TILED ACCELERATED LIGHTING ---
                    let tileX = u32(fragCoord.x / f32(config.tileSize));
                    let tileY = u32(fragCoord.y / f32(config.tileSize));
                    let tileId = tileX + tileY * config.tileCountX;
                    
                    var totalDiffuse = vec3(0.0);
                    var totalSpecular = vec3(0.0);
                    
                    let lightCount = min(tileLightId.data[tileId].count, config.numTileLightSlot);
                    for (var i = 0u; i < lightCount; i++) {
                        let lightIdx = tileLightId.data[tileId].lightId[i];
                        let light = lightsBuffer.lights[lightIdx];
                        
                        let lightPos = light.position.xyz; 
                        let Lvec = lightPos - worldPos;
                        let dist = length(Lvec);
                        
                        if (dist < light.radius) {
                            let direction = Lvec / dist;
                            
                            // Physically inspired quadratic falloff with smoothing window
                            let distanceSquared = dist * dist;
                            let attenuation = 1.0 / (max(distanceSquared, 1.0)); // Prevent infinity at origin
                            
                            let ratio = dist / light.radius;
                            let window = saturate(1.0 - ratio * ratio * ratio * ratio);
                            var falloff = attenuation * window * window;
                            
                            // Normalize intensity so it's impactful at studio distances
                            falloff *= 40.0; 
                            
                            // Spot light check
                            if (light.coneAngle > 0.0) {
                                let spotDir = normalize(light.direction);
                                let cosTheta = dot(direction, -spotDir);
                                if (cosTheta < light.coneAngle) {
                                    falloff = 0.0;
                                } else {
                                    // Soft edge interpolation
                                    let epsilon = 0.05;
                                    falloff *= saturate((cosTheta - light.coneAngle) / epsilon);
                                }
                            }
                            
                            // Diffuse (Lambert)
                            let nDotL = max(dot(N, direction), 0.0);
                            totalDiffuse += light.color * nDotL * falloff * 8.0; 
                            
                            // Specular (Blinn-Phong with Fresnel)
                            let H = normalize(direction + V);
                            let nDotH = max(dot(N, H), 0.0);
                            totalSpecular += light.color * pow(nDotH, 128.0) * falloff * 4.0;
                        }
                    }

                    // --- GLOBAL BASE LIGHTING (Reduced for Studio Effect) ---
                    // Key Light (Neutral-Warm Rim/Fill)
                    // 1. Key Light (Strong, warm-white from top-front-right)
                    let L1 = normalize(vec3(0.6, 0.8, 0.5)); 
                    let keyColor = vec3(1.0, 0.96, 0.9) * 1.0; 
                    let nDotL1 = max(dot(N, L1), 0.0);
                    
                    // 2. Fill Light (Soft, cool-blue from top-back-left to lift shadows)
                    let L2 = normalize(vec3(-0.6, 0.4, -0.5));
                    let fillColor = vec3(0.7, 0.8, 1.0) * 0.35;
                    let nDotL2 = max(dot(N, L2), 0.0);
                    
                    // Specular from Key Light (Blinn-Phong)
                    let H1 = normalize(L1 + V);
                    let spec = pow(max(dot(N, H1), 0.0), 32.0) * 0.25;
                    let specColor = vec3(1.0, 1.0, 1.0);

                    // Composite direct lighting
                    var direct = baseColor * (nDotL1 * keyColor + nDotL2 * fillColor) + (specColor * spec) + (baseColor * totalDiffuse);

                    // Ambient Lighting (Very subtle deep navy fill)
                    let ambientLighting = vec3(0.12, 0.13, 0.15);

                    // Fresnel / Backlight Rim Effect
                    let fresnel = 0.04 + (1.0 - 0.04) * pow(1.0 - saturate(dot(V, N)), 5.0);
                    let backlight = pow(1.0 - saturate(dot(V, N)), 4.0) * 0.15;
                    
                    // Composite
                    var specular = totalSpecular + (backlight * vec3(0.9, 0.95, 1.0));
                    var ambient = baseColor * ambientLighting;
                    
                    var finalColor = direct + ambient + (specular * fresnel);
                    
                    // Apply Emissive Glow
                    finalColor += baseColor * glow * 10.0;

                    // --- POST-PROCESS ---
                    // Slight exposure boost to ensure highlights pop against the dark environment
                    finalColor = aces_tonemap(finalColor * 1.2); 
                    finalColor = saturate_color(finalColor, 1.05); 
                    finalColor = pow(finalColor, vec3(1.0 / 2.2)); // Gamma correction

                    return vec4<f32>(finalColor, 1.0);
                }
            `
        });

        // Bind Group Layouts
        const frameBindGroupLayout = this.device.createBindGroupLayout({
            entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }]
        });

        const modelBindGroupLayout = this.device.createBindGroupLayout({
            entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }]
        });

        this.materialBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } }
            ]
        });

        const lightBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
            ]
        });

        this.lightBindGroup = this.device.createBindGroup({
            layout: lightBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.lightCulling.lightDataGPUBuffer } },
                { binding: 1, resource: { buffer: this.lightCulling.tileLightIdGPUBuffer } },
                { binding: 2, resource: { buffer: this.lightCulling.configBuffer } }
            ]
        });

        this.frameBindGroup = this.device.createBindGroup({
            layout: frameBindGroupLayout,
            entries: [{ binding: 0, resource: { buffer: this.frameUniformBuffer } }]
        });

        this.modelBindGroup = this.device.createBindGroup({
            layout: modelBindGroupLayout,
            entries: [{ binding: 0, resource: { buffer: this.modelMatrixStorageBuffer } }]
        });

        this.pipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [frameBindGroupLayout, modelBindGroupLayout, this.materialBindGroupLayout, lightBindGroupLayout]
            }),
            vertex: {
                module: shaderModule,
                entryPoint: 'vs_main',
                buffers: [
                    {
                        arrayStride: 12, // pos
                        attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }]
                    },
                    {
                        arrayStride: 12, // norm
                        attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' }]
                    },
                    {
                        arrayStride: 8, // uv
                        attributes: [{ shaderLocation: 2, offset: 0, format: 'float32x2' }]
                    },
                    {
                        arrayStride: 12, // color
                        attributes: [{ shaderLocation: 3, offset: 0, format: 'float32x3' }]
                    }
                ]
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fs_main',
                targets: [{ format: this.format }]
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'none'
            },
            multisample: {
                count: 4,
            },
            depthStencil: {
                format: 'depth24plus',
                depthWriteEnabled: true,
                depthCompare: 'less'
            }
        });

        // Initialize Ground Plane
        if (this.groundEnabled) {
            const groundGeo = Geometry.createPlane(this.device, 1000);
            this.geometryCache.set("ground", groundGeo);
            this.ground = new Drawable(this.device, groundGeo, null, null);
            this.ground.transform = new Transform();
            this.ground.color = [0.05, 0.05, 0.08, 1.0]; // Dark slate color
            this.addDrawable(this.ground);
        }

        this.initialized = true;
    }

    async loadOBJ(url) {
        const response = await fetch(url);
        const text = await response.text();
        const mesh = new OBJ.Mesh(text);

        const geo = new Geometry(this.device);
        geo.fromObjMesh(mesh);

        const drawable = new Drawable(this.device, geo, null, null);
        drawable.transform = new Transform();
        drawable.instanceOffset = this.drawables.length;
        drawable.name = mesh.name || "mesh";
        this.needsBufferUpdate = true;
        
        this.addDrawable(drawable);
        return drawable;
    }

    clearScene() {
        for (const geo of this.geometryCache.values()) {
            if (this.ground && geo === this.ground.geometry) continue;
            geo.destroy();
        }
        this.geometryCache.clear();
        if (this.ground) {
            this.geometryCache.set("ground", this.ground.geometry);
            this.drawables = [this.ground];
        } else {
            this.drawables = [];
        }
        this.drawablesSorted = [];
        this.hasSceneCenter = false;
        vec3.set(this.sceneCenter, 0, 0, 0);
    }

    getOrCreateGeometry(threeGeo) {
        const uuid = threeGeo.uuid;
        if (this.geometryCache.has(uuid)) {
            return this.geometryCache.get(uuid);
        }
        const geo = new Geometry(this.device);
        geo.fromThreeGeometry(threeGeo);
        this.geometryCache.set(uuid, geo);
        return geo;
    }

    async _getGPUTexture(threeTexture) {
        if (!threeTexture) return this.defaultWhiteTextureView;
        if (this.textureCache.has(threeTexture.uuid)) {
            return this.textureCache.get(threeTexture.uuid);
        }

        let image = threeTexture.image;
        if (!image) return this.defaultWhiteTextureView;

        // Ensure HTMLImageElement is loaded
        if (image instanceof HTMLImageElement && !image.complete) {
            await new Promise((resolve, reject) => {
                image.onload = resolve;
                image.onerror = reject;
            });
        }

        let source;
        try {
            if (image instanceof ImageBitmap) {
                source = image;
            } else if (image instanceof HTMLImageElement || image instanceof HTMLCanvasElement || image instanceof OffscreenCanvas) {
                source = await createImageBitmap(image);
            } else {
                return this.defaultWhiteTextureView;
            }

            const texture = this.device.createTexture({
                size: [source.width, source.height],
                format: 'rgba8unorm',
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
            });

            this.device.queue.copyExternalImageToTexture(
                { source: source },
                { texture: texture },
                [source.width, source.height]
            );

            const view = texture.createView();
            this.textureCache.set(threeTexture.uuid, view);
            return view;
        } catch (e) {
            console.warn("Failed to load texture:", e);
            return this.defaultWhiteTextureView;
        }
    }

    _getMaterialBindGroup(textureView) {
        const view = textureView || this.defaultWhiteTextureView;
        if (this.materialBindGroupCache.has(view)) {
            return this.materialBindGroupCache.get(view);
        }

        const bindGroup = this.device.createBindGroup({
            layout: this.materialBindGroupLayout,
            entries: [
                { binding: 0, resource: this.defaultSampler },
                { binding: 1, resource: view }
            ]
        });

        this.materialBindGroupCache.set(view, bindGroup);
        return bindGroup;
    }

    _getMaterialColor(material) {
        if (!material) return [1.0, 1.0, 1.0, 1.0];
        
        // Handle array of materials (pick primary or average? first is usually safest)
        if (Array.isArray(material)) {
            return this._getMaterialColor(material[0]);
        }

        const color = [1.0, 1.0, 1.0, 1.0];
        
        // 1. Try standard color (Three.js materials)
        if (material.color) {
            color[0] = material.color.r;
            color[1] = material.color.g;
            color[2] = material.color.b;
        } 
        // 2. Try emissive as fallback if color is black (common in some CAD exports)
        else if (material.emissive && (material.emissive.r > 0 || material.emissive.g > 0 || material.emissive.b > 0)) {
            color[0] = material.emissive.r;
            color[1] = material.emissive.g;
            color[2] = material.emissive.b;
        }

        // 3. Handle opacity
        if (material.opacity !== undefined) {
            color[3] = material.opacity;
        }

        return color;
    }

    async loadGLB(url, onProgress) {
        const dracoLoader = new DRACOLoader();
        dracoLoader.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.7/');
        
        const loader = new GLTFLoader();
        loader.setDRACOLoader(dracoLoader);

        const gltf = await new Promise((resolve, reject) => {
            loader.load(url, resolve, (xhr) => {
                if (onProgress) onProgress(xhr.loaded / xhr.total * 0.5); // First 50% is download
            }, reject);
        });
        
        const meshes = [];
        gltf.scene.traverse(node => {
            if (node.isMesh) meshes.push(node);
        });

        // Chunk processing to keep UI responsive
        const CHUNK_SIZE = 100;
        for (let i = 0; i < meshes.length; i += CHUNK_SIZE) {
            const end = Math.min(i + CHUNK_SIZE, meshes.length);
            for (let j = i; j < end; j++) {
                const meshNode = meshes[j];
                const geo = this.getOrCreateGeometry(meshNode.geometry);
                const materialColor = this._getMaterialColor(meshNode.material);
                const textureView = await this._getGPUTexture(meshNode.material ? meshNode.material.map : null);

                if (meshNode.isInstancedMesh) {
                    const drawable = new Drawable(this.device, geo, null, null);
                    drawable.transform = new Transform();
                    drawable.color = materialColor;
                    drawable.textureView = textureView;
                    drawable.instanceOffset = this.drawables.length;
                    drawable.instanceCount = meshNode.count;
                    drawable.name = meshNode.name || "instanced_mesh";
                    drawable.instanceMatrices = new Float32Array(meshNode.count * 16);
                    
                    for (let k = 0; k < meshNode.count; k++) {
                        const m = mat4.create();
                        meshNode.getMatrixAt(k, m);
                        const worldM = mat4.create();
                        mat4.multiply(worldM, meshNode.matrixWorld.elements, m);
                        drawable.instanceMatrices.set(worldM, k * 16);
                    }
                    this.addDrawable(drawable);
                } else {
                    const drawable = new Drawable(this.device, geo, null, null);
                    drawable.transform = new Transform();
                    drawable.color = materialColor;
                    drawable.textureView = textureView;
                    drawable.name = meshNode.name || "mesh";
                    meshNode.updateWorldMatrix(true, true);
                    mat4.copy(drawable.transform.modelMatrix, meshNode.matrixWorld.elements);
                    drawable.transform.dirty = false;
                    this.addDrawable(drawable);
                }
            }
            if (onProgress) onProgress(0.5 + (i / meshes.length) * 0.5);
            await new Promise(r => setTimeout(r, 0)); // Yield to main thread
        }

        dracoLoader.dispose();
    }

    async loadFBX(url, onProgress) {
        const loader = new FBXLoader();
        const fbx = await new Promise((resolve, reject) => {
            loader.load(url, resolve, (xhr) => {
                if (onProgress) onProgress(xhr.loaded / xhr.total * 0.5);
            }, reject);
        });
        
        const meshes = [];
        fbx.traverse(node => {
            if (node.isMesh) meshes.push(node);
        });

        const CHUNK_SIZE = 100;
        for (let i = 0; i < meshes.length; i += CHUNK_SIZE) {
            const end = Math.min(i + CHUNK_SIZE, meshes.length);
            for (let j = i; j < end; j++) {
                const meshNode = meshes[j];
                const geo = this.getOrCreateGeometry(meshNode.geometry);
                const materialColor = this._getMaterialColor(meshNode.material);
                const textureView = await this._getGPUTexture(meshNode.material ? meshNode.material.map : null);

                if (meshNode.isInstancedMesh) {
                    const drawable = new Drawable(this.device, geo, null, null);
                    drawable.transform = new Transform();
                    drawable.color = materialColor;
                    drawable.textureView = textureView;
                    drawable.instanceCount = meshNode.count;
                    drawable.name = meshNode.name || "instanced_mesh";
                    drawable.instanceMatrices = new Float32Array(meshNode.count * 16);
                    
                    for (let k = 0; k < meshNode.count; k++) {
                        const m = mat4.create();
                        meshNode.getMatrixAt(k, m);
                        const worldM = mat4.create();
                        mat4.multiply(worldM, meshNode.matrixWorld.elements, m);
                        drawable.instanceMatrices.set(worldM, k * 16);
                    }
                    this.addDrawable(drawable);
                } else {
                    const drawable = new Drawable(this.device, geo, null, null);
                    drawable.transform = new Transform();
                    drawable.color = materialColor;
                    drawable.textureView = textureView;
                    drawable.name = meshNode.name || "mesh";
                    meshNode.updateWorldMatrix(true, true);
                    mat4.copy(drawable.transform.modelMatrix, meshNode.matrixWorld.elements);
                    drawable.transform.dirty = false;
                    this.addDrawable(drawable);
                }
            }
            if (onProgress) onProgress(0.5 + (i / meshes.length) * 0.5);
            await new Promise(r => setTimeout(r, 0));
        }
    }

    addDrawable(drawable) {
        this.drawables.push(drawable);
        this.needsBufferUpdate = true;
    }


    recenterCamera() {
        if (this.drawables.length === 0) return;

        let min = [Infinity, Infinity, Infinity];
        let max = [-Infinity, -Infinity, -Infinity];

        for (const drawable of this.drawables) {
            if (this.ground && drawable === this.ground) continue;
            const geo = drawable.geometry;
            if (geo.min && geo.max) {
                const addPoints = (mat) => {
                    const corners = [
                        [geo.min[0], geo.min[1], geo.min[2]],
                        [geo.max[0], geo.min[1], geo.min[2]],
                        [geo.min[0], geo.max[1], geo.min[2]],
                        [geo.max[0], geo.max[1], geo.min[2]],
                        [geo.min[0], geo.min[1], geo.max[2]],
                        [geo.max[0], geo.min[1], geo.max[2]],
                        [geo.min[0], geo.max[1], geo.max[2]],
                        [geo.max[0], geo.max[1], geo.max[2]],
                    ];
                    for (const corner of corners) {
                        const worldPos = vec3.create();
                        vec3.transformMat4(worldPos, corner, mat);
                        min[0] = Math.min(min[0], worldPos[0]);
                        min[1] = Math.min(min[1], worldPos[1]);
                        min[2] = Math.min(min[2], worldPos[2]);
                        max[0] = Math.max(max[0], worldPos[0]);
                        max[1] = Math.max(max[1], worldPos[1]);
                        max[2] = Math.max(max[2], worldPos[2]);
                    }
                };

                if (drawable.instanceCount > 1 && drawable.instanceMatrices) {
                    for (let i = 0; i < drawable.instanceCount; i++) {
                        const mat = drawable.instanceMatrices.subarray(i * 16, i * 16 + 16);
                        addPoints(mat);
                    }
                } else {
                    const modelMatrix = drawable.transform ? drawable.transform.getModelMatrix() : mat4.create();
                    addPoints(modelMatrix);
                }
            }
        }

        if (min[0] === Infinity) {
            this.camera.eye = vec3.fromValues(20, 20, 20);
            this.camera.target = vec3.fromValues(0, 0, 0);
        } else {
            const center = vec3.fromValues(
                (min[0] + max[0]) / 2,
                (min[1] + max[1]) / 2,
                (min[2] + max[2]) / 2
            );

            vec3.copy(this.sceneCenter, center);
            this.hasSceneCenter = true;
            this.needsBufferUpdate = true;

            const size = vec3.fromValues(
                max[0] - min[0],
                max[1] - min[1],
                max[2] - min[2]
            );
            const dist = vec3.length(size) || 10;

            this.camera.target = [0, 0, 0];
            this.camera.eye = [dist * 1.2, dist * 0.8, dist * 1.2];
            this.camera.radius = vec3.length(this.camera.eye);

            if (this.ground && this.ground.transform) {
                this.ground.transform.setTranslation(vec3.fromValues(0, min[1], 0));
            }

            // Update light culling bounds relative to scene center
            const localMin = vec3.create();
            const localMax = vec3.create();
            vec3.sub(localMin, min, center);
            vec3.sub(localMax, max, center);
            // Expand bounds slightly for lights
            vec3.scale(localMin, localMin, 1.5);
            vec3.scale(localMax, localMax, 1.5);
            this.lightCulling.updateBounds(localMin, localMax);
            
            // Ensure lights are OFF initially
            this.toggleGlow(false);
        }
        
        if (this.camera.updateOrbit) this.camera.updateOrbit();
        this.camera.updateView();
    }

    toggleGlow(active) {
        const lights = [];
        const sceneOffset = vec3.create();
        vec3.negate(sceneOffset, this.sceneCenter);
        
        for (const drawable of this.drawables) {
            if (drawable.name && drawable.name.toLowerCase().includes('cl')) {
                drawable.glow = active ? 1.0 : 0.0;
                
                if (active) {
                    const addLight = (mat) => {
                        const pos = vec3.fromValues(mat[12], mat[13], mat[14]);
                        vec3.add(pos, pos, sceneOffset);
                        
                        // Extract direction from matrix (assume pointing down -Y or similar)
                        const dir = vec3.fromValues(mat[4], mat[5], mat[6]); // Y-axis of local space
                        vec3.negate(dir, dir);
                        
                        lights.push({
                            position: pos,
                            color: this.studioLightColor,
                            radius: this.studioLightRadius,
                            direction: dir,
                            angle: 0.8 // Concentrated cone
                        });
                    };

                    if (drawable.instanceCount > 1 && drawable.instanceMatrices) {
                        for (let i = 0; i < drawable.instanceCount; i++) {
                            const mat = drawable.instanceMatrices.subarray(i * 16, i * 16 + 16);
                            addLight(mat);
                        }
                    } else {
                        const modelMat = drawable.transform ? drawable.transform.getModelMatrix() : mat4.create();
                        addLight(modelMat);
                    }
                }
            }
        }

        if (active && lights.length > 0) {
            this.lightCulling.setManualLights(lights);
        } else {
            this.lightCulling.setManualLights([]);
        }

        this.needsBufferUpdate = true;
    }

    resize() {
        const pixelRatio = window.devicePixelRatio || 1;
        const width = Math.floor(this.canvas.clientWidth * pixelRatio);
        const height = Math.floor(this.canvas.clientHeight * pixelRatio);

        if (this.canvas.width !== width || this.canvas.height !== height) {
            this.canvas.width = width;
            this.canvas.height = height;

            if (this.depthTexture) this.depthTexture.destroy();
            if (this.msaaColorTexture) this.msaaColorTexture.destroy();

            this.depthTexture = this.device.createTexture({
                size: [width, height],
                format: 'depth24plus',
                sampleCount: 4,
                usage: GPUTextureUsage.RENDER_ATTACHMENT
            });

            this.msaaColorTexture = this.device.createTexture({
                size: [width, height],
                format: this.format, // Use the stored format
                sampleCount: 4,
                usage: GPUTextureUsage.RENDER_ATTACHMENT
            });

            this.renderPassDescriptor.colorAttachments[0].view = this.msaaColorTexture.createView();
            this.renderPassDescriptor.depthStencilAttachment.view = this.depthTexture.createView();
            
            this.camera.aspect = width / height;
            this.camera.updateProjection();
        }
    }

    render() {
        if (!this.initialized) return;
        this.resize();

        // 1. Prepare instance data (ONLY if something changed)
        if (this.needsBufferUpdate) {
            const sceneOffset = vec3.create();
            vec3.negate(sceneOffset, this.sceneCenter);
            
            // Sort drawables by geometry and texture for better batching and cache locality
            this.drawables.sort((a, b) => {
                if (a.geometry !== b.geometry) {
                    return a.geometry.id > b.geometry.id ? 1 : -1;
                }
                if (a.textureView !== b.textureView) {
                    return (a.textureView ? 1 : 0) - (b.textureView ? 1 : 0);
                }
                return 0;
            });

            let instanceCountTotal = 0;
            const maxStorageSize = this.device.limits.maxStorageBufferBindingSize;
            const maxInstancesPossible = Math.floor(maxStorageSize / 96); // 96 bytes per entry (16*4 + 4*4 + 4*4)
            const actualMax = Math.min(this.maxInstances, maxInstancesPossible);

            for (const drawable of this.drawables) {
                if (instanceCountTotal >= actualMax) break;
                
                drawable.instanceOffset = instanceCountTotal;
                const dColor = drawable.color || [1, 1, 1, 1];
                const dExtra = [drawable.glow || 0, 0, 0, 0];
                
                if (drawable.instanceCount > 1 && drawable.instanceMatrices) {
                    const count = Math.min(drawable.instanceCount, actualMax - instanceCountTotal);
                    for (let i = 0; i < count; i++) {
                        const instMat = drawable.instanceMatrices.subarray(i * 16, i * 16 + 16);
                        const mat = mat4.create();
                        mat4.translate(mat, mat, sceneOffset);
                        mat4.multiply(mat, mat, instMat);
                        
                        this.instanceDataCentered.set(mat, instanceCountTotal * 24);
                        this.instanceDataCentered.set(dColor, instanceCountTotal * 24 + 16);
                        this.instanceDataCentered.set(dExtra, instanceCountTotal * 24 + 20);
                        instanceCountTotal++;
                    }
                    drawable.renderedInstanceCount = count;
                } else {
                    const modelMatrix = drawable.transform ? drawable.transform.getModelMatrix() : mat4.create();
                    const mat = mat4.create();
                    mat4.translate(mat, mat, sceneOffset);
                    mat4.multiply(mat, mat, modelMatrix);
                    
                    this.instanceDataCentered.set(mat, instanceCountTotal * 24);
                    this.instanceDataCentered.set(dColor, instanceCountTotal * 24 + 16);
                    this.instanceDataCentered.set(dExtra, instanceCountTotal * 24 + 20);
                    instanceCountTotal++;
                    drawable.renderedInstanceCount = 1;
                }
            }
            this.totalInstances = instanceCountTotal;

            if (this.totalInstances > 0) {
                this.device.queue.writeBuffer(
                    this.modelMatrixStorageBuffer, 
                    0, 
                    this.instanceDataCentered.buffer, 
                    0, 
                    this.totalInstances * 96
                );
            }
            this.needsBufferUpdate = false;
        }

        // 2. Prepare Frame Uniforms
        this.camera.updateView();
        const projMatrix = this.camera.projectionMatrix;
        const viewMatrix = this.camera.viewMatrix;
        const cameraPos = new Float32Array(this.camera.eye);

        const frameData = new Float32Array(64);
        frameData.set(viewMatrix, 0);
        frameData.set(projMatrix, 16);
        frameData.set(cameraPos, 32);
        frameData.set(this.sceneCenter, 36);
        this.device.queue.writeBuffer(this.frameUniformBuffer, 0, frameData);

        // 3. Render Pass
        const commandEncoder = this.device.createCommandEncoder();
        
        // Update Light Culling (Compute Pass)
        this.lightCulling.update(commandEncoder);

        const texture = this.context.getCurrentTexture();
        if (!texture) return;
        const textureView = texture.createView();
        
        this.renderPassDescriptor.colorAttachments[0].view = this.msaaColorTexture.createView();
        this.renderPassDescriptor.colorAttachments[0].resolveTarget = textureView;
        this.renderPassDescriptor.depthStencilAttachment.view = this.depthTexture.createView();

        const passEncoder = commandEncoder.beginRenderPass(this.renderPassDescriptor);
        passEncoder.setPipeline(this.pipeline);
        passEncoder.setBindGroup(0, this.frameBindGroup);
        passEncoder.setBindGroup(1, this.modelBindGroup);
        passEncoder.setBindGroup(3, this.lightBindGroup);

        // Group by geometry and texture to batched draw calls
        let currentGeo = null;
        let currentTextureView = null;
        let batchInstanceCount = 0;
        let batchFirstInstance = 0;
        const defaultView = this.defaultWhiteTextureView;

        for (const drawable of this.drawables) {
            if (drawable.instanceOffset >= this.totalInstances) break;
            
            const geo = drawable.geometry;
            const texView = drawable.textureView || defaultView;

            if (geo !== currentGeo || texView !== currentTextureView) {
                // Submit previous batch
                if (currentGeo && batchInstanceCount > 0) {
                    passEncoder.drawIndexed(currentGeo.indexCount, batchInstanceCount, 0, 0, batchFirstInstance);
                }
                
                // Start new batch
                if (geo !== currentGeo) {
                    currentGeo = geo;
                    geo.setBuffers(passEncoder);
                }
                if (texView !== currentTextureView) {
                    currentTextureView = texView;
                    const materialBG = this._getMaterialBindGroup(texView);
                    passEncoder.setBindGroup(2, materialBG);
                }

                batchFirstInstance = drawable.instanceOffset;
                batchInstanceCount = drawable.renderedInstanceCount;
            } else {
                // Continue batch (since they are sorted by geometry and texture)
                batchInstanceCount += drawable.renderedInstanceCount;
            }
        }

        // Final batch
        if (currentGeo && batchInstanceCount > 0) {
            passEncoder.drawIndexed(currentGeo.indexCount, batchInstanceCount, 0, 0, batchFirstInstance);
        }

        passEncoder.end();
        this.device.queue.submit([commandEncoder.finish()]);
    }
}

