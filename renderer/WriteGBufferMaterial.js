const vertexShaderBlinnPhong = `
struct Uniforms {
    modelViewMatrix : mat4x4<f32>,
    projectionMatrix : mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms : Uniforms;

struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) fragPosition : vec3<f32>,
    @location(1) fragNormal   : vec3<f32>,
    @location(2) fragUV       : vec2<f32>,
};

@vertex
fn main(
    @location(0) position : vec3<f32>,
    @location(1) normal   : vec3<f32>,
    @location(2) uv       : vec2<f32>
) -> VertexOutput {
    var output : VertexOutput;

    let posVS = uniforms.modelViewMatrix * vec4<f32>(position, 1.0);

    output.position     = uniforms.projectionMatrix * posVS;
    output.fragPosition = posVS.xyz;
    output.fragNormal   = normalize(
        (uniforms.modelViewMatrix * vec4<f32>(normal, 0.0)).xyz
    );
    output.fragUV       = uv;

    return output;
}
`;

const fragmentShaderGBuffer = `
struct GBufferOutput {
    @location(0) position : vec4<f32>,
    @location(1) normal   : vec4<f32>,
    @location(2) albedo   : vec4<f32>,
};

@group(0) @binding(1) var mySampler : sampler;
@group(0) @binding(2) var albedoMap : texture_2d<f32>;

@fragment
fn main(
    @location(0) fragPosition : vec3<f32>,
    @location(1) fragNormal   : vec3<f32>,
    @location(2) fragUV       : vec2<f32>
) -> GBufferOutput {
    var output : GBufferOutput;

    let texColor = textureSample(albedoMap, mySampler, fragUV);

    output.position = vec4<f32>(fragPosition, 1.0);
    output.normal   = vec4<f32>(normalize(fragNormal), 1.0);
    output.albedo   = texColor;

    return output;
}
`;

export default class WriteGBufferMaterial {
    static setup(device) {
        this.whiteTexture = device.createTexture({
            size: [1, 1, 1],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });

        device.queue.writeTexture(
            { texture: this.whiteTexture },
            new Uint8Array([255, 255, 255, 255]),
            { bytesPerRow: 4 },
            [1, 1, 1]
        );

        this.uniformsBindGroupLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: { type: 'uniform' }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    sampler: { type: 'filtering' }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: 'float' }
                }
            ]
        });

        this.vertexShader   = vertexShaderBlinnPhong;
        this.fragmentShader = fragmentShaderGBuffer;
    }

    constructor(sampler, albedoMap, uniformBuffer) {
        this.uniformsBindGroupLayout =
            WriteGBufferMaterial.uniformsBindGroupLayout;

        const finalAlbedo =
            albedoMap || WriteGBufferMaterial.whiteTexture;

        this.bindings = [
            {
                binding: 0,
                resource: { buffer: uniformBuffer },
            },
            {
                binding: 1,
                resource: sampler,
            },
            {
                binding: 2,
                resource: finalAlbedo.createView(),
            }
        ];
    }
}
