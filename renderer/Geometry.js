export default class Geometry {
    static nextId = 0;
    constructor(device) {
        this.device = device;
        this.id = Geometry.nextId++;
        this.indexFormat = "uint32";
        this.indexCount = 0;
        
        this.verticesBuffer = null;
        this.normalsBuffer = null;
        this.uvsBuffer = null;
        this.colorsBuffer = null;
        this.indicesBuffer = null;
    }

    /**
     * Explicitly releases GPU memory. 
     * Essential for WebGPU to prevent OOM/MakeResident crashes.
     */
    destroy() {
        if (this.verticesBuffer) this.verticesBuffer.destroy();
        if (this.normalsBuffer) this.normalsBuffer.destroy();
        if (this.uvsBuffer) this.uvsBuffer.destroy();
        if (this.colorsBuffer) this.colorsBuffer.destroy();
        if (this.indicesBuffer) this.indicesBuffer.destroy();
        
        this.verticesBuffer = null;
        this.normalsBuffer = null;
        this.uvsBuffer = null;
        this.colorsBuffer = null;
        this.indicesBuffer = null;
    }

    fromObjMesh(mesh) {
        this.fromData(
            new Float32Array(mesh.vertices),
            new Float32Array(mesh.vertexNormals),
            new Float32Array(mesh.textures),
            new Uint32Array(mesh.indices)
        );
    }

    setBuffers(passEncoder) {
        if (!this.verticesBuffer || !this.indicesBuffer) return;

        passEncoder.setVertexBuffer(0, this.verticesBuffer);
        passEncoder.setVertexBuffer(1, this.normalsBuffer);
        passEncoder.setVertexBuffer(2, this.uvsBuffer);
        passEncoder.setVertexBuffer(3, this.colorsBuffer);
        passEncoder.setIndexBuffer(this.indicesBuffer, this.indexFormat);
    }

    draw(passEncoder, instanceCount = 1) {
        if (this.indexCount === 0) return;
        passEncoder.drawIndexed(this.indexCount, instanceCount);
    }

    fromThreeGeometry(threeGeo) {
        if (!threeGeo.attributes.normal) {
            threeGeo.computeVertexNormals();
        }

        const position = threeGeo.attributes.position;
        const count = position.count;
        
        let vertices;
        // Optimization: use original array if compatible (Float32, non-interleaved, 3 components)
        if (position.array instanceof Float32Array && !position.isInterleavedBufferAttribute && position.itemSize === 3) {
            vertices = position.array;
        } else {
            vertices = new Float32Array(count * 3);
            for (let i = 0; i < count; i++) {
                vertices[i * 3] = position.getX(i);
                vertices[i * 3 + 1] = position.getY(i);
                vertices[i * 3 + 2] = position.getZ(i);
            }
        }

        let normals;
        const nAttr = threeGeo.attributes.normal;
        if (nAttr && nAttr.array instanceof Float32Array && !nAttr.isInterleavedBufferAttribute && nAttr.itemSize === 3) {
            normals = nAttr.array;
        } else {
            normals = new Float32Array(count * 3);
            for (let i = 0; i < count; i++) {
                if (nAttr) {
                    normals[i * 3] = nAttr.getX(i);
                    normals[i * 3 + 1] = nAttr.getY(i);
                    normals[i * 3 + 2] = nAttr.getZ(i);
                } else {
                    normals[i * 3] = 0;
                    normals[i * 3 + 1] = 1;
                    normals[i * 3 + 2] = 0;
                }
            }
        }

        let uvs;
        const uAttr = threeGeo.attributes.uv;
        if (uAttr && uAttr.array instanceof Float32Array && !uAttr.isInterleavedBufferAttribute && uAttr.itemSize === 2) {
            uvs = uAttr.array;
        } else {
            uvs = new Float32Array(count * 2);
            for (let i = 0; i < count; i++) {
                if (uAttr) {
                    uvs[i * 2] = uAttr.getX(i);
                    uvs[i * 2 + 1] = uAttr.getY(i);
                }
            }
        }

        let colors;
        const cAttr = threeGeo.attributes.color;
        if (cAttr && cAttr.array instanceof Float32Array && !cAttr.isInterleavedBufferAttribute && cAttr.itemSize === 3) {
            colors = cAttr.array;
        } else {
            colors = new Float32Array(count * 3);
            if (cAttr) {
                for (let i = 0; i < count; i++) {
                    colors[i * 3] = cAttr.getX(i);
                    colors[i * 3 + 1] = cAttr.getY(i);
                    colors[i * 3 + 2] = cAttr.getZ(i);
                }
            } else {
                colors.fill(1.0);
            }
        }
        
        let indices;
        if (threeGeo.index) {
            indices = threeGeo.index.array;
        } else {
            indices = new Uint32Array(count);
            for (let i = 0; i < count; i++) indices[i] = i;
        }

        this.fromData(
            vertices,
            normals,
            uvs,
            indices instanceof Uint32Array || indices instanceof Uint16Array ? indices : new Uint32Array(indices),
            colors
        );
    }

    fromData(vertices, normals, uvs, indices, colors = null) {
        // Cleanup if this instance is being reused
        this.destroy();

        this.indexCount = indices.length;
        this.indexFormat = (indices instanceof Uint16Array) ? "uint16" : "uint32";

        // Bounding box
        this.min = [Infinity, Infinity, Infinity];
        this.max = [-Infinity, -Infinity, -Infinity];
        for (let i = 0; i < vertices.length; i += 3) {
            this.min[0] = Math.min(this.min[0], vertices[i]);
            this.min[1] = Math.min(this.min[1], vertices[i + 1]);
            this.min[2] = Math.min(this.min[2], vertices[i + 2]);
            this.max[0] = Math.max(this.max[0], vertices[i]);
            this.max[1] = Math.max(this.max[1], vertices[i + 1]);
            this.max[2] = Math.max(this.max[2], vertices[i + 2]);
        }

        const createBuffer = (data, usage) => {
            // Ensure data is properly typed and size is multiple of 4 bytes
            let uploadData = data;
            if (!(data instanceof ArrayBuffer) && data.buffer) {
                uploadData = data;
            }

            const byteLength = uploadData.byteLength || 0;
            if (byteLength === 0) {
                return this.device.createBuffer({
                    size: 4,
                    usage: usage | GPUBufferUsage.COPY_DST,
                });
            }

            let finalData = uploadData;
            const paddedByteLength = (byteLength + 3) & ~3;
            if (byteLength !== paddedByteLength) {
                const padded = new Uint8Array(paddedByteLength);
                const source = (uploadData.buffer) 
                    ? new Uint8Array(uploadData.buffer, uploadData.byteOffset, uploadData.byteLength)
                    : new Uint8Array(uploadData);
                padded.set(source);
                finalData = padded;
            }

            const buf = this.device.createBuffer({
                size: finalData.byteLength,
                usage: usage | GPUBufferUsage.COPY_DST,
            });

            this.device.queue.writeBuffer(buf, 0, finalData);
            return buf;
        };

        this.verticesBuffer = createBuffer(vertices || new Float32Array(0), GPUBufferUsage.VERTEX);
        this.normalsBuffer  = createBuffer(normals  || new Float32Array(0),  GPUBufferUsage.VERTEX);
        this.uvsBuffer      = createBuffer(uvs      || new Float32Array(0),      GPUBufferUsage.VERTEX);
        this.colorsBuffer   = createBuffer(colors   || new Float32Array(vertices ? vertices.length : 0).fill(1.0), GPUBufferUsage.VERTEX);
        this.indicesBuffer  = createBuffer(indices  || new Uint32Array(0),  GPUBufferUsage.INDEX);
    }

    static createPlane(device, size = 100) {
        const geo = new Geometry(device);
        const vertices = new Float32Array([
            -size, 0, -size,
             size, 0, -size,
             size, 0,  size,
            -size, 0,  size,
        ]);
        const normals = new Float32Array([
            0, 1, 0,
            0, 1, 0,
            0, 1, 0,
            0, 1, 0,
        ]);
        const uvs = new Float32Array([
            0, 0,
            1, 0,
            1, 1,
            0, 1,
        ]);
        const indices = new Uint32Array([
            0, 2, 1, 0, 3, 2,
        ]);
        geo.fromData(vertices, normals, uvs, indices);
        return geo;
    }

    static createBox(device) {
        const geo = new Geometry(device);
        const size = 1.0;
        const vertices = new Float32Array([
            // Front face
            -size, -size,  size,  size, -size,  size,  size,  size,  size, -size,  size,  size,
            // Back face
            -size, -size, -size, -size,  size, -size,  size,  size, -size,  size, -size, -size,
            // Top face
            -size,  size, -size, -size,  size,  size,  size,  size,  size,  size,  size, -size,
            // Bottom face
            -size, -size, -size,  size, -size, -size,  size, -size,  size, -size, -size,  size,
            // Right face
             size, -size, -size,  size,  size, -size,  size,  size,  size,  size, -size,  size,
            // Left face
            -size, -size, -size, -size, -size,  size, -size,  size,  size, -size,  size, -size,
        ]);

        const normals = new Float32Array([
            // Front
            0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
            // Back
            0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1,
            // Top
            0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
            // Bottom
            0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0,
            // Right
            1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
            // Left
            -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,
        ]);

        const uvs = new Float32Array([
            // Front
            0, 0, 1, 0, 1, 1, 0, 1,
            // Back
            1, 0, 1, 1, 0, 1, 0, 0,
            // Top
            0, 1, 0, 0, 1, 0, 1, 1,
            // Bottom
            1, 1, 0, 1, 0, 0, 1, 0,
            // Right
            1, 0, 1, 1, 0, 1, 0, 0,
            // Left
            0, 0, 1, 0, 1, 1, 0, 1,
        ]);

        const indices = new Uint32Array([
            0, 1, 2, 0, 2, 3,    // Front
            4, 5, 6, 4, 6, 7,    // Back
            8, 9, 10, 8, 10, 11, // Top
            12, 13, 14, 12, 14, 15, // Bottom
            16, 17, 18, 16, 18, 19, // Right
            20, 21, 22, 20, 22, 23  // Left
        ]);

        geo.fromData(vertices, normals, uvs, indices);
        return geo;
    }

    /* =========================
       ✅ REQUIRED FIX (ADDED)
       Vertex buffer layout
    ========================= */
    get vertexBuffers() {
        return [
            {
                arrayStride: 12,
                attributes: [
                    { shaderLocation: 0, offset: 0, format: "float32x3" }
                ],
            },
            {
                arrayStride: 12,
                attributes: [
                    { shaderLocation: 1, offset: 0, format: "float32x3" }
                ],
            },
            {
                arrayStride: 8,
                attributes: [
                    { shaderLocation: 2, offset: 0, format: "float32x2" }
                ],
            },
            {
                arrayStride: 12,
                attributes: [
                    { shaderLocation: 3, offset: 0, format: "float32x3" }
                ],
            },
        ];
    }
}
