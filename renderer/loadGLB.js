import * as THREE from 'https://unpkg.com/three@0.158.0/build/three.module.js';
import { GLTFLoader } from 'https://unpkg.com/three@0.158.0/examples/jsm/loaders/GLTFLoader.js';

export async function loadGLB(url) {
  const loader = new GLTFLoader();
  const gltf = await loader.loadAsync(url);

  const drawables = [];

  gltf.scene.updateWorldMatrix(true, true);

  gltf.scene.traverse((node) => {
    if (!node.isMesh) return;

    const geometry = node.geometry;
    const primitives = [];

    // POSITION (required)
    const posAttr = geometry.attributes.position;
    const norAttr = geometry.attributes.normal;

    const positions = new Float32Array(posAttr.array);
    const normals   = norAttr ? new Float32Array(norAttr.array) : null;

    // INDICES (required)
    const indices = geometry.index
      ? new Uint32Array(geometry.index.array)
      : null;

    primitives.push({
      positions,
      normals,
      indices,
      indexCount: indices.length
    });

    drawables.push({
      modelMatrix: node.matrixWorld.toArray(),
      primitives
    });
  });

  return drawables;
}
