export default class Drawable {
  constructor(device, geometry, material) {
    this.device = device;
    this.geometry = geometry;
    this.material = material;

    this.transform = null;
    this.instanceOffset = 0;
    this.instanceCount = 1;
    this.instanceMatrices = null; // For THREE.InstancedMesh
    this.color = [1, 1, 1, 1];
    this.glow = 0.0;
    this.name = "";
    this.textureView = null;
  }

  createBindGroup(pipeline) {
    // No longer needed as we use global bind groups in DeferredRenderer
  }
}
