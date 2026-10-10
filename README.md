# WebGPU 3D Engine

A browser-based 3D model viewer and rendering engine built with React, TypeScript, Vite, and WebGPU. It supports loading GLB, GLTF, OBJ, and FBX assets.

## Requirements

- Node.js (LTS recommended)
- A browser with WebGPU support for the rendering pipeline

## Run locally

```bash
npm install
npm run dev
```

Open the local URL printed by Vite.

## Production build

```bash
npm run build
npm run preview
```

## Notes

- Use the file picker or drag a supported 3D model into the page.
- The renderer and model-loading implementation are contained in the existing source files.
