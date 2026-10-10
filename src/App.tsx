/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useRef, useState, ChangeEvent, DragEvent } from "react";
import Camera from "../renderer/Camera.js";
import DeferredRenderer from "../renderer/DeferredRenderer.js";
import Geometry from "../renderer/Geometry.js";
import Drawable from "../renderer/Drawable.js";
import WriteGBufferMaterial from "../renderer/WriteGBufferMaterial.js";
import Transform from "../renderer/Transform.js";

// @ts-ignore
const GPU_BUFFER_USAGE_UNIFORM = 0x0040;
// @ts-ignore
const GPU_BUFFER_USAGE_COPY_DST = 0x0008;

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [renderer, setRenderer] = useState<DeferredRenderer | null>(null);
  const [loading, setLoading] = useState<{ active: boolean, progress: number, status: string }>({ active: false, progress: 0, status: '' });
  const [hasLoadedModel, setHasLoadedModel] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    async function initWebGPU() {
      if (!navigator.gpu) {
        setError("WebGPU not supported in this browser. Please use Chrome or another WebGPU-enabled browser.");
        return;
      }

      if (canvasRef.current) {
        try {
          const adapter = await navigator.gpu.requestAdapter();
          if (!adapter) {
            setError("No GPU adapter found.");
            return;
          }
          const device = await adapter.requestDevice();

          const dr = new DeferredRenderer(device, canvasRef.current);
          await dr.init();

          setRenderer(dr);

          let animationFrameId: number;
          const frame = () => {
            dr.render();
            animationFrameId = requestAnimationFrame(frame);
          }
          animationFrameId = requestAnimationFrame(frame);
          
          return () => cancelAnimationFrame(animationFrameId);
        } catch (e: any) {
          setError(`Init failed: ${e.message}`);
        }
      }
    }

    initWebGPU();
  }, []);

  const handleDragOver = (e: DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = async (e: DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      await loadModelFile(file);
    }
  };

  const handleFileUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      await loadModelFile(file);
    }
  };

  const loadModelFile = async (file: File) => {
    if (!renderer) return;
    const url = URL.createObjectURL(file);
    const extension = file.name.split('.').pop()?.toLowerCase();

    setLoading({ active: true, progress: 0, status: 'Loading file...' });
    try {
      renderer.clearScene();
      const onProgress = (p: number) => {
        setLoading(l => ({ ...l, progress: p, status: p === 1 ? 'Parsing 3D Data...' : `Loading: ${Math.round(p * 100)}%` }));
      };

      if (extension === 'obj') {
        await renderer.loadOBJ(url);
      } else if (extension === 'glb' || extension === 'gltf') {
        await renderer.loadGLB(url, onProgress);
      } else if (extension === 'fbx') {
        await renderer.loadFBX(url, onProgress);
      } else {
        alert("Unsupported file format. Please use .obj, .glb, or .fbx");
        return;
      }
      renderer.recenterCamera();
      setHasLoadedModel(true);
    } catch (e) {
      console.error("Failed to load model", e);
      alert(`Failed to load model: ${e instanceof Error ? e.message : 'Unknown error'}`);
    } finally {
      setLoading({ active: false, progress: 0, status: '' });
    }
  };

  return (
    <div 
      className="relative min-h-screen w-full overflow-hidden bg-[#f4f7fc] font-sans text-slate-900 select-none"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Neutral workspace background */}
      <div className="fixed inset-0 bg-[#f4f7fc] pointer-events-none" />

      {/* Main Canvas Container - Full Screen */}
      <div className="fixed inset-0 z-0 overflow-hidden">
        {error ? (
          <div className="absolute inset-0 flex items-center justify-center p-4">
            <div className="bg-red-500/10 border border-red-500/50 text-red-400 px-8 py-6 rounded-3xl text-sm font-bold shadow-2xl backdrop-blur-xl animate-in fade-in zoom-in duration-500">
              <div className="flex items-center gap-3">
                <div className="w-2 h-2 rounded-full bg-red-500 animate-ping" />
                {error}
              </div>
            </div>
          </div>
        ) : (
          <div className="relative w-full h-full">
            {loading.active && (
              <div className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-slate-950/80 backdrop-blur-2xl transition-all duration-700">
                <div className="relative w-72 h-1 bg-white/5 rounded-full overflow-hidden mb-6">
                  <div 
                    className="absolute inset-y-0 left-0 bg-gradient-to-r from-blue-600 to-emerald-400 shadow-[0_0_15px_rgba(59,130,246,0.5)] transition-all duration-700 ease-out"
                    style={{ width: `${Math.max(5, loading.progress * 100)}%` }}
                  />
                </div>
                <div className="text-[10px] font-black uppercase tracking-[0.4em] text-blue-400/80 drop-shadow-2xl">
                  {loading.status}
                </div>
              </div>
            )}
            <canvas 
              id="webgpu-canvas"
              ref={canvasRef} 
              className={`w-full h-full block ${hasLoadedModel ? "opacity-100 cursor-grab active:cursor-grabbing" : "opacity-0 pointer-events-none"}`}
              style={{ transition: "opacity 180ms ease" }}
            />
          </div>
        )}
      </div>

      {/* Drag Over Full Screen Glassmorphic Indication Overlay */}
      {isDragging && (
        <div className="fixed inset-0 z-40 m-5 rounded-2xl border-2 border-dashed border-blue-500 bg-blue-50/90 pointer-events-none flex flex-col items-center justify-center gap-4">
          <div className="w-14 h-14 rounded-full bg-white border border-blue-200 flex items-center justify-center">
            <svg className="w-8 h-8 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
            </svg>
          </div>
          <span className="text-sm font-semibold text-blue-700">Drop 3D model here to render</span>
        </div>
      )}

      {/* Initial state: simple, professional upload panel */}
      {!hasLoadedModel && !loading.active && (
        <div className="fixed inset-0 z-20 flex items-center justify-center p-5 pointer-events-none">
          <div className={`w-full max-w-[560px] rounded-[24px] border border-slate-200 bg-white p-8 sm:p-9 text-center shadow-[0_18px_50px_rgba(15,23,42,0.08)] flex flex-col items-center gap-5 pointer-events-auto ${isDragging ? "border-blue-500 bg-blue-50" : ""}`}>
            <label className="flex h-20 w-20 cursor-pointer items-center justify-center rounded-full bg-blue-50 text-blue-600 transition-colors hover:bg-blue-100">
              <svg className="h-9 w-9" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.7} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
              </svg>
              <input type="file" className="hidden" accept=".obj,.glb,.gltf,.fbx" onChange={handleFileUpload} />
            </label>

            <div className="space-y-2">
              <h1 className="text-2xl font-semibold tracking-tight text-slate-900">Upload 3D Asset</h1>
              <p className="mx-auto max-w-md text-sm leading-6 text-slate-500">
                Drag and drop a 3D model here, or select a file to get started.
              </p>
            </div>

            <label className={`flex min-h-[128px] w-full cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border border-dashed px-5 py-6 transition-colors ${isDragging ? "border-blue-500 bg-blue-50" : "border-slate-300 bg-slate-50 hover:border-blue-400 hover:bg-blue-50/50"}`}>
              <svg className="h-8 w-8 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.7} d="M12 16V4m0 0L7 9m5-5l5 5M4 16.5v1A2.5 2.5 0 006.5 20h11a2.5 2.5 0 002.5-2.5v-1" />
              </svg>
              <span className="text-sm font-medium text-slate-600">Choose a file or drag it here</span>
              <span className="text-xs text-slate-400">Supported formats: GLB, GLTF, OBJ, FBX</span>
              <input type="file" className="hidden" accept=".obj,.glb,.gltf,.fbx" onChange={handleFileUpload} />
            </label>

            <div className="flex flex-wrap items-center justify-center gap-2 pt-1 text-xs font-medium text-slate-600">
              {[".glb", ".gltf", ".obj", ".fbx"].map((format) => (
                <span key={format} className="rounded-full bg-slate-100 px-4 py-2">{format}</span>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Floating Replace / Reset button in the top-right corner when model is loaded */}
      {hasLoadedModel && (
        <div className="fixed top-6 right-6 z-30 flex items-center gap-3 pointer-events-auto animate-in fade-in duration-300">
          <button 
            onClick={() => renderer?.recenterCamera()}
            className="flex items-center gap-2 px-4 py-2 bg-slate-900/60 hover:bg-slate-850 backdrop-blur-md border border-white/10 rounded-xl text-xs font-bold text-white shadow-lg transition-all active:scale-95"
            title="Recenter Camera View"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
            Recenter
          </button>
          
          <label className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 border border-blue-500 rounded-xl text-xs font-bold text-white shadow-lg cursor-pointer transition-all active:scale-95">
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
            </svg>
            Replace Asset
            <input type='file' className="hidden" accept=".obj,.glb,.gltf,.fbx" onChange={handleFileUpload} />
          </label>
        </div>
      )}
    </div>
  );
}

