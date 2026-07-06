/**
 * CleanFrame — Main Application Logic
 *
 * Frame-by-frame canvas-based logo removal pipeline.
 * Detects and removes the Gemini 4-pointed sparkle watermark
 * using temporal analysis for detection and boundary-inward
 * inpainting for pixel-perfect removal.
 */

import { FFmpeg } from '@ffmpeg/ffmpeg';
import { fetchFile, toBlobURL } from '@ffmpeg/util';

// ── State ──────────────────────────────────────────────────────────
let ffmpeg = null;
let ffmpegLoaded = false;
let videoFile = null;
let videoMeta = { width: 0, height: 0, duration: 0, fps: 30 };
let region = { x: 0, y: 0, w: 100, h: 50 };
let logoMask = null; // Uint8Array mask of logo pixels within the region
let isDrawing = false;
let drawStart = { x: 0, y: 0 };
let isProcessing = false;
let cancelRequested = false;

// ── DOM References ─────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const loader = $('#ffmpeg-loader');
const appWrapper = $('#app-wrapper');

const uploadSection = $('#upload-section');
const editorSection = $('#editor-section');
const processingSection = $('#processing-section');
const resultsSection = $('#results-section');

const dropZone = $('#drop-zone');
const fileInput = $('#file-input');

const videoPreview = $('#video-preview');
const selectionCanvas = $('#selection-canvas');
const canvasHint = $('#canvas-hint');
const videoContainer = $('#video-container');
const fileName = $('#file-name');
const videoMetaEl = $('#video-meta');

const btnPlay = $('#btn-play');
const playIcon = $('#play-icon');
const pauseIcon = $('#pause-icon');
const scrubber = $('#video-scrubber');
const timeDisplay = $('#time-display');

const inputX = $('#input-x');
const inputY = $('#input-y');
const inputW = $('#input-w');
const inputH = $('#input-h');
const qualitySelect = $('#quality-select');

const btnProcess = $('#btn-process');
const btnExtract = $('#btn-extract');
const btnAutoDetect = $('#btn-auto-detect');
const btnCancel = $('#btn-cancel');
const btnNew = $('#btn-new');
const btnDownloadVideo = $('#btn-download-video');
const btnDownloadLogo = $('#btn-download-logo');

const autoDetectStatus = $('#auto-detect-status');

const progressPercent = $('#progress-percent');
const progressMessage = $('#progress-message');
const progressSub = $('#progress-sub');
const progressBar = $('#progress-bar');

const resultVideo = $('#result-video');
const resultVideoCard = $('#result-video-card');
const resultVideoInfo = $('#result-video-info');
const resultLogo = $('#result-logo');
const resultLogoCard = $('#result-logo-card');
const resultsSubtitle = $('#results-subtitle');

let resultVideoURL = null;
let resultLogoURL = null;

// ══════════════════════════════════════════════════════════════════
//  UTILITIES
// ══════════════════════════════════════════════════════════════════

function formatTime(seconds) {
  if (!seconds || !isFinite(seconds)) return '0:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function seekVideo(time) {
  return new Promise((resolve) => {
    if (Math.abs(videoPreview.currentTime - time) < 0.01) {
      resolve();
      return;
    }
    const handler = () => {
      videoPreview.removeEventListener('seeked', handler);
      // Small delay to ensure frame is fully rendered
      setTimeout(resolve, 30);
    };
    videoPreview.addEventListener('seeked', handler);
    videoPreview.currentTime = time;
    // Fallback timeout
    setTimeout(() => {
      videoPreview.removeEventListener('seeked', handler);
      resolve();
    }, 3000);
  });
}

function canvasToBlobAsync(canvas, type, quality) {
  return new Promise((resolve) => {
    canvas.toBlob(resolve, type, quality);
  });
}

async function cleanupVFS() {
  const filesToDelete = ['input.mp4', 'output.mp4', 'logo.png', 'audio.aac', 'audio.m4a'];
  for (const f of filesToDelete) {
    try { await ffmpeg.deleteFile(f); } catch {}
  }
}

// ══════════════════════════════════════════════════════════════════
//  FFMPEG INITIALIZATION
// ══════════════════════════════════════════════════════════════════

async function initFFmpeg() {
  ffmpegLoaded = true;
  loader.classList.add('loaded');
  appWrapper.style.display = '';
  setTimeout(() => (loader.style.display = 'none'), 600);
}

// ══════════════════════════════════════════════════════════════════
//  FILE UPLOAD
// ══════════════════════════════════════════════════════════════════

function setupUpload() {
  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
  });
  dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });
}

function handleFile(file) {
  const validExts = ['.mp4', '.webm', '.mov'];
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  if (!validExts.includes(ext) && !file.type.startsWith('video/')) {
    alert('Please upload an MP4, WebM, or MOV video.');
    return;
  }
  if (file.size > 500 * 1024 * 1024) {
    alert('File too large (max 500MB for in-browser processing).');
    return;
  }

  videoFile = file;
  const url = URL.createObjectURL(file);
  videoPreview.src = url;

  videoPreview.onloadedmetadata = () => {
    videoMeta.width = videoPreview.videoWidth;
    videoMeta.height = videoPreview.videoHeight;
    videoMeta.duration = videoPreview.duration;

    // Estimate FPS — most AI-generated videos are 24fps
    videoMeta.fps = 24;

    fileName.textContent = file.name;
    videoMetaEl.innerHTML = `
      <span>📐 ${videoMeta.width}×${videoMeta.height}</span>
      <span>⏱ ${formatTime(videoMeta.duration)}</span>
      <span>📦 ${formatSize(file.size)}</span>
      <span>🎞 ~${videoMeta.fps}fps</span>
    `;

    setupCanvas();
    showSection('editor');

    // Auto-detect watermark position
    setTimeout(() => autoDetectLogo(), 300);
  };
}

// ══════════════════════════════════════════════════════════════════
//  SECTION MANAGEMENT
// ══════════════════════════════════════════════════════════════════

function showSection(name) {
  [uploadSection, editorSection, processingSection, resultsSection].forEach((s) =>
    s.classList.add('hidden')
  );
  const map = { upload: uploadSection, editor: editorSection, processing: processingSection, results: resultsSection };
  if (map[name]) {
    map[name].classList.remove('hidden');
    map[name].style.animation = 'none';
    map[name].offsetHeight;
    map[name].style.animation = '';
  }
}

function updateProgress(pct, message, sub) {
  progressPercent.textContent = `${pct}%`;
  progressBar.style.width = `${pct}%`;
  if (message) progressMessage.textContent = message;
  if (sub !== undefined) progressSub.textContent = sub;
}

// ══════════════════════════════════════════════════════════════════
//  AUTO-DETECT LOGO — Temporal Frame Analysis
// ══════════════════════════════════════════════════════════════════

async function autoDetectLogo() {
  if (!videoFile) return;

  const statusEl = autoDetectStatus;
  statusEl.textContent = '🔍 Analyzing video on GPU…';
  statusEl.className = 'auto-detect-status';
  if (btnAutoDetect) { btnAutoDetect.disabled = true; btnAutoDetect.textContent = '⏳ Detecting…'; }

  try {
    const formData = new FormData();
    formData.append('file', videoFile);

    const response = await fetch('/api/detect-logo', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(await response.text());
    }

    const bbox = await response.json();
    region = { x: bbox.x, y: bbox.y, w: bbox.w, h: bbox.h };
    syncInputsFromRegion();
    drawOverlay();
    clearActivePresets();

    statusEl.textContent = `✅ Watermark detected — (${region.w}×${region.h}px at x=${region.x}, y=${region.y})`;
    statusEl.className = 'auto-detect-status success';
  } catch (err) {
    console.error('Auto-detect error:', err);
    applyPreset('bottom-right');
    statusEl.textContent = '⚠️ Auto-detection failed — using default bottom-right preset.';
    statusEl.className = 'auto-detect-status warning';
  } finally {
    if (btnAutoDetect) {
      btnAutoDetect.disabled = false;
      btnAutoDetect.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg> Auto-Detect Logo`;
    }
  }
}

// ══════════════════════════════════════════════════════════════════
//  LOGO MASK GENERATION
// ══════════════════════════════════════════════════════════════════

/**
 * Creates a per-pixel binary mask of logo pixels within the detected region.
 * Uses temporal consistency + brightness analysis across multiple frames.
 *
 * A pixel is marked as "logo" if:
 * 1. It has LOW temporal variance (doesn't change across frames = overlay)
 * 2. It's BRIGHTER than the surrounding moving content
 */
async function createLogoMask(capturedFrames) {
  const { x, y, w, h } = region;
  const W = videoMeta.width;
  const mask = new Uint8Array(w * h);

  // If we already have captured frames, reuse them
  let frames = capturedFrames;
  if (!frames || frames.length < 3) {
    frames = [];
    const dur = videoMeta.duration;
    const times = [dur * 0.2, dur * 0.4, dur * 0.6, dur * 0.8];
    for (const t of times) {
      await seekVideo(t);
      const c = document.createElement('canvas');
      c.width = W; c.height = videoMeta.height;
      const ctx = c.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(videoPreview, 0, 0, W, videoMeta.height);
      frames.push(ctx.getImageData(0, 0, W, videoMeta.height));
    }
  }

  // Compute per-pixel metrics within the logo region
  for (let ly = 0; ly < h; ly++) {
    for (let lx = 0; lx < w; lx++) {
      const gx = x + lx, gy = y + ly;
      if (gx >= W || gy >= videoMeta.height) continue;
      const pi = (gy * W + gx) * 4;

      // Collect RGB across frames
      let sumR = 0, sumG = 0, sumB = 0;
      const vals = [];
      for (const frame of frames) {
        const r = frame.data[pi], g = frame.data[pi + 1], b = frame.data[pi + 2];
        const brightness = (r * 0.299 + g * 0.587 + b * 0.114);
        vals.push(brightness);
        sumR += r; sumG += g; sumB += b;
      }

      const n = vals.length;
      const meanBright = vals.reduce((s, v) => s + v, 0) / n;

      // Temporal variance (how much does this pixel change?)
      let variance = 0;
      for (const v of vals) variance += (v - meanBright) ** 2;
      variance /= n;

      // Max channel difference across frame pairs
      let maxDiff = 0;
      for (let f = 1; f < frames.length; f++) {
        maxDiff = Math.max(maxDiff,
          Math.abs(frames[f].data[pi] - frames[0].data[pi]),
          Math.abs(frames[f].data[pi + 1] - frames[0].data[pi + 1]),
          Math.abs(frames[f].data[pi + 2] - frames[0].data[pi + 2])
        );
      }

      // Logo pixel criteria:
      // - Low temporal variation (the sparkle doesn't change)
      // - OR the pixel is consistently brighter than context
      const isStatic = maxDiff < 20;
      const isBright = meanBright > 120;
      const isModBright = meanBright > 80;

      if (isStatic && isModBright) {
        mask[ly * w + lx] = 1;
      }
    }
  }

  // Morphological cleanup: remove isolated pixels and fill small gaps
  const cleaned = morphClean(mask, w, h);

  // Dilate slightly to ensure we cover the edges of the sparkle
  const dilated = morphDilate(cleaned, w, h, 2);

  const totalPixels = dilated.reduce((s, v) => s + v, 0);
  console.log(`[mask] Created mask: ${totalPixels} logo pixels out of ${w * h} total (${(totalPixels / (w * h) * 100).toFixed(1)}%)`);

  // If mask is empty or too small, fall back to sparkle shape
  if (totalPixels < 20) {
    console.log('[mask] Adaptive mask too small, using sparkle shape fallback');
    return generateSparkleMask(w, h);
  }

  // If mask covers > 90% of region, it's probably a false positive (static bg)
  if (totalPixels > w * h * 0.9) {
    console.log('[mask] Mask covers entire region (static background), using sparkle shape fallback');
    return generateSparkleMask(w, h);
  }

  return dilated;
}

/** Remove isolated pixels (noise) from a binary mask */
function morphClean(mask, w, h) {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!mask[y * w + x]) continue;
      // Count neighbors
      let neighbors = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = x + dx, ny = y + dy;
          if (nx >= 0 && nx < w && ny >= 0 && ny < h && mask[ny * w + nx]) neighbors++;
        }
      }
      if (neighbors >= 2) out[y * w + x] = 1; // keep if at least 2 neighbors
    }
  }
  return out;
}

/** Dilate a binary mask by 'radius' pixels */
function morphDilate(mask, w, h, radius) {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!mask[y * w + x]) continue;
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
            out[ny * w + nx] = 1;
          }
        }
      }
    }
  }
  return out;
}

/**
 * Generates a programmatic 4-pointed Gemini sparkle mask.
 * Used as fallback when adaptive detection can't create a good mask.
 * Uses an astroid (hypocycloid) curve: x=cos³(t), y=sin³(t)
 */
function generateSparkleMask(w, h) {
  const mask = new Uint8Array(w * h);
  const cx = w / 2, cy = h / 2;
  const rx = w / 2 * 0.92, ry = h / 2 * 0.92;

  // Generate astroid boundary points
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');

  ctx.beginPath();
  const steps = 200;
  for (let i = 0; i <= steps; i++) {
    const t = (i / steps) * 2 * Math.PI;
    const px = cx + rx * Math.pow(Math.cos(t), 3);
    const py = cy + ry * Math.pow(Math.sin(t), 3);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.closePath();
  ctx.fillStyle = 'white';
  ctx.fill();

  // Read back as mask
  const imgData = ctx.getImageData(0, 0, w, h);
  for (let i = 0; i < w * h; i++) {
    mask[i] = imgData.data[i * 4] > 128 ? 1 : 0;
  }

  // Dilate by 1px for safety
  return morphDilate(mask, w, h, 1);
}

// ══════════════════════════════════════════════════════════════════
//  INPAINTING ENGINE — Boundary-Inward Fill
// ══════════════════════════════════════════════════════════════════

/**
 * Inpaints the logo region in-place on a canvas context.
 *
 * Implements a Fast Marching Method (FMM) approximation with gradient propagation:
 * - Computes the distance from the boundary for each masked pixel.
 * - Iteratively fills pixels starting from the boundary inward (onion peeling).
 * - Interpolates pixels using neighboring gradient-corrected color values.
 * - Applies a bilateral-like spatial blending to match background texture noise.
 */


function inpaintFrame(ctx, mask, regionBox, fixedOffset = null, prevInpaintData = null) {
  const { x, y, w, h } = regionBox;
  const W = videoMeta.width;
  const H = videoMeta.height;

  // 1. Get raw frame data containing the region plus a search margin
  const searchMargin = 64;
  const rx = Math.max(0, x - searchMargin);
  const ry = Math.max(0, y - searchMargin);
  const rw = Math.min(w + 2 * searchMargin, W - rx);
  const rh = Math.min(h + 2 * searchMargin, H - ry);

  const imgData = ctx.getImageData(rx, ry, rw, rh);
  const pixels = imgData.data;

  // Map absolute coordinates to local frame coordinates
  const localX = x - rx;
  const localY = y - ry;

  // Local mask matching the exact logo region coordinates
  const localMask = new Uint8Array(rw * rh);
  for (let my = 0; my < h; my++) {
    for (let mx = 0; mx < w; mx++) {
      if (mask[my * w + mx]) {
        const lx = mx + localX;
        const ly = my + localY;
        if (lx >= 0 && lx < rw && ly >= 0 && ly < rh) {
          localMask[ly * rw + lx] = 1;
        }
      }
    }
  }

  // 2. Identify the boundary band (pixels outside the mask, but within 6px of the mask edge)
  const boundaryBand = new Uint8Array(rw * rh);
  const bandWidth = 6;
  for (let ly = 0; ly < rh; ly++) {
    for (let lx = 0; lx < rw; lx++) {
      if (localMask[ly * rw + lx] === 1) continue;

      let nearMask = false;
      for (let dy = -bandWidth; dy <= bandWidth && !nearMask; dy++) {
        for (let dx = -bandWidth; dx <= bandWidth && !nearMask; dx++) {
          const nx = lx + dx;
          const ny = ly + dy;
          if (nx >= 0 && nx < rw && ny >= 0 && ny < rh) {
            if (localMask[ny * rw + nx] === 1) {
              nearMask = true;
            }
          }
        }
      }
      if (nearMask) {
        boundaryBand[ly * rw + lx] = 1;
      }
    }
  }

  // 3. Search for the best candidate patch displacement (dx, dy)
  let bestDx = 0;
  let bestDy = 0;
  let foundPatch = false;

  if (fixedOffset) {
    bestDx = fixedOffset.dx;
    bestDy = fixedOffset.dy;
    foundPatch = true;
  } else {
    let minSSD = Infinity;
    const step = 2;
    for (let dy = -searchMargin; dy <= searchMargin; dy += step) {
      for (let dx = -searchMargin; dx <= searchMargin; dx += step) {
        if (Math.abs(dx) < 8 && Math.abs(dy) < 8) continue;

        let overlapsMask = false;
        for (let my = 0; my < h && !overlapsMask; my++) {
          for (let mx = 0; mx < w && !overlapsMask; mx++) {
            if (mask[my * w + mx]) {
              const sx = localX + mx + dx;
              const sy = localY + my + dy;
              if (sx < 0 || sx >= rw || sy < 0 || sy >= rh || localMask[sy * rw + sx] === 1) {
                overlapsMask = true;
              }
            }
          }
        }
        if (overlapsMask) continue;

        let ssd = 0;
        let count = 0;
        for (let ly = 0; ly < rh; ly++) {
          for (let lx = 0; lx < rw; lx++) {
            if (boundaryBand[ly * rw + lx] === 1) {
              const sx = lx + dx;
              const sy = ly + dy;
              if (sx >= 0 && sx < rw && sy >= 0 && sy < rh) {
                const targetIdx = (ly * rw + lx) * 4;
                const sourceIdx = (sy * rw + sx) * 4;

                const dr = pixels[targetIdx] - pixels[sourceIdx];
                const dg = pixels[targetIdx + 1] - pixels[sourceIdx + 1];
                const db = pixels[targetIdx + 2] - pixels[sourceIdx + 2];

                ssd += dr * dr + dg * dg + db * db;
                count++;
              }
            }
          }
        }

        if (count > 0) {
          const normSSD = ssd / count;
          if (normSSD < minSSD) {
            minSSD = normSSD;
            bestDx = dx;
            bestDy = dy;
            foundPatch = true;
          }
        }
      }
    }

    if (foundPatch) {
      let fineDx = bestDx;
      let fineDy = bestDy;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const testDx = bestDx + dx;
          const testDy = bestDy + dy;

          let ssd = 0;
          let count = 0;
          for (let ly = 0; ly < rh; ly++) {
            for (let lx = 0; lx < rw; lx++) {
              if (boundaryBand[ly * rw + lx] === 1) {
                const sx = lx + testDx;
                const sy = ly + testDy;
                if (sx >= 0 && sx < rw && sy >= 0 && sy < rh) {
                  const targetIdx = (ly * rw + lx) * 4;
                  const sourceIdx = (sy * rw + sx) * 4;

                  const dr = pixels[targetIdx] - pixels[sourceIdx];
                  const dg = pixels[targetIdx + 1] - pixels[sourceIdx + 1];
                  const db = pixels[targetIdx + 2] - pixels[sourceIdx + 2];

                  ssd += dr * dr + dg * dg + db * db;
                  count++;
                }
              }
            }
          }
          if (count > 0) {
            const normSSD = ssd / count;
            if (normSSD < minSSD) {
              minSSD = normSSD;
              fineDx = testDx;
              fineDy = testDy;
            }
          }
        }
      }
      bestDx = fineDx;
      bestDy = fineDy;
    }
  }

  const outputInpaintData = {
    r: new Uint8Array(rw * rh),
    g: new Uint8Array(rw * rh),
    b: new Uint8Array(rw * rh),
    dx: bestDx,
    dy: bestDy,
    found: foundPatch
  };

  // 4. Fill mask pixels using Poisson Blending Solver if patch is found
  if (foundPatch) {
    const laplacianR = new Float32Array(rw * rh);
    const laplacianG = new Float32Array(rw * rh);
    const laplacianB = new Float32Array(rw * rh);

    for (let ly = 1; ly < rh - 1; ly++) {
      for (let lx = 1; lx < rw - 1; lx++) {
        const idx = ly * rw + lx;
        if (localMask[idx] === 1) {
          const sx = lx + bestDx;
          const sy = ly + bestDy;

          if (sx >= 1 && sx < rw - 1 && sy >= 1 && sy < rh - 1) {
            const spi = (sy * rw + sx) * 4;
            const spi_up = ((sy - 1) * rw + sx) * 4;
            const spi_down = ((sy + 1) * rw + sx) * 4;
            const spi_left = (sy * rw + (sx - 1)) * 4;
            const spi_right = (sy * rw + (sx + 1)) * 4;

            laplacianR[idx] = 4 * pixels[spi] - pixels[spi_up] - pixels[spi_down] - pixels[spi_left] - pixels[spi_right];
            laplacianG[idx] = 4 * pixels[spi + 1] - pixels[spi_up + 1] - pixels[spi_down + 1] - pixels[spi_left + 1] - pixels[spi_right + 1];
            laplacianB[idx] = 4 * pixels[spi + 2] - pixels[spi_up + 2] - pixels[spi_down + 2] - pixels[spi_left + 2] - pixels[spi_right + 2];
          }
        }
      }
    }

    const solR = new Float32Array(rw * rh);
    const solG = new Float32Array(rw * rh);
    const solB = new Float32Array(rw * rh);

    for (let ly = 0; ly < rh; ly++) {
      for (let lx = 0; lx < rw; lx++) {
        const idx = ly * rw + lx;
        const pi = idx * 4;
        if (localMask[idx] === 1) {
          const sx = lx + bestDx;
          const sy = ly + bestDy;
          if (sx >= 0 && sx < rw && sy >= 0 && sy < rh) {
            const spi = (sy * rw + sx) * 4;
            solR[idx] = pixels[spi];
            solG[idx] = pixels[spi + 1];
            solB[idx] = pixels[spi + 2];
          }
        } else {
          solR[idx] = pixels[pi];
          solG[idx] = pixels[pi + 1];
          solB[idx] = pixels[pi + 2];
        }
      }
    }

    const nextR = new Float32Array(rw * rh);
    const nextG = new Float32Array(rw * rh);
    const nextB = new Float32Array(rw * rh);

    const iterations = 45;
    for (let iter = 0; iter < iterations; iter++) {
      for (let ly = 1; ly < rh - 1; ly++) {
        for (let lx = 1; lx < rw - 1; lx++) {
          const idx = ly * rw + lx;
          if (localMask[idx] === 1) {
            const up = (ly - 1) * rw + lx;
            const down = (ly + 1) * rw + lx;
            const left = ly * rw + (lx - 1);
            const right = ly * rw + (lx + 1);

            nextR[idx] = 0.25 * (solR[up] + solR[down] + solR[left] + solR[right] - laplacianR[idx]);
            nextG[idx] = 0.25 * (solG[up] + solG[down] + solG[left] + solG[right] - laplacianG[idx]);
            nextB[idx] = 0.25 * (solB[up] + solB[down] + solB[left] + solB[right] - laplacianB[idx]);
          }
        }
      }

      for (let ly = 1; ly < rh - 1; ly++) {
        for (let lx = 1; lx < rw - 1; lx++) {
          const idx = ly * rw + lx;
          if (localMask[idx] === 1) {
            solR[idx] = nextR[idx];
            solG[idx] = nextG[idx];
            solB[idx] = nextB[idx];
          }
        }
      }
    }

    const blendFactor = 0.82;
    for (let ly = 0; ly < rh; ly++) {
      for (let lx = 0; lx < rw; lx++) {
        const idx = ly * rw + lx;
        if (localMask[idx] === 1) {
          const pi = idx * 4;
          
          let targetR = Math.round(solR[idx]);
          let targetG = Math.round(solG[idx]);
          let targetB = Math.round(solB[idx]);

          if (prevInpaintData) {
            const pr = prevInpaintData.r[idx];
            const pg = prevInpaintData.g[idx];
            const pb = prevInpaintData.b[idx];
            targetR = Math.round(targetR * blendFactor + pr * (1.0 - blendFactor));
            targetG = Math.round(targetG * blendFactor + pg * (1.0 - blendFactor));
            targetB = Math.round(targetB * blendFactor + pb * (1.0 - blendFactor));
          }

          outputInpaintData.r[idx] = targetR;
          outputInpaintData.g[idx] = targetG;
          outputInpaintData.b[idx] = targetB;

          pixels[pi] = Math.max(0, Math.min(255, targetR));
          pixels[pi + 1] = Math.max(0, Math.min(255, targetG));
          pixels[pi + 2] = Math.max(0, Math.min(255, targetB));
        }
      }
    }
  } else {
    console.log('[inpaint] No suitable texture patch found, falling back to spatial FMM');
    
    const distMap = new Float32Array(rw * rh);
    distMap.fill(999999);
    const fmmQueue = [];

    for (let ly = 0; ly < rh; ly++) {
      for (let lx = 0; lx < rw; lx++) {
        const idx = ly * rw + lx;
        if (localMask[idx] === 1) {
          let isBoundary = false;
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              const nx = lx + dx, ny = ly + dy;
              if (nx >= 0 && nx < rw && ny >= 0 && ny < rh && localMask[ny * rw + nx] === 0) {
                isBoundary = true;
              }
            }
          }
          if (isBoundary) {
            distMap[idx] = 0;
            fmmQueue.push({ x: lx, y: ly, dist: 0 });
          }
        }
      }
    }

    fmmQueue.sort((a, b) => a.dist - b.dist);
    while (fmmQueue.length > 0) {
      const { x: cx, y: cy, dist: cd } = fmmQueue.shift();
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = cx + dx, ny = cy + dy;
          if (nx >= 0 && nx < rw && ny >= 0 && ny < rh) {
            const nidx = ny * rw + nx;
            if (localMask[nidx] === 1) {
              const weight = (dx !== 0 && dy !== 0) ? 1.414 : 1.0;
              const newDist = cd + weight;
              if (newDist < distMap[nidx]) {
                distMap[nidx] = newDist;
                fmmQueue.push({ x: nx, y: ny, dist: newDist });
              }
            }
          }
        }
      }
      fmmQueue.sort((a, b) => a.dist - b.dist);
    }

    const pixelsToFill = [];
    for (let ly = 0; ly < rh; ly++) {
      for (let lx = 0; lx < rw; lx++) {
        const idx = ly * rw + lx;
        if (localMask[idx] === 1) {
          pixelsToFill.push({ x: lx, y: ly, dist: distMap[idx], idx });
        }
      }
    }
    pixelsToFill.sort((a, b) => a.dist - b.dist);

    const status = new Uint8Array(localMask);
    for (const p of pixelsToFill) {
      const px = p.x, py = p.y, pidx = p.idx;
      let sumR = 0, sumG = 0, sumB = 0, totalW = 0;
      const r_search = Math.max(5, Math.ceil(p.dist) + 3);

      for (let dy = -r_search; dy <= r_search; dy++) {
        for (let dx = -r_search; dx <= r_search; dx++) {
          const nx = px + dx, ny = py + dy;
          if (nx < 0 || nx >= rw || ny < 0 || ny >= rh) continue;
          const nidx = ny * rw + nx;

          if (status[nidx] === 0) {
            const d_space = Math.sqrt(dx * dx + dy * dy);
            if (d_space === 0) continue;
            let w_space = 1.0 / Math.pow(d_space, 2.2);
            if (distMap[nidx] === 0) w_space *= 1.8;

            const npi = nidx * 4;
            sumR += pixels[npi] * w_space;
            sumG += pixels[npi + 1] * w_space;
            sumB += pixels[npi + 2] * w_space;
            totalW += w_space;
          }
        }
      }

      if (totalW > 0) {
        const ppi = pidx * 4;
        
        let targetR = Math.round(sumR / totalW);
        let targetG = Math.round(sumG / totalW);
        let targetB = Math.round(sumB / totalW);

        if (prevInpaintData) {
          const pr = prevInpaintData.r[pidx];
          const pg = prevInpaintData.g[pidx];
          const pb = prevInpaintData.b[pidx];
          const bf = 0.85;
          targetR = Math.round(targetR * bf + pr * (1.0 - bf));
          targetG = Math.round(targetG * bf + pg * (1.0 - bf));
          targetB = Math.round(targetB * bf + pb * (1.0 - bf));
        }

        outputInpaintData.r[pidx] = targetR;
        outputInpaintData.g[pidx] = targetG;
        outputInpaintData.b[pidx] = targetB;

        pixels[ppi] = targetR;
        pixels[ppi + 1] = targetG;
        pixels[ppi + 2] = targetB;
        status[pidx] = 0;
      }
    }
  }

  // 5. Add noise dithering matching background film grain
  for (let ly = 0; ly < rh; ly++) {
    for (let lx = 0; lx < rw; lx++) {
      const idx = ly * rw + lx;
      if (localMask[idx] === 1) {
        const ppi = idx * 4;
        const noise = (Math.random() - 0.5) * 1.5;
        pixels[ppi] = Math.max(0, Math.min(255, pixels[ppi] + noise));
        pixels[ppi + 1] = Math.max(0, Math.min(255, pixels[ppi + 1] + noise));
        pixels[ppi + 2] = Math.max(0, Math.min(255, pixels[ppi + 2] + noise));
      }
    }
  }

  ctx.putImageData(imgData, rx, ry);
  return outputInpaintData;
}


let pollInterval = null;

async function processVideo() {
  if (!videoFile || isProcessing) return;

  isProcessing = true;
  cancelRequested = false;
  showSection('processing');
  updateProgress(0, 'Uploading video…', 'Sending video file to background GPU worker');

  try {
    const formData = new FormData();
    formData.append('file', videoFile);

    // Build URL with optional crop coordinates if region has been changed
    let uploadUrl = '/api/upload';
    if (region.w > 4 && region.h > 4) {
      uploadUrl += `?x=${region.x}&y=${region.y}&w=${region.w}&h=${region.h}`;
    }

    const response = await fetch(uploadUrl, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(await response.text());
    }

    const result = await response.json();
    const jobId = result.job_id;

    updateProgress(2, 'In queue…', 'Waiting for GPU slot');

    // Poll status every 2 seconds
    pollInterval = setInterval(async () => {
      if (cancelRequested) {
        clearInterval(pollInterval);
        return;
      }

      try {
        const statusRes = await fetch(`/api/status/${jobId}`);
        if (!statusRes.ok) throw new Error("Status query failed");

        const statusData = await statusRes.json();
        const { status, progress, stage, message } = statusData;

        if (status === 'queued') {
          updateProgress(2, 'In queue…', message);
        } else if (status === 'processing') {
          // Map backend progress (0-1) to UI progress
          const pct = Math.round(progress * 100);
          updateProgress(pct, `Stage: ${stage}`, message);
        } else if (status === 'completed') {
          clearInterval(pollInterval);
          updateProgress(100, 'Finalizing download…', 'Fetching output video');

          // Download clean video
          const downloadRes = await fetch(`/api/download/${jobId}`);
          if (!downloadRes.ok) throw new Error("Video download failed");
          const videoBlob = await downloadRes.blob();

          showResults(videoBlob, null);
          isProcessing = false;
        } else if (status === 'failed') {
          clearInterval(pollInterval);
          throw new Error(message || "Processing failed on server.");
        }
      } catch (pollErr) {
        console.error("Polling error:", pollErr);
      }
    }, 2000);

  } catch (err) {
    console.error('Processing error:', err);
    if (pollInterval) clearInterval(pollInterval);
    alert('Processing failed: ' + (err.message || String(err)));
    showSection('editor');
    isProcessing = false;
  }
}

// ══════════════════════════════════════════════════════════════════
//  EXTRACT LOGO AS HD PNG
// ══════════════════════════════════════════════════════════════════

async function extractLogo() {
  if (!videoFile || isProcessing) return;

  isProcessing = true;
  showSection('processing');
  updateProgress(20, 'Uploading video…', 'Extracting logo region in HD');

  try {
    const formData = new FormData();
    formData.append('file', videoFile);

    let extractUrl = '/api/extract';
    if (region.w > 4 && region.h > 4) {
      extractUrl += `?x=${region.x}&y=${region.y}&w=${region.w}&h=${region.h}`;
    }

    const response = await fetch(extractUrl, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(await response.text());
    }

    const logoBlob = await response.blob();
    showResults(null, logoBlob);
  } catch (err) {
    console.error('Logo extraction error:', err);
    alert('Logo extraction failed: ' + (err.message || String(err)));
    showSection('editor');
  } finally {
    isProcessing = false;
  }
}

// ══════════════════════════════════════════════════════════════════
//  CANVAS DRAWING & REGION SELECTION
// ══════════════════════════════════════════════════════════════════

function setupCanvas() {
  const canvas = selectionCanvas;
  const container = videoContainer;

  function resizeCanvas() {
    const rect = container.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    drawOverlay();
  }
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  canvas.addEventListener('mousedown', onDrawStart);
  canvas.addEventListener('mousemove', onDrawMove);
  canvas.addEventListener('mouseup', onDrawEnd);
  canvas.addEventListener('mouseleave', onDrawEnd);

  canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    const t = e.touches[0], rect = canvas.getBoundingClientRect();
    onDrawStart({ offsetX: t.clientX - rect.left, offsetY: t.clientY - rect.top });
  });
  canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    const t = e.touches[0], rect = canvas.getBoundingClientRect();
    onDrawMove({ offsetX: t.clientX - rect.left, offsetY: t.clientY - rect.top });
  });
  canvas.addEventListener('touchend', onDrawEnd);
}

function canvasToVideo(cx, cy) {
  const rect = videoContainer.getBoundingClientRect();
  const cRatio = rect.width / rect.height;
  const vRatio = videoMeta.width / videoMeta.height;
  let dW, dH, oX, oY;
  if (cRatio > vRatio) {
    dH = rect.height; dW = rect.height * vRatio; oX = (rect.width - dW) / 2; oY = 0;
  } else {
    dW = rect.width; dH = rect.width / vRatio; oX = 0; oY = (rect.height - dH) / 2;
  }
  return { x: Math.round(((cx - oX) / dW) * videoMeta.width), y: Math.round(((cy - oY) / dH) * videoMeta.height) };
}

function videoToCanvas(vx, vy) {
  const rect = videoContainer.getBoundingClientRect();
  const cRatio = rect.width / rect.height;
  const vRatio = videoMeta.width / videoMeta.height;
  let dW, dH, oX, oY;
  if (cRatio > vRatio) {
    dH = rect.height; dW = rect.height * vRatio; oX = (rect.width - dW) / 2; oY = 0;
  } else {
    dW = rect.width; dH = rect.width / vRatio; oX = 0; oY = (rect.height - dH) / 2;
  }
  return { x: (vx / videoMeta.width) * dW + oX, y: (vy / videoMeta.height) * dH + oY };
}

function onDrawStart(e) {
  isDrawing = true;
  drawStart = { x: e.offsetX, y: e.offsetY };
  if (canvasHint) canvasHint.classList.add('hidden');
}

function onDrawMove(e) {
  if (!isDrawing) return;
  const v1 = canvasToVideo(drawStart.x, drawStart.y);
  const v2 = canvasToVideo(e.offsetX, e.offsetY);
  const x = Math.max(0, Math.min(v1.x, v2.x));
  const y = Math.max(0, Math.min(v1.y, v2.y));
  region = {
    x, y,
    w: Math.max(1, Math.min(Math.abs(v2.x - v1.x), videoMeta.width - x)),
    h: Math.max(1, Math.min(Math.abs(v2.y - v1.y), videoMeta.height - y)),
  };
  syncInputsFromRegion();
  drawOverlay();
}

function onDrawEnd() {
  if (isDrawing) {
    isDrawing = false;
    clearActivePresets();
    // Regenerate mask for new region
    logoMask = generateSparkleMask(region.w, region.h);
  }
}

function drawOverlay() {
  const canvas = selectionCanvas;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (region.w <= 0 || region.h <= 0) return;

  const tl = videoToCanvas(region.x, region.y);
  const br = videoToCanvas(region.x + region.w, region.y + region.h);
  const rw = br.x - tl.x, rh = br.y - tl.y;

  // Dim everything outside
  ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.clearRect(tl.x, tl.y, rw, rh);

  // Dashed border
  ctx.strokeStyle = '#8b5cf6';
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.strokeRect(tl.x, tl.y, rw, rh);
  ctx.setLineDash([]);

  // Corner handles
  ctx.fillStyle = '#8b5cf6';
  const hs = 8;
  [[tl.x, tl.y], [tl.x + rw, tl.y], [tl.x, tl.y + rh], [tl.x + rw, tl.y + rh]].forEach(([cx, cy]) => {
    ctx.fillRect(cx - hs / 2, cy - hs / 2, hs, hs);
  });

  // Dimension label
  if (tl.y > 20) {
    ctx.fillStyle = 'rgba(139, 92, 246, 0.85)';
    ctx.font = '600 11px Inter, sans-serif';
    const label = `${region.w}×${region.h}`;
    ctx.fillRect(tl.x, tl.y - 22, ctx.measureText(label).width + 12, 18);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, tl.x + 6, tl.y - 8);
  }
}

// ── Presets ────────────────────────────────────────────────────────
function applyPreset(position) {
  const w = videoMeta.width, h = videoMeta.height;
  const logoW = Math.round(Math.min(w, h) * 0.15);
  const logoH = Math.round(logoW * 0.55);
  const pad = Math.round(Math.min(w, h) * 0.02);

  switch (position) {
    case 'top-left': region = { x: pad, y: pad, w: logoW, h: logoH }; break;
    case 'top-right': region = { x: w - logoW - pad, y: pad, w: logoW, h: logoH }; break;
    case 'bottom-left': region = { x: pad, y: h - logoH - pad, w: logoW, h: logoH }; break;
    case 'bottom-right': region = { x: w - logoW - pad, y: h - logoH - pad, w: logoW, h: logoH }; break;
  }

  syncInputsFromRegion();
  drawOverlay();
  logoMask = generateSparkleMask(region.w, region.h);
}

function setupPresets() {
  document.querySelectorAll('.preset-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.preset-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      applyPreset(btn.dataset.preset);
    });
  });
}

function clearActivePresets() {
  document.querySelectorAll('.preset-btn').forEach((b) => b.classList.remove('active'));
}

// ── Coordinate Inputs ──────────────────────────────────────────────
function setupCoordInputs() {
  [inputX, inputY, inputW, inputH].forEach((inp) => {
    inp.addEventListener('input', () => {
      region.x = Math.max(0, parseInt(inputX.value) || 0);
      region.y = Math.max(0, parseInt(inputY.value) || 0);
      region.w = Math.max(1, parseInt(inputW.value) || 1);
      region.h = Math.max(1, parseInt(inputH.value) || 1);
      region.x = Math.min(region.x, videoMeta.width - 1);
      region.y = Math.min(region.y, videoMeta.height - 1);
      region.w = Math.min(region.w, videoMeta.width - region.x);
      region.h = Math.min(region.h, videoMeta.height - region.y);
      clearActivePresets();
      drawOverlay();
      logoMask = generateSparkleMask(region.w, region.h);
    });
  });
}

function syncInputsFromRegion() {
  inputX.value = region.x;
  inputY.value = region.y;
  inputW.value = region.w;
  inputH.value = region.h;
}

// ── Video Playback Controls ────────────────────────────────────────
function setupPlaybackControls() {
  btnPlay.addEventListener('click', () => {
    videoPreview.paused ? videoPreview.play() : videoPreview.pause();
  });
  videoPreview.addEventListener('play', () => { playIcon.style.display = 'none'; pauseIcon.style.display = ''; });
  videoPreview.addEventListener('pause', () => { playIcon.style.display = ''; pauseIcon.style.display = 'none'; });
  videoPreview.addEventListener('timeupdate', () => {
    if (!videoPreview.duration) return;
    scrubber.value = (videoPreview.currentTime / videoPreview.duration) * 100;
    timeDisplay.textContent = `${formatTime(videoPreview.currentTime)} / ${formatTime(videoPreview.duration)}`;
  });
  scrubber.addEventListener('input', () => {
    if (videoPreview.duration) videoPreview.currentTime = (scrubber.value / 100) * videoPreview.duration;
  });
}

// ══════════════════════════════════════════════════════════════════
//  RESULTS & DOWNLOAD
// ══════════════════════════════════════════════════════════════════

function showResults(videoBlob, logoBlob) {
  showSection('results');
  if (resultVideoURL) URL.revokeObjectURL(resultVideoURL);
  if (resultLogoURL) URL.revokeObjectURL(resultLogoURL);

  if (videoBlob) {
    resultVideoURL = URL.createObjectURL(videoBlob);
    resultVideo.src = resultVideoURL;
    resultVideoCard.style.display = '';
    resultVideoInfo.innerHTML = `<span>📦 ${formatSize(videoBlob.size)}</span>`;
    btnDownloadVideo.onclick = () => downloadBlob(videoBlob, getCleanFileName('video'));
    resultsSubtitle.textContent = `Logo removed · ${formatSize(videoBlob.size)} · 1080p quality`;
  } else {
    resultVideoCard.style.display = 'none';
  }

  if (logoBlob) {
    resultLogoURL = URL.createObjectURL(logoBlob);
    resultLogo.src = resultLogoURL;
    resultLogoCard.style.display = '';
    btnDownloadLogo.onclick = () => downloadBlob(logoBlob, getCleanFileName('logo'));
    if (!videoBlob) resultsSubtitle.textContent = `Logo extracted · ${formatSize(logoBlob.size)}`;
  } else {
    resultLogoCard.style.display = 'none';
  }
}

function downloadBlob(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(a.href);
}

function getCleanFileName(type) {
  const base = videoFile?.name?.replace(/\.[^.]+$/, '') || 'video';
  return type === 'logo' ? `${base}_logo.png` : `${base}_clean.mp4`;
}

// ── Action Buttons ─────────────────────────────────────────────────
function setupActions() {
  btnProcess.addEventListener('click', processVideo);
  btnExtract.addEventListener('click', extractLogo);
  if (btnAutoDetect) btnAutoDetect.addEventListener('click', autoDetectLogo);

  btnCancel.addEventListener('click', () => {
    cancelRequested = true;
    isProcessing = false;
    showSection('editor');
  });

  btnNew.addEventListener('click', () => {
    if (resultVideoURL) URL.revokeObjectURL(resultVideoURL);
    if (resultLogoURL) URL.revokeObjectURL(resultLogoURL);
    resultVideoURL = resultLogoURL = null;
    videoFile = null;
    logoMask = null;
    videoPreview.src = '';
    resultVideo.src = '';
    resultLogo.src = '';
    fileInput.value = '';
    if (autoDetectStatus) autoDetectStatus.className = 'auto-detect-status hidden';
    showSection('upload');
  });
}

// ── Boot ───────────────────────────────────────────────────────────
function init() {
  setupUpload();
  setupPresets();
  setupCoordInputs();
  setupPlaybackControls();
  setupActions();
  initFFmpeg();
}

init();
