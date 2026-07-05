# Gemini & Google Lab Flow Video Logo Remover

A high-performance, client-side web application built with **Vite** and **FFmpeg.wasm** to cleanly remove Gemini watermarks from videos in up to 1080p quality, as well as extract watermark logos as lossless PNGs.

## Key Features
- **Temporal Locked Patch Inpainting**: Locks the background texture offset globally across all frames so wave/ripple patterns animate in sync.
- **Poisson Image Editing Solver**: Resolves gradient membrane color boundary conditions seamlessly for a seamless blend.
- **100% Client-Side**: No servers involved, preserving absolute data privacy.
- **Lossless Frame Processing**: PNG frame encoding pipeline prevents JPEG block artifacts.
- **HD Logo Extraction**: Crops the watermark to a high-quality PNG.

## Setup & Local Development
1. Clone this repository.
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the dev server:
   ```bash
   npm run dev
   ```

## Deploying to Vercel
This project uses Vercel's headers configuration in `vercel.json` to enable **SharedArrayBuffer** support required by FFmpeg.wasm:
```json
{
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        { "key": "Cross-Origin-Embedder-Policy", "value": "require-corp" },
        { "key": "Cross-Origin-Opener-Policy", "value": "same-origin" }
      ]
    }
  ]
}
```
Simply connect your GitHub repository to Vercel, select the **Vite** project preset, and deploy!
