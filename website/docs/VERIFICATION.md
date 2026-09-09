# Verification record

## 2026-09-09 — English template adaptation

**Local preview:** `http://127.0.0.1:4322/`  
**Browser:** Codex In-app Browser  
**Build:** Astro static output

| Check | Actual result |
| --- | --- |
| Production build | `npm run build` passed. Astro generated one static route. |
| English metadata and UI | The page title, description, `lang="en"`, navigation, headings, fallbacks, accessible labels, and footer are English. |
| Template provenance | The selected Academic Project Page Template snapshot is pinned at `d38af1ccae1ce82c3404d2820c4c646afd409f81`; source files and CC BY-SA 4.0 notice are present. |
| Architecture exports | `tinyhr-flowchart.pdf` is one page at 1584 × 630 points; `tinyhr-flowchart.png` is 3200 × 1273 pixels. Both show all six diagram panels, the residual U-shaped connector, and the final Heart Rate block. |
| Wide viewport | Measured 1443 × 960 CSS pixels. Navigation was visible; the real hero preview measured 960 px wide; `scrollWidth` was 1432 px. |
| Mobile viewport | Measured 391 × 845 CSS pixels and 360 × 845 CSS pixels. The primary action spans the first row and the two secondary actions wrap beneath it; native media preserved their aspect ratios, the inline PDF viewer was omitted, and the direct PDF actions remained available. |
| Tablet navigation | Measured 800 CSS pixels. The section navigation remained visible as a three-column header; it is hidden only below 768 px. |
| Method interaction | The `How it works` anchor reached `#method`. Opening `View PDF inline` loaded `/downloads/tinyhr-flowchart.pdf` into the same-origin iframe at 560 px height. |
| Browser console | No warnings or errors were reported after page load and PDF-viewer interaction. |
| Reduced motion | The hero script only calls `play()` when `prefers-reduced-motion: reduce` is not matched. It does not call `load()` while the video has `preload="none"`. |
| Resource structure | The page contains direct GitHub, report, architecture, and video links. Hugging Face is displayed as planned text without a fabricated URL. |

## Scope remaining outside this verification

- No GitHub upload, Pages configuration, custom domain, or production hosting was performed.
- No external-browser or mobile-device compatibility run was performed.
- No live camera, upload, inference, account, or model-hosting function is part of this static page.
