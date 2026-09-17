# Verification record

## 2026-09-16 - HERTZ collection layout and base-path repair

**Scope:** Shared HERTZ identity, independent TinyHR/AdaChrom method cards, future-estimator placeholder, and repository-subpath asset handling.

| Check | Actual result |
| --- | --- |
| Astro diagnostics | `npm run check` completed with 0 errors, 0 warnings, and 0 hints. |
| Production build | `npm run build` passed and generated `/` and `/zh/`. |
| Base-path output | Both generated pages reference `/seetapsych-hertz/media/adachrom-pipeline.png`; no authored `src="/..."`, `href="/..."`, `poster="/..."`, or `data-src="/..."` remains in `website/src`. |
| Desktop layout | English method navigation, light-blue TinyHR card, light-green AdaChrom card, pipeline figure, and extension placeholder were visually inspected. |
| Mobile layout | English and Chinese method sections were checked at a 390 x 844 viewport; cards stack without horizontal overflow. |
| Browser console | No warnings or errors were recorded during the targeted preview checks. |
| Report transcription | Root `README.md` contains the complete Section 3.2.3.1 text from report pages 9-11, including four ROI strategies, BVP extraction equations, and FFT post-processing. |

The full SeetaPsych report remains a local reference and is not distributed with the repository.

## 2026-09-15 — AdaChrom method addition

**Local preview:** `http://127.0.0.1:4325/`

**Browser:** Headless Chromium, driven by Playwright

**Scope:** Website content, layout, and extracted AdaChrom figure; no model accuracy or inference-speed benchmark was independently run during website QA.

| Check | Actual result |
| --- | --- |
| Production build | `npm run build` passed and generated `/` and `/zh/`. |
| Responsive layout | Final AdaChrom content was checked at observed 1028 px desktop and 403 px mobile widths; no horizontal overflow. Screenshots were visually reviewed. |
| Method navigation | AdaChrom navigation reached `#adachrom` on both routes. |
| AdaChrom content | One concise paragraph is present in both languages, followed by the pipeline figure. |
| Figure asset | The 1170 × 460 PNG was cropped from Figure 3 on report page 10 and visually checked for complete labels and arrows. |
| TinyHR benchmark copy | English and Chinese text identifies 100 runs per device, 80 ms mean on Intel Core i9-13900KF CPU at 3.00 GHz, and 6 ms mean on NVIDIA H20 GPU. Values are team-provided and were not independently rerun during website QA. |
| Browser exceptions | No uncaught page errors were captured during these targeted checks. |
| Source review | AdaChrom text and pipeline checked against report section 3.2 and rendered page 10. The full source report is not distributed with the website. |

These checks cover the local static preview, not production deployment or testing on physical mobile devices.

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
