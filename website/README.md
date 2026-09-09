# SeetaPsych Hertz — Research Project Page

A static research showcase for TinyHR: facial video, remote photoplethysmography (rPPG), and heart-rate estimation.

**Revision: 2026-09-09 · English template redesign implemented locally.** The current site is the English Astro adaptation: it uses the selected template's research-page composition, real recorded video, report-backed method text, and the supplied architecture diagram. Current build and browser evidence is recorded in [docs/VERIFICATION.md](docs/VERIFICATION.md).

## 1. Fixed scope

Build one English page with a striking video introduction, a readable architecture figure, report-backed method text, evaluation results, and real resource links. Keep the initial implementation simple.

- All authored interface text must be English: navigation, headings, buttons, captions, accessible labels, fallback messages, page metadata, and footer.
- Use the selected open-source template below as the actual starting layout. Port its relevant HTML and CSS into the existing Astro project.
- Show the supplied architecture diagram in **How it works**, with a PDF export and an image preview. Use the technical report for the English explanation.
- Keep project documentation and public copy focused on the software, research, evidence, and implementation.
- Retain recorded video playback. Live inference, camera access, uploads, accounts, databases, localization, theme switching, custom video controls, and complex animation are outside this revision.
- Repository publishing and hosting configuration are separate work. This specification does not represent a deployed site.

This README describes the current implementation and its maintained source assets.

## 2. Selected GitHub template

**Use [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template), by Eliahu Horwitz.** [Live example](https://eliahuhorwitz.github.io/Academic-project-page-template/). Reviewed on 2026-09-09, using the `master` version.

It provides the title/action group, teaser video, research sections, and PDF presentation needed here. Its flexible media blocks suit a project whose strongest material is an actual demonstration. The upstream [README](https://github.com/eliahuhorwitz/Academic-project-page-template#readme) describes its reusable sections.

| Candidate | Decision |
| --- | --- |
| [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template) | Selected: reusable research and media sections with responsive styling. |
| [Nerfies project page](https://github.com/nerfies/nerfies.github.io) | Useful visual precedent and an acknowledged source of the selected template; its paper-specific interactions are unnecessary here. |
| [kmranrg academic template](https://github.com/kmranrg/academic-project-page-template) | Viable simpler alternative; keep one template source for this implementation. |

### Exact adaptation plan

Preserve the recognizable centered project introduction and wide teaser layout. Customize the colors, typography, real media, and research content to create a distinctive Hertz page.

| Upstream area | Local destination | Required adaptation |
| --- | --- | --- |
| Title group and `publication-links` in `index.html` | `Hero.astro` | English headline and three real actions; remove sample authors and venue. |
| `hero teaser` block | `Hero.astro` | Existing 12-second preview below the centered title. |
| Abstract/content section | `Method.astro` | Replace placeholder text with the exact overview and module explanations in §6. |
| Video presentation | `Demo.astro` | Local full MP4 with native controls. |
| Poster PDF area | `MethodFigure.astro` | Large diagram preview, direct PDF link, expandable local PDF viewer. |
| Footer | `Footer.astro` | Project identity and template attribution. |

The selected repository was retrieved on 2026-09-09 at commit `d38af1ccae1ce82c3404d2820c4c646afd409f81`. Its resolved source, retrieval date, and CC BY-SA 4.0 notice are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The inspected upstream `index.html`, `static/css/index.css`, and README are preserved in `design/vendor/academic-project-page-template/` for traceability.

Port the relevant markup and style rules into the existing components and `src/styles/template.css`. Implement the small set of needed container, column, link, and button rules there; use `src/styles/global.css` for the Hertz overrides, loaded second. Do not load the complete Bulma framework or the old CSS alongside conflicting copied rules. This is a concrete template adaptation, not just a link to an inspiration page.

Remove unused carousel, slider, More Works, BibTeX, sample authors, sample research, stock media, favicon, and publication metadata. Do not import jQuery, the Adobe PDF SDK, analytics, remote fonts, or icon packages. Native video and PDF rendering cover this page.

### Attribution

The selected template declares **CC BY-SA 4.0** and credits Nerfies. Preserve that attribution and identify the local changes in `THIRD_PARTY_NOTICES.md`. Use this footer text, with links:

> Website design adapted from Academic Project Page Template and Nerfies. Adapted website design: CC BY-SA 4.0.

Link the template name to its repository, Nerfies to [its project page](https://nerfies.github.io/), and the license to [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Keep upstream notices for any retained third-party code. This statement concerns the website design; do not assign that license to model code, weights, videos, or reports.

## 3. Technology and project structure

| Area | Fixed choice |
| --- | --- |
| Framework | Existing Astro 5.0.0, static output |
| Language | TypeScript 5.9.3 and small native browser scripts |
| Styles | Adapted template CSS, CSS variables, Grid and Flexbox |
| Fonts | System sans-serif and monospace; no remote requests |
| Media | Local H.264 MP4, JPEG poster, PNG diagram, local PDF |
| Dependency management | npm; preserve the existing lockfile |
| Interface | English only, `<html lang="en">` |
| Preview | Root path `/`; production origin and Pages base path remain unconfigured |

Work inside `seetapsych-hertz-web/`. Reuse the project and components. Do not scaffold a second app or add React, Vue, Next.js, Tailwind, GSAP, or Three.js.

```text
src/
  pages/index.astro
  layouts/BaseLayout.astro
  components/
    Header.astro
    Hero.astro
    Demo.astro
    Method.astro
    MethodFigure.astro             # add
    Results.astro
    Resources.astro
    Footer.astro
  data/site.ts
  styles/template.css             # add: adapted rules
  styles/global.css
public/media/
public/downloads/
design/vendor/academic-project-page-template/  # add: source snapshot
design/asset-manifest.json
THIRD_PARTY_NOTICES.md             # add during template import
README.md
docs/VERIFICATION.md
```

`BaseLayout.astro` owns metadata, document language, and stylesheet order. `index.astro` assembles the sections. Put all repeated copy, numbers, asset paths, and external URLs in `src/data/site.ts`.

Existing commands:

```bash
npm ci
npm run dev -- --port 4322
npm run build
npm run preview -- --port 4322
```

Run development and production preview at different times if using the same port. `npm run check` currently runs the same Astro build as `npm run build`; it is not a separate type checker. Do not describe a successful build as complete browser verification.

## 4. Source assets and diagram preparation

### Existing assets

| Path | Use |
| --- | --- |
| `public/media/hero-preview.mp4` | 12-second silent preview, 856 × 480, derived from seconds 8–20 of the recording |
| `public/media/demo-full.mp4` | Full silent recording, approximately 25 seconds, 856 × 480 |
| `public/media/demo-poster.jpg` | Real recording frame for both players |
| `public/downloads/tinyhr-technical-report.pdf` | Current 7-page English technical report |

Preserve the entire video frame with `aspect-ratio: 856 / 480` and `object-fit: contain`. Do not redraw waveforms or readings. The relationship between the two HR readouts has not been verified; do not label them as prediction versus ground truth.

### Diagram deliverables — generated

The editable source is `design/source/tinyhr-flowchart-web.pptx`, a one-slide English diagram. The generated outputs are:

| File | Verified output |
| --- | --- |
| `public/downloads/tinyhr-flowchart.pdf` | One-page, complete diagram PDF; 1584 × 630 points. |
| `public/media/tinyhr-flowchart.png` | Complete diagram PNG rendered at 3200 × 1273 pixels on a white background. |

**Correct the export canvas first.** The source slide is 20 × 11.25 inches, while its diagram group starts at x = -0.283 inches and spans approximately 21.491 inches. A direct slide export clips both edges, including part of HR post-processing. This was observed in a temporary review export; that clipped export is not a production asset.

The generated working copy at `design/source/tinyhr-flowchart-web.pptx` preserves the native connector relationships that produce the residual U-shaped path. Its canvas is extended to 22 × 11.25 inches and the complete diagram group is moved as a unit to retain approximately 0.25-inch side margins. The decorative slide number is removed by the final PDF crop. Full-width parentheses in the supplied slide labels are replaced with ASCII parentheses only to avoid renderer glyph substitution. No scientific labels, panels, or arrows are redrawn. The original source remains unchanged and the export relationship is recorded in the asset manifest.

Use a local PowerPoint-compatible renderer for the slide export and a PDF renderer for the PNG. Do not rename a PPTX extension to PDF, use an operating-system screenshot, redraw the diagram with generated imagery, or substitute a generic three-box diagram. Inspect both exports for font substitution, cropped labels, missing arrows, and unreadable text before use. Check the input-size parentheses for missing glyphs and confirm the entire final Heart Rate block is present.

Preserve the editable source file. Update `design/asset-manifest.json` with export dimensions, byte sizes, SHA-256 hashes, and the source relationship. The report's existing public copy matches the supplied PDF:

```text
Technical report SHA-256:
a8ff7c929bf49cee620e80b6d3de5041599e1602ea0fbffaf303cfa1137199ef

Source slide SHA-256:
295d1ed4634688a8774b5c61d0106e6495da3c5d24cb9b78d22f7b19a60d1f8a
```

### Source differences and fixed handling

The supplied diagram and the report's written description differ in several details. The report's own embedded figure also retains some conflicting labels. Preserve the supplied diagram's scientific content, and use the report's written description for the method prose. This precedence is an editorial choice for this page, not proof that the current GitHub implementation matches either document.

| Detail | Supplied slide | Technical report | Website prose |
| --- | --- | --- | --- |
| Temporal modeling | Hierarchical modeling / TPT labels | One MTF stage, without hierarchical temporal downsampling, pp. 2–3 | Single convolutional MTF stage |
| Appearance branch | RGB frames shown in the stem | Final model uses the difference branch only, pp. 1–2 | Difference features; no active RGB appearance branch |
| Spatial embedding | Overlapping-patch label | Kernel 4, stride 4, from 32 × 32 to 8 × 8, p. 2 | Give the report's explicit kernel and stride |
| Prediction head | MLP projection label | Two pointwise convolutional layers, p. 3 | Pointwise convolutional waveform predictor |
| Filtering | Detrending and first-order Butterworth | Second-order Butterworth, 0.75–2.5 Hz, p. 6 | Second-order filter; no added detrending claim |

Use the exact caption and disclosure in §6.4 so these differences are visible to readers. Do not silently edit the source diagram or claim the documents agree. This does not block the English layout or original diagram export.

## 5. Visual design

**Centered research headline → large real video → light research sections.** The headline and moving demonstration should establish the purpose immediately; the architecture figure provides the next strong visual anchor.

```css
:root {
  --ink: #071b24;
  --ink-raised: #0d2832;
  --paper: #f3f5f0;
  --white: #f7faf7;
  --mint: #77e5c8;
  --text: #132b32;
  --muted: #53696e;
  --muted-dark: #b3c9ca;
  --line: #d1dbd5;
  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
  --font-mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
```

- Use a 1200 px maximum content width; side padding 48 px desktop, 32 px tablet, 20 px below 768 px.
- Hero text is centered, max-width 960 px. H1: 80 px / 1.04 desktop, 64 px tablet, `clamp(40px, 10vw, 52px)` mobile; weight 750, letter-spacing -0.045em.
- H2: 40 px / 1.2 desktop, 30 px / 1.25 mobile. Body: 18 px / 1.7 desktop, 16 px / 1.7 mobile. Long prose max-width 760 px.
- Section spacing: 80 px desktop, 48 px mobile. Use typography, whitespace, and fine rules instead of a grid of promotional cards.
- Buttons: minimum height 48 px, horizontal padding 22 px, 10 px radius. Primary: mint on ink; secondary: transparent with visible border.
- Media: 14 px radius; one subtle shadow around the hero video. Diagram sits on white with a 1 px border.
- Only the hero video supplies continuous motion. No particles, simulated BPM counters, moving backgrounds, text entrance sequences, or animated charts.
- A small static mint rule can accent headings. Do not use the old large decorative ribbon.
- Keep content visible without JavaScript. Respect reduced-motion preferences and preserve visible keyboard focus.

Desktop opening target at 1440 × 960: show the navigation, complete headline, actions, and a substantial portion of the video without scrolling. Avoid oversized top whitespace. Use a full-width video below the centered text, not the previous left/right split.

## 6. Page order and exact English copy

Order: **Header → Hero → Demo → How it works → Evaluation → Resources → Footer**.

Anchors: `top`, `demo`, `method`, `results`, `resources`. Use the wording below without adding unverified claims.

### 6.1 Header

Height 64 px, dark background, normal document flow.

- Brand: `SeetaPsych Hertz` → `#top`.
- Navigation: `Demo` → `#demo`; `How it works` → `#method`; `Results` → `#results`; `Resources` → `#resources`.
- Right link: `GitHub ↗` → `https://github.com/seetapsych/seetapsych-hertz`.
- Below 768 px, show only the brand and GitHub link. Do not add a menu drawer.

### 6.2 Hero

Dark background. Desktop top padding 32 px; mobile 24 px. Center title and actions, then place the teaser below them.

| Element | Exact text |
| --- | --- |
| Eyebrow | `SEETAPSYCH HERTZ / TINYHR` |
| H1, two lines | `See the pulse.` / `Through video.` |
| Description | `A lightweight model that recovers pulse waveforms and estimates heart rate from facial video.` |
| Primary action | `Watch demo` → `#demo` |
| Secondary action | `Code on GitHub ↗` → model repository |
| Third action | `Technical report` → report PDF |
| Video label | `RECORDED DEMO · 12-SECOND PREVIEW` |
| Caption | `Facial video, a predicted pulse waveform, and heart-rate estimates in one recorded demonstration.` |

Spacing: eyebrow to H1 16 px; H1 to description 20 px; description to actions 24 px; actions to video 32 px. Video width is 100%, max-width 960 px. Keep labels above or below the video rather than over the face or readings.

Use native `<video controls muted loop playsinline preload="none">`, real poster, width/height attributes, and the existing preview MP4. Attempt playback through a short script only when reduced motion is not requested. Catch a rejected `play()` promise and leave the poster and controls available. No custom player.

On mobile, keep two headline lines. Stack the primary action full-width, followed by the two secondary links on a wrapping row. Preserve the complete video frame.

### 6.3 Demo

Light background. Eyebrow: `SEE IT IN ACTION`. H2: `From facial video to a pulse waveform.`

Body:

> Watch the recorded TinyHR pipeline: facial video, a predicted rPPG waveform, and heart-rate estimates displayed together.

Center the existing full MP4 at max-width 960 px. Use native `controls playsinline preload="none"`, the real poster, and no autoplay or loop.

- Caption: `25-second recording · Silent playback`.
- Note: `This recording illustrates the pipeline. Accuracy is reported separately in the evaluation below.`
- Persistent link: `Download demo video`.
- Video fallback: `Your browser does not support embedded video. Download the demo to watch it locally.`
- Accessible name: `TinyHR recorded demonstration`.

The hero action scrolls here; it does not trigger playback.

### 6.4 How it works

Light background, top rule. Eyebrow: `FROM VIDEO TO PULSE`. H2: `How TinyHR works`.

Render in this order: overview → full-width source diagram → PDF controls and source note → four module explanations → heart-rate calculation → collapsed training details.

**Overview — exact copy:**

> Remote photoplethysmography (rPPG) estimates pulse-related signals from subtle changes in light reflected by facial skin. TinyHR processes a clip of 160 RGB facial frames at 128 × 128 pixels. Its lightweight convolutional pipeline emphasizes differences between neighboring frames, builds compact spatial features, and combines temporal information at multiple scales. The network predicts one rPPG waveform sample per video frame. Heart rate is then calculated from the predicted waveform using filtering and spectral analysis.

Source label: `Method description: TinyHR technical report, pages 1–6.`, linked to `/downloads/tinyhr-technical-report.pdf#page=1`.

**Diagram — implement in MethodFigure.astro:**

- Place the PNG in a semantic `figure`, full content width, natural aspect ratio, `height: auto`. Do not shrink it into a method card.
- Set actual width and height attributes from the export. Use `loading="lazy"` and `decoding="async"`.
- Alt: `Supplied TinyHR pipeline schematic showing facial video input, feature extraction, waveform prediction, and heart-rate post-processing.`
- Image link and separate action: `Open architecture PDF ↗` → `/downloads/tinyhr-flowchart.pdf`.
- Persistent download action: `Download architecture PDF` with the download attribute.
- Caption: `Architecture schematic supplied with the project. Some labels differ from the technical report; the description below follows the report.`
- A closed `details` with summary `View PDF inline` contains a same-origin PDF iframe, title `TinyHR architecture PDF`, width 100%, height 560 px, and lazy loading. Prefer assigning its URL on the first expansion; keep the direct PDF link usable without scripts.
- On screens below 768 px, omit the inline viewer and keep the image plus open/download links. Opening the PDF provides zoom for small diagram labels; do not crop the diagram or cause page-wide horizontal scrolling.

Second closed disclosure: `Diagram and report differences`. Exact text:

> The supplied diagram includes hierarchical/TPT labels, an RGB branch, overlapping patches, an MLP projection label, and first-order filtering. The technical report describes a single convolutional MTF stage, difference-only input processing, kernel-4/stride-4 spatial embedding, a pointwise convolutional waveform predictor, and second-order filtering. The diagram is preserved as supplied; the method text follows the report.

**Four modules — exact titles and copy:**

Display two columns on desktop, one on mobile. Use numbered headings and fine separators, with no illustrated substitute for the supplied diagram.

| Number / title | Copy |
| --- | --- |
| `01 / Frame Difference Fusion Stem` | Four neighboring-frame difference maps are concatenated into a 12-channel representation. The final model uses this difference branch without the original RGB appearance branch. A 5 × 5 convolution, batch normalization, ReLU, pooling, and a 3 × 3 convolution produce 16-channel spatial features. |
| `02 / Spatial Patch Embedding` | A frame-wise 2D convolution with kernel size 4 and stride 4 reduces each feature map from 32 × 32 to 8 × 8, with 32 channels. Batch normalization and ReLU follow. The temporal dimension is preserved. |
| `03 / Multi-scale Temporal Feature Block` | A single convolutional stage enhances spatial features and pools them into a temporal sequence. Four branches combine information from different temporal offsets, followed by pointwise fusion and a temporal feed-forward network. Residual connections retain the input features; no hierarchical temporal downsampling is used. |
| `04 / Waveform Predictor` | Spatial global average pooling and two pointwise convolutional layers produce a single-channel rPPG waveform. The output contains one value per input frame: 160 samples for a 160-frame clip. Heart rate is computed after waveform prediction. |

Module source label: `Architecture details: technical report, pages 1–3.`

**Heart-rate calculation — exact copy:**

H3: `From waveform to heart rate`.

> During inference, a second-order Butterworth band-pass filter retains frequencies from 0.75 to 2.5 Hz. Welch's method estimates the power spectral density of the filtered waveform. The dominant frequency is converted to beats per minute.

Formula, rendered as text with semantic subscripts if needed:

`Heart rate (BPM) = 60 × dominant frequency (Hz)`

Source: `Inference procedure: technical report, page 6.` → report PDF `#page=6`.

**Closed training disclosure:**

Summary: `Training objectives`.

> Training combines a negative Pearson correlation objective for waveform agreement, a frequency-domain cross-entropy objective, and a KL-divergence objective for heart-rate distributions. The reported weights are 0.2, 1.0, and 1.0, respectively.

Formula: `L = 0.2 L_time + L_CE + L_KL`.

Source: `Training objectives: technical report, pages 3–6.`

Do not add loss derivations, dataset carousels, or additional training diagrams in this revision.

### 6.5 Evaluation

Light background, top rule. Eyebrow: `EVALUATION`. H2: `Reported performance`.

Use a prominent static `3.88` with unit `BPM` and label `Mean absolute error (MAE)`. Number size: 88 px desktop, 64 px mobile. Beside it, show:

| Label | Value |
| --- | --- |
| Dataset | VIPL-HR V1 |
| Test split | 22 subjects · 485 videos |
| Evaluation protocol | Held-out test set, excluded from training according to the report |
| Source | TinyHR technical report, page 7 |

Fixed note:

> Author-reported performance on the specified test set. This value is not a per-video error bound.

Link: `Read the evaluation details` → `/downloads/tinyhr-technical-report.pdf#page=7`.

Do not infer competitive ranking, general accuracy, or performance on other datasets. These results have not been independently reproduced for the website.

### 6.6 Resources

H2: `Explore the project`.

Body: `Find the source code, setup instructions, and technical documentation.`

Use four full-width resource rows, separated by fine rules.

| Title | Description | Action |
| --- | --- | --- |
| GitHub | Source code, setup, and usage instructions. | `View repository ↗` → model repository |
| Technical report | Architecture, training objectives, and reported evaluation. | `Download report PDF` → local report |
| Architecture diagram | The supplied pipeline schematic in PDF format. | `Download architecture PDF` → exported diagram |
| Hugging Face | Model distribution and interactive demos are planned. | `Planned` as plain text, no fake link |

No unverified install commands, model license badges, paper links, author lists, or invented citation block.

### 6.7 Footer

Dark background, 32 px vertical padding. Brand: `SeetaPsych Hertz`. Description: `TinyHR · Heart-rate estimation from facial video`. Link: `Back to top ↑`.

Add the template credit from §2 in readable muted text. Do not retain sample identities or metadata from the upstream template.

## 7. Data, metadata, and language rules

Keep the following facts and paths in `src/data/site.ts`; add the new paths only when their exports exist:

```ts
export const site = {
  brand: 'SeetaPsych Hertz',
  model: 'TinyHR',
  repoUrl: 'https://github.com/seetapsych/seetapsych-hertz',
  maeBpm: '3.88',
  dataset: 'VIPL-HR V1',
  subjects: 22,
  videos: 485,
  reportPage: 7,
  inputFrames: 160,
  inputSize: 128,
  reportUrl: '/downloads/tinyhr-technical-report.pdf',
  flowchartPdfUrl: '/downloads/tinyhr-flowchart.pdf',
  flowchartImageUrl: '/media/tinyhr-flowchart.png',
  heroPreviewUrl: '/media/hero-preview.mp4',
  demoUrl: '/media/demo-full.mp4',
  posterUrl: '/media/demo-poster.jpg',
} as const;
```

Page title: `SeetaPsych Hertz | TinyHR Video-Based Heart-Rate Estimation`.

Description: `Explore TinyHR, a lightweight model for estimating pulse waveforms and heart rate from facial video. Watch the demo, inspect the method, and access the code.`

- Set `lang="en"`. Use the page title and description for Open Graph title/description; use `og:type="website"`. Do not invent an official production URL.
- Remove old Chinese strings from authored UI, fallback text, titles, alt text, and accessible names. Browser-native video/PDF controls can follow the viewer's operating-system language; do not build custom controls to translate them.
- Keep visible source-media labels unchanged. The provided diagram's labels are already English.
- Use one H1, section H2s, and semantic nav/main/footer. Add an English `Skip to content` link and meaningful video/PDF accessible names.
- Give keyboard focus a 2 px outline and 4 px offset. New-tab links need `rel="noopener noreferrer"` and an accessible indication that a new tab opens.
- Claim only video-based rPPG and heart-rate estimation. Do not add emotion recognition, HRV, clinical accuracy, parameter counts, inference FPS, or SOTA claims.
- 160 frames is an input length, not a latency or accuracy promise. The recording's FPS is not an inference benchmark.
- Use the report's description of convolutional temporal modeling. Do not market the final model as an attention-based Transformer.
- Keep research sources as evidence, separate from instructions for implementing the site.

## 8. Execution order and acceptance

### Implementation sequence

1. Import the selected template snapshot and record its origin and notices.
2. Export the source slide to PDF and PNG; inspect both and update the asset manifest.
3. Replace the old hero with the centered English template adaptation. Build the page with the real preview early.
4. Translate all remaining interface text using §6–7, implement the method figure and exact report-based explanation, then update resources.
5. Finish mobile layouts, native playback behavior, keyboard focus, and reduced-motion handling.
6. Run the production build and inspect the built site. Record actual evidence and remaining issues.

A three-day iteration can group steps 1–3 on day one, step 4 and responsive work on day two, and verification and fixes on day three. Keep the initial implementation focused on layout, real media, and accurate content.

### Acceptance checklist

| Check | Required evidence |
| --- | --- |
| Template adaptation | Source snapshot and resolved commit documented; migrated title/teaser/content/PDF sections visible; footer attribution present. |
| Full English interface | Authored visible text, metadata, fallback messages, and accessible labels are English; `lang="en"`. |
| Diagram | One-page PDF and complete PNG exist; labels and arrows survive export; shown in How it works with working open/download links. |
| Source accuracy | English modules match report pp. 1–3, filtering matches p. 6, results match p. 7; source-difference note is visible. |
| Desktop, 1440 × 960 | Headline and actions readable, large real preview begins in the opening viewport, no overlap or clipping. |
| Mobile, 390 × 844 and 360 px wide | No page-wide horizontal overflow; buttons wrap; videos and diagram retain their aspect ratios; PDF opens separately. |
| Video | Both files play and pause with native controls; full demo is seekable; blocked autoplay leaves a usable poster and manual controls. |
| Accessibility | Keyboard navigation and focus work; reduced-motion mode avoids attempted autoplay; core text and direct links work without JavaScript. |
| Resources | All anchors, repository links, MP4, PNG, report PDF, and architecture PDF resolve; planned resources remain non-clickable. |
| Build | `npm run build` succeeds; production preview has no introduced console errors or missing local assets. |

Do not add an automated test suite for this static redesign. A build and focused browser checks are sufficient. Inspect the mobile PDF fallback in a browser that does not embed PDFs when available, and state which browser was actually checked.

Update [docs/VERIFICATION.md](docs/VERIFICATION.md) with the implementation date, browser, viewport, checked actions, and actual outcome. Add real release screenshots only when they are available.

**Completion status: the English template adaptation is implemented and checked locally.** Hosting and repository publishing remain separate work.
