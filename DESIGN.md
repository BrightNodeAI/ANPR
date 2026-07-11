# Bright Node — ANPR app design brief

The single source of truth for how the ANPR web app looks. Build the UI with the tokens in
`theme.css`. Never hardcode hex values — reference the CSS variables.

## Brand
- **Company:** Bright Node.
- **Product:** ANPR (video → license-plate recognition).
- **Tagline:** DATA · VISION · PYTHON.
- **Logo:** `logo-mark.svg` (icon only — use in the app header + favicon) and `logo-full.svg`
  (mark + wordmark + tagline — use on the landing/hero or an empty state). The mark is six blue
  nodes around a glowing gold center. Wordmark is "Bright" (slate) + "Node" (blue), sentence case.
- **Voice:** clear, professional, sentence case everywhere. No exclamation marks in UI copy.

## Theme — "Bright Light"
Clean white surfaces, brand **blue** for structure and primary actions, brand **gold** as the
focal accent used sparingly (logo, hero metric, active detection, progress highlight).

| Role | Token | Value |
|------|-------|-------|
| Primary blue | `--bn-accent` | `#2F6BE0` |
| Blue hover | `--bn-accent-hover` | `#245BD0` |
| Blue tint | `--bn-accent-soft` | `#E9EFFD` |
| Gold accent | `--bn-gold` | `#F5A623` |
| Gold strong (text) | `--bn-gold-strong` | `#E08E0B` |
| Page bg | `--bn-page` | `#F7F8FA` |
| Card | `--bn-surface` | `#FFFFFF` |
| Inset/metric | `--bn-surface-soft` | `#F3F5F8` |
| Border | `--bn-border` | `#E6E8EC` |
| Text | `--bn-text` | `#14161A` |
| Text 2 | `--bn-text-2` | `#5B6470` |

**Gold discipline:** gold is a spotlight, not a fill. Use it for the hero "plates found" number,
the progress bar, and the logo — not for large areas, backgrounds, or every button.

## Typography
- **Fonts:** Inter (UI), JetBrains Mono (plate numbers only) — loaded via Google Fonts in `theme.css`.
- **Scale runs ~20% larger than a typical UI baseline** (brand preference): body 17px, h1 26px,
  h2 19px, metric values 29px, table cells 16px, chips/labels 13–14px. Weights: 400 and 500 only.
- Sentence case for all labels, buttons, headings. Plate numbers are monospace.

## Region chips (semantic, consistent everywhere)
- UK → `bn-chip uk` (blue) · USA → `bn-chip us` (green) · Hong Kong → `bn-chip hk` (coral) ·
  Universal → `bn-chip univ` (violet).

## Confidence colors
- `bn-conf high` (green ≥85%) · `bn-conf mid` (amber 70–84%) · `bn-conf low` (red <70%).

## Screens & components
1. **Header** — `logo-mark.svg` + "BrightNode" wordmark, minimal nav (Analyze / History / Docs).
   No "CPU" or dev badges.
2. **Upload** — dashed dropzone (`bn-dropzone`). Empty state: "Drag & drop a video, or browse" with
   subtext "Accepts MP4, MOV, AVI, MKV, WebM and more". **Accept all common video formats**, not just
   MP4 (backend reads via OpenCV/ffmpeg — support depends on the server's ffmpeg codecs). Once a file
   is chosen, show its name + resolution + duration.
3. **Region selector** — segmented control (`bn-seg`), options USA / UK / Hong Kong / Universal;
   Universal is the sensible default.
4. **Analyze** — primary blue button (`bn-btn`). Kicks off the async job.
5. **Progress** — `bn-progress` (gold fill) with "Analyzing… frame N of M" while the job runs.
6. **Results** — metric cards (`bn-metric`; the "plates found" value uses `.gold`), then the
   detected-plates table (`bn-table`) with columns Plate / Region / Confidence / First seen,
   plus a strip of **annotated-frame thumbnails** (real snapshots from the video with the plate
   boxed + labeled — labeled "Annotated frames") and a download button for the annotated video.

## Rules
- Reference `theme.css` variables/classes; don't invent new colors.
- Keep it flat and clean — the boldness comes from the blue/gold palette and the larger type, not
  from shadows or gradients (the logo's gold glow is the one intentional gradient).
- Sentence case, contractions, no "successfully"/"please" in UI copy.
