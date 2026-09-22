# Image generation and editing

Read this reference when text-to-code needs its page reference, when a new complex
raster asset is needed, or when an existing image must be repaired, extended, or
restyled. For page references, first read [visual direction](visual-direction.md).

## Defaults

| Setting | Default |
|---|---|
| API base | `https://openrouter.ai/api/v1/` |
| Model | `inclusionai/ming-image-0.1-design` |
| Size | `2048x2048` |

Set the shared endpoint with `LING_UI_DESIGN_API_BASE` and image defaults with
`LING_UI_DESIGN_IMAGE_*` in the skill-root `.env` or process environment.
`LING_UI_DESIGN_API_KEY` is required, shared with decomposition, and has no default.

Use the default `2048x2048` for page references. Treat it as a design canvas for a
scrollable page, not as proof that the entire page fits in one browser viewport. Keep
sections at a believable scale and let the composition continue below the fold rather
than compressing the whole page. Override the size only when the user specifies a
different reference canvas. Do not upscale a visibly soft result in CSS.

## Generate

`--format` accepts `jpeg` (default), `jpg` (alias), `png`, or `webp`.
Use a matching output extension; JPEG files may use `.jpg` or `.jpeg`.

```bash
LING_UI_DESIGN_SKILL_ROOT=/absolute/path/to/ling-ui-design
cd /absolute/path/to/target-app
python "$LING_UI_DESIGN_SKILL_ROOT/scripts/generate_image.py" \
  --prompt "scrollable museum landing UI, editorial grid, quiet stone gallery, natural desktop scale, no device chrome" \
  --out artifacts/museum-reference.jpg
```

Describe subject, composition, palette, lighting, and intended placement. Do not ask
the image model to render meaningful UI copy; add text in HTML/CSS.

A generated full-page mockup is visual direction, not final layout. Never embed it as
the implementation. Do not regenerate it to fix copy or to re-split sections; do that
in code. For text-only work, skip generation only when existing visual references
already define the design or the user explicitly chooses direct coding.

## Edit or reconstruct

```bash
python "$LING_UI_DESIGN_SKILL_ROOT/scripts/generate_image.py" \
  --image artifacts/occluded-product.png \
  --prompt "restore the complete product behind the foreground label; preserve shape, material, camera angle, and lighting" \
  --format png \
  --out artifacts/product-restored.png
```

Do not regenerate the page because an asset has an overlay. Extract rasters from
the image and background layers with `scripts/crop_elements.py --layers
<decompose-dir> --outdir <crops-dir>`. If the asset is absent from every layer
or every crop is unusable, use the one allowed targeted decomposition retry
first. Edit the asset only if that retry still cannot recover it.

## Inspect and retry

Open every result before integrating it. For a **page mockup**, retry only if it is
unusable as visual direction. For an **asset** edit, revise the prompt to name the
observed defect: wrong crop, incorrect subject count, unwanted lettering, palette
drift, missing negative space, or inconsistent perspective. Do not repeat an identical
failed request more than once; normally stop after three attempts.

The helper sends images in a standard multimodal `image_url` field. It never places
base64 image bytes into a text prompt and never logs the API key.
