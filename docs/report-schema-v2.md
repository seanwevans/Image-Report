# IR report schema v2 — pixel-attributable measurement substrate

Status: **proposal (design only, not yet implemented)**
Supersedes: the v1.x flat `<analysis><bounding_boxes>` + page-level `<hashes>` report.

This document specifies the output format for the pixel-attributable version of
`ir`. It is the accounting layer described in the project roadmap: instead of
reducing a page to a flat list of boxes and hashes, the report becomes a ledger
over the page's ink, where every ink pixel is either **attributed** to a glyph
instance or accounted for as **residual**.

The six design decisions that were open when this was sketched are resolved here
(see [Decisions](#decisions-resolved)); each is annotated inline so they can be
revisited without re-reading the whole document.

---

## 1. Principle

Two rules govern where data lives and what must hold:

1. **Pixel data lives in images; structure lives in XML.** A pixel→instance map
   is encoded as an image sidecar (a label map), never inlined into XML.
   Per-pixel data in XML would be catastrophic (a 600-dpi letter page is ~34M
   pixels).
2. **Conservation.** At the ledger layer, the partition of ink is exact:

   ```
   |attributed| + |residual| == |ink|
   ```

   This is a machine-checkable invariant, not an aspiration. `residual` is
   defined as `ink ∧ ¬attributed` and is the primary deliverable of the tool.

Connected components on the binarized ink mask form the **base partition**:
they cover and partition the foreground by construction, so conservation holds
for free and every ink pixel starts life attributed to some component. Glyph
segmentation is a *refinement* on top of that base, and its errors
(over-/under-segmentation) are recorded as explicit transactions rather than
silent failures (§8).

---

## 2. The report is a bundle

A single page produces one XML report plus a set of image sidecars that share
its stem. For an input `page001.png`:

| File | Format | Role |
|------|--------|------|
| `page001.ir.xml` | XML | Report: metadata, ledger, group tables, glyph table, symbol dictionary |
| `page001.ir.ink.png` | 1-bit grayscale PNG | The ink mask — the conservation **denominator** |
| `page001.ir.labels.cc.png` | 16-bit grayscale PNG | pixel → connected-component id (base partition) |
| `page001.ir.labels.glyph.png` | 16-bit grayscale PNG | pixel → glyph-instance id (**ledger layer**) |
| `page001.ir.labels.word.png` | 16-bit grayscale PNG | pixel → word id |
| `page001.ir.labels.line.png` | 16-bit grayscale PNG | pixel → line id |
| `page001.ir.residual.png` | 1-bit grayscale PNG | `ink ∧ ¬attributed` at the ledger layer |

> **Decision 1 — one label map per layer.** Each hierarchy layer gets its own
> pixel-exact label map rather than expressing words/lines only as XML boxes.
> This costs a few extra files per page but keeps every layer pixel-attributable
> and lets the conservation check run at any layer. Layers that a given run does
> not compute are simply omitted (the XML `<glyphs>`/`<words>`/`<lines>` element
> for that layer is absent, and no sidecar is written).

> **Decision 4 — residual mask is always emitted.** Even when the residual
> fraction is ~0, `page001.ir.residual.png` is written (it compresses to almost
> nothing when empty). "Sometimes present" is worse for downstream tooling than
> "always present, sometimes blank," and the whole point of the tool is that the
> residual is inspectable.

All sidecar paths in the XML are stored **relative to the XML file** so a report
bundle can be moved as a unit.

---

## 3. Label-map encoding

- **Single-channel 16-bit grayscale PNG**, lossless (standard PNG `zlib`
  deflate). PNG gives lossless compression tuned for exactly this kind of
  large-flat-region integer data.
- Pixel value `0` = **unattributed at this layer**. Background paper and
  residual ink both read as `0`; they are disambiguated with the ink mask:

  ```
  residual = ink AND (labels == 0)
  ```

- Pixel value `k ≥ 1` = instance id `k`, matching the `id` attribute of the
  corresponding record in the XML table for that layer.
- **Id assignment is deterministic:** ids are assigned in reading order
  (bounding box sorted top→bottom, then left→right). Re-running the pipeline on
  the same input produces byte-identical label maps.

> **Decision 2 — 16-bit ceiling, with a documented escape hatch.** 16-bit
> grayscale gives 65,535 non-zero ids per layer, which comfortably exceeds the
> glyph/component count of any realistic page (dense text is a few thousand
> glyphs). The pipeline **should assert** that a layer's id count stays below
> 65,536 and fail loudly otherwise. If a future use case genuinely needs more
> (e.g. a merged multi-page atlas), the escape hatch is to store the label map
> as an RGBA PNG with the id packed little-endian across the four 8-bit channels
> (`id = R | G<<8 | B<<16 | A<<24`), giving 2³²−1 ids. This packing is
> **specified but intentionally not implemented** in the first version; the
> reader/writer should carry a single `channels` field so the format can grow
> into it without a version bump.

The XML records which encoding a map uses:

```xml
<labels layer="glyph" file="page001.ir.labels.glyph.png"
        bit_depth="16" channels="1" max_id="1234"/>
```

---

## 4. Conservation ledger

The ledger is the heart of the report — three numbers and two masks:

```xml
<conservation unit="pixels" ledger_layer="glyph">
  <ink total="1048576" mask="page001.ir.ink.png"
       threshold="sauvola" window="25" k="0.2"/>
  <attributed total="1041200" fraction="0.99297"/>
  <residual   total="7376"    fraction="0.00703"
              mask="page001.ir.residual.png"/>
</conservation>
```

- `ink/@total` = `|ink|`, the count of foreground pixels in the ink mask.
- `attributed/@total` = number of ink pixels with a non-zero id in the ledger
  layer's label map.
- `residual/@total` = `ink − attributed`, and equals the set count of
  `residual.png`.
- **Invariant:** `attributed.total + residual.total == ink.total`. A validator
  asserts this exactly; a mismatch is a bug in the pipeline, not a rounding
  artifact.

> **Decision 6 — Sauvola with exposed parameters.** The binarization moves from
> global Otsu to **Sauvola local adaptive thresholding**, because global Otsu
> fails on uneven scanner illumination and on near-blank pages (where it
> amplifies paper grain into "ink"). Defaults: window `25`, `k = 0.2`, exposed
> as CLI flags `--threshold {sauvola,otsu}`, `--sauvola-window`, `--sauvola-k`
> (Otsu retained as an opt-in for reproducing v1 behavior). The chosen method
> and its parameters are recorded on the `<ink>` element so a report is
> self-describing.

This section replaces v1's `calculate_spectral_analysis`, which computed the
foreground mask and then discarded it in favor of row/column counts. The mask is
now kept as the ledger denominator. (Row/column projection profiles, if still
wanted, can remain as an optional descriptor on line/word groups using the RLE
encoding already in `utils.rle_encode`.)

---

## 5. Glyph table

```xml
<glyphs layer="glyph" count="1234"
        label_map="page001.ir.labels.glyph.png">
  <glyph id="42" parent_word="7" parent_line="2" cluster="17">
    <box x1="611" y1="288" x2="629" y2="312" width="18" height="24"/>
    <mask ref="labels.glyph.png#42" area="118"/>
    <descriptors>
      <hash type="skeleton">3f10114058f</hash>
      <hash type="zoning">0a4c2b19</hash>
      <hash type="junction">3d22101130370002</hash>
      <hash type="stroke_direction">1112201340</hash>
      <hash type="contour">1a03b0…</hash>
    </descriptors>
    <hypothesis char="e" confidence="0.94" source="cluster"/>
  </glyph>
  <!-- … -->
</glyphs>
```

- **Mask is a reference, not inline pixels.** `ref="labels.glyph.png#42"` names
  the label map and the id within it; the label map is the single source of
  truth for which pixels belong to the glyph. `area` is cached for convenience
  and cross-checking.

  > **Decision 3 — reference masks by default; inline RLE optional.** The
  > default keeps XML small. An inline form is available for exporting a glyph
  > as a self-contained record:
  >
  > ```xml
  > <mask encoding="rle-rows" origin="611,288" width="18" height="24" area="118">
  >   0:3 1:12 0:3 …
  > </mask>
  > ```
  >
  > It reuses the `value:run` RLE codec (`utils.rle_encode`/`rle_decode`),
  > scanning the glyph's cropped bounding box row-major. Emitting it is gated by
  > a flag (e.g. `--inline-glyph-masks`); it is off by default.

- **Descriptors** are the structural hashes from `hashing_config.py`
  (`skeleton`, `zoning`, `junction`, `stroke_direction`, `contour`, …), now fed
  **glyph crops** — the per-glyph input their docstrings always described.
  Computed on a full page they are semantic mush; on a glyph crop they are
  discriminative. Which descriptors to emit is configurable via the existing
  `--hashes` selector.
- **`cluster`** links the glyph to a symbol-dictionary entry (§6).
- **`parent_word` / `parent_line`** encode the containment hierarchy (§7).

---

## 6. Symbol dictionary — recognize once, propagate

Descriptor hashes bucket identical / near-identical glyph instances. Each bucket
(cluster) is recognized once and the result propagates to every instance. This
is the JBIG2 text-region symbol-dictionary trick: pixel-exact glyph-instance
extraction followed by clustering.

```xml
<symbols count="88" method="descriptor-hash">
  <symbol cluster="17" exemplar_glyph="42" instances="53"
          char="e" confidence="0.94">
    <descriptors>
      <hash type="skeleton">3f10114058f</hash>
    </descriptors>
  </symbol>
  <!-- … -->
</symbols>
```

- `exemplar_glyph` points at the representative instance's `id`.
- `instances` is the cluster size (how many glyph records carry
  `cluster="17"`).
- The character hypothesis lives here and is copied down to instances with
  `source="cluster"`; an instance may override with `source="instance"` if it is
  re-recognized individually.

Clustering thresholds and the descriptor(s) used for bucketing are recorded on
`<symbols>` (`method`) so results are reproducible. The recognition model itself
is out of scope for this document — the schema only needs somewhere to write its
output, which is here.

---

## 7. Hierarchy

Nested structure is **kept as containment**, which is what replaces NMS. NMS
treated the dot of an `i` inside its stroke's region, or a glyph inside a word
inside a line, as redundancy to suppress; here it is hierarchy to record.

Groups are stored as **flat tables per layer with parent references** (rather
than deeply nested XML), which scales better and mirrors the one-label-map-per-
layer file layout:

```xml
<lines count="45" label_map="page001.ir.labels.line.png">
  <line id="2"><box x1="96" y1="280" x2="1490" y2="320"/></line>
  <!-- … -->
</lines>

<words count="320" label_map="page001.ir.labels.word.png">
  <word id="7" parent_line="2"><box x1="604" y1="286" x2="712" y2="316"/></word>
  <!-- … -->
</words>
```

- Each glyph references its `parent_word`; each word references its
  `parent_line`. Ids are per-layer (a `word` id and a `glyph` id may both be
  `7`; they are disambiguated by layer).
- Containment is expected but **not required** to be strict — a glyph whose
  pixels are not fully inside its parent word's box is still valid (touching
  characters, diacritics). The relationship is assignment, not geometric
  nesting.

---

## 8. Segmentation as auditable transactions

Connected components on the ink mask are the base ledger (`labels.cc.png`): a
true partition of foreground pixels, so conservation holds by construction. The
glyph layer is a refinement that splits touching glyphs and merges broken ones.
Because the base partition already accounts for every ink pixel, segmentation
errors become explicit, auditable transactions against the ledger instead of
silent failures:

```xml
<segmentation base="cc" refined="glyph">
  <splits count="12"/>   <!-- one CC became several glyphs (touching characters) -->
  <merges count="8"/>    <!-- several CCs became one glyph (i, j, ï, broken strokes) -->
</segmentation>
```

MSER (or a learned segmenter) is legitimate as the *refinement* engine but not
as the base ledger: MSER regions overlap, duplicate across thresholds, and miss
low-contrast strokes, so they neither cover nor partition the ink. CC is the
base; MSER/learned segmentation refines it.

---

## 9. Full skeleton example

```xml
<?xml version="1.0" encoding="UTF-8"?>
<image_report version="2.0" source_file="page001.png">
  <metadata>
    <!-- unchanged from v1: source_path, filename, dimensions_pixels,
         dimensions_physical (paper-size guess + confidence), … -->
  </metadata>

  <labels>
    <layer name="cc"    file="page001.ir.labels.cc.png"    bit_depth="16" channels="1" max_id="1301"/>
    <layer name="glyph" file="page001.ir.labels.glyph.png" bit_depth="16" channels="1" max_id="1234"/>
    <layer name="word"  file="page001.ir.labels.word.png"  bit_depth="16" channels="1" max_id="320"/>
    <layer name="line"  file="page001.ir.labels.line.png"  bit_depth="16" channels="1" max_id="45"/>
  </labels>

  <conservation unit="pixels" ledger_layer="glyph">
    <ink total="1048576" mask="page001.ir.ink.png"
         threshold="sauvola" window="25" k="0.2"/>
    <attributed total="1041200" fraction="0.99297"/>
    <residual   total="7376"    fraction="0.00703"
                mask="page001.ir.residual.png"/>
  </conservation>

  <segmentation base="cc" refined="glyph">
    <splits count="12"/>
    <merges count="8"/>
  </segmentation>

  <lines  count="45"  label_map="page001.ir.labels.line.png"> … </lines>
  <words  count="320" label_map="page001.ir.labels.word.png"> … </words>
  <glyphs count="1234" label_map="page001.ir.labels.glyph.png"> … </glyphs>

  <symbols count="88" method="descriptor-hash"> … </symbols>
</image_report>
```

---

## 10. Versioning and compatibility

> **Decision 5 — clean v2 break.** The report `version` becomes `2.0`. The v1
> `<analysis><bounding_boxes>` block and the page-level `<hashes>` block are
> **not** carried forward: per-glyph boxes replace the flat box list, and
> per-glyph descriptors replace page-level hashes (which were semantically weak
> anyway). `<metadata>` is preserved as-is, including the corrected paper-size
> confidence.
>
> A consumer can detect the format from the `version` attribute
> (`major == 1` vs `major == 2`) and there is no ambiguous overlap, so a clean
> break is safer than a half-populated hybrid. If a transition period is needed,
> a `--compat v1` flag could additionally emit the old blocks, but that is not
> part of the default v2 output.

---

## 11. Validation checklist

A conforming report satisfies:

1. `conservation`: `attributed.total + residual.total == ink.total`.
2. For the ledger layer, the number of distinct non-zero ids in the label map
   equals `glyphs/@count`, and each id `1..max_id` appears in exactly one glyph
   record.
3. Every `residual.png` set pixel is an ink pixel with label `0`; every ink
   pixel with label `0` is a residual pixel.
4. Every `parent_word` / `parent_line` reference resolves to an existing record
   in the named layer.
5. Every `cluster` on a glyph resolves to a `<symbol>`, and each symbol's
   `instances` equals the number of glyphs pointing at it.
6. Each layer's `max_id < 2^(8 * bit_depth/8 * channels)` (i.e. fits the
   declared encoding).

---

## 12. Build order (once approved)

The changes are staged so each is independently reviewable, in the spirit of the
existing one-PR-per-fix workflow:

1. **`find_bounding_boxes → find_glyph_regions`** — keep the MSER/CC point sets,
   return `(mask, box)` pairs, replace NMS with containment-aware grouping.
   (Resolves the pre-existing MSER test failure at the same time.)
2. **Sauvola binarization + conservation ledger** — keep the ink mask, compute
   `|ink| / |attributed| / |residual|`, emit `ink.png` / `residual.png` and the
   `<conservation>` section.
3. **Label-map writer + `<labels>`/`<glyphs>` tables** — the pixel→instance
   maps and the glyph table with mask references.
4. **Repoint structural hashes at glyph crops** — feed `hashing_config`
   descriptors the per-glyph masks; add them to glyph records.
5. **Symbol dictionary / clustering** — bucket glyphs by descriptor, emit
   `<symbols>`.
6. **Hierarchy layers (word/line) + segmentation transactions** — the remaining
   group tables and the `<segmentation>` audit block.

The recognition model is deliberately *not* in this list; the schema exists to
give it a place to write, and the accounting layer stands on its own without it.
