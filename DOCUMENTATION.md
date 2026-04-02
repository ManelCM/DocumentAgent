# DocumentAgent — Complete Technical Documentation

> **Purpose:** End-to-end document understanding pipeline that takes a PDF or image, detects layout regions, assigns reading order, processes each region with the right specialist (OCR, VLM, formula extraction, table parsing), resolves cross-references, and produces a structured JSON output ready for RAG, QA, or any downstream task.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Data Model — `types.py`](#2-data-model--typespy)
3. [Document Loading — `io.py`](#3-document-loading--iopy)
4. [Layout Detection — `layout.py`](#4-layout-detection--layoutpy)
5. [Reading Order — `order.py`](#5-reading-order--orderpy)
6. [Hierarchy & Inline Math — `hierarchy.py`](#6-hierarchy--inline-math--hierarchypy)
7. [VLM Prompts & Config — `vlm_openai.py`](#7-vlm-prompts--config--vlm_openapy)
8. [Graph Nodes — `nodes.py`](#8-graph-nodes--nodespy)
9. [LangGraph Pipeline — `graph.py`](#9-langgraph-pipeline--graphpy)
10. [CLI Entry Point — `cli.py`](#10-cli-entry-point--clipy)
11. [Techniques Reference](#11-techniques-reference)
12. [Output Schema](#12-output-schema)

---

## 1. Architecture Overview

The agent is a **linear-then-parallel LangGraph pipeline**. Each page of a document goes through a fixed sequence of stages, then fans out to specialist nodes that run in parallel, and finally converges back to produce the output.

```
PDF / Image
     │
     ▼
┌─────────────────────┐
│   load_document     │  Convert PDF pages → numpy BGR images
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   detect_layout     │  PaddleOCR PP-DocLayout_plus-L
│                     │  → bounding boxes + semantic labels
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   reading_order     │  LayoutReader (microsoft/layoutreader)
│                     │  or column-detection heuristic fallback
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│      hierarchy      │  IoU containment + same-line inline math
│                     │  stitching → MIXED blocks
└──────────┬──────────┘
           │
    ┌──────┴──────────────────────────────────────────┐
    │  Fan-out (all run in parallel)                  │
    │                                                  │
    ▼          ▼          ▼       ▼      ▼    ▼    ▼  │
 text_    mixed_    image_   chart_  formula_ table_ other_
spec.     spec.     spec.    spec.   spec.    spec.  spec.
    │                                                  │
    └──────────────────┬───────────────────────────────┘
                       │  Fan-in
                       ▼
          ┌─────────────────────┐
          │  reduce_specialists │  Merge all payloads into blocks
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  cross_reference    │  Regex scan: Figure X, Table X,
          │                     │  Eq. X, [N] citations
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  association_node   │  Caption↔figure linking,
          │                     │  full-page image detection,
          │                     │  embedded formula linking
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │      aggregate      │  Sort by reading order →
          │                     │  final JSON output
          └─────────────────────┘
```

**Key design decisions:**

| Decision | Reason |
|---|---|
| LangGraph `StateGraph` | Declarative graph wiring; automatic parallel fan-out/fan-in; fallback sequential mode if LangGraph is missing |
| Parallel specialists | Each block type is independent; ThreadPoolExecutor inside each specialist node adds a second level of parallelism (across blocks of the same type) |
| Parallel nodes return delta only | Each specialist returns only the keys it modifies, avoiding `InvalidUpdateError` when LangGraph merges concurrent writes |
| Graceful fallback chain | Every specialist falls back (PaddleOCR → Tesseract → empty string; VLM → OCR → empty) so the pipeline never crashes on missing services |

---

## 2. Data Model — `types.py`

### `BlockType` (Enum)

Semantic classification of a detected region. Every block in the document gets exactly one type.

| Value | Meaning | Specialist |
|---|---|---|
| `text` | Body text, paragraphs, references, section headings, equation labels | PaddleOCR text recognition |
| `formula` | Standalone mathematical formula/equation | VLM → LaTeX |
| `image` | Photo or generic image | VLM → description |
| `chart` | Graph, plot, or data visualisation | VLM → structured semantics |
| `figure` | Figure container (may contain image + caption) | VLM → description |
| `caption` | Caption or title directly attached to a figure/table | PaddleOCR text recognition |
| `table` | Tabular data structure | VLM → structured rows/columns |
| `header` | Page header / running head | PaddleOCR text recognition |
| `footer` | Page footer, footnote, page number | PaddleOCR text recognition |
| `mixed` | Inline math stitched together with surrounding text | VLM (handles both prose + LaTeX) |
| `other` | Unrecognised label — OCR'd as a fallback | PaddleOCR text recognition |

---

### `BoundingBox` (dataclass)

Pixel-space rectangle on a page image, in **BGR numpy coordinate space** (origin top-left).

```
(x1, y1) ───────────────┐
    │                   │
    │    document block  │
    │                   │
    └─────────────── (x2, y2)
```

**Properties (all computed, no storage overhead):**

| Property | Formula | Use |
|---|---|---|
| `width` | `max(0, x2 - x1)` | Clamped so inverted boxes give 0, not negative |
| `height` | `max(0, y2 - y1)` | Same |
| `area` | `width × height` | Used for IoU, containment, and page-coverage ratio |
| `cx` | `(x1 + x2) / 2` | Block centre X — used for column assignment and proximity |
| `cy` | `(y1 + y2) / 2` | Block centre Y — used for same-line detection and sorting |

---

### `DocumentBlock` (dataclass)

The central data unit. One instance per detected region on a page.

| Field | Type | Description |
|---|---|---|
| `block_id` | `str` | Unique ID (`"b1"`, `"b2"`, …, or `"mixed_<hex>"` for stitched blocks) |
| `page_index` | `int` | 0-based page number |
| `bbox` | `BoundingBox` | Pixel coordinates on the page image |
| `block_type` | `BlockType` | Semantic type (drives specialist routing) |
| `detector_label` | `str` | Raw label from PaddleOCR (e.g. `"paragraph_title"`, `"formula"`) — preserved for debugging and reading order hints |
| `confidence` | `float` | Detector confidence [0, 1] |
| `reading_order` | `int` | 0-based rank assigned by reading order stage (-1 = unprocessed) |
| `parent_id` | `str \| None` | Block ID of the containing block (set by hierarchy) |
| `child_ids` | `List[str]` | IDs of contained or stitched child blocks |
| `relations` | `Dict` | Semantic links: `"describes"`, `"captions"`, `"embedded_formulas"`, `"references"`, `"is_page_image"`, etc. |
| `payload` | `Dict` | Output from the specialist: OCR text, LaTeX, VLM JSON, table rows, etc. |
| `line_group_id` | `str \| None` | ID of the MIXED parent block when this block is part of an inline math group |
| `skip_specialist` | `bool` | `True` if this block has been absorbed into a MIXED parent and should not be dispatched again |

**`as_dict()`** serialises the block to a JSON-compatible dict (the format written to the output file).

---

### `AgentState` (TypedDict)

The shared state object that flows through the LangGraph pipeline. Every node receives the current state and returns a partial update (a dict containing only the keys it modifies).

| Key | Type | Set by |
|---|---|---|
| `input_path` | `str` | CLI / caller |
| `run_id` | `str` | `load_document` |
| `pages` | `List[np.ndarray]` | `load_document` |
| `page_sizes` | `List[Dict]` | `load_document` |
| `blocks` | `List[DocumentBlock]` | `detect_layout`, then mutated by subsequent nodes |
| `warnings` | `List[str]` | Any node that encounters a non-fatal issue |
| `text_updates` | `Dict[block_id → payload]` | `text_specialist` |
| `mixed_updates` | `Dict[block_id → payload]` | `mixed_specialist` |
| `image_updates` | `Dict[block_id → payload]` | `image_specialist` |
| `chart_updates` | `Dict[block_id → payload]` | `chart_specialist` |
| `formula_updates` | `Dict[block_id → payload]` | `formula_specialist` |
| `table_updates` | `Dict[block_id → payload]` | `table_specialist` |
| `other_updates` | `Dict[block_id → payload]` | `other_specialist` |
| `*_warnings` | `List[str]` | Corresponding specialist |
| `output` | `Dict` | `aggregate` |
| `status` | `Literal["init","running","done","error"]` | `load_document`, `aggregate` |

> **Why separate update dicts instead of writing directly to `blocks`?**
> LangGraph runs parallel nodes in separate threads. Mutating a shared list concurrently is not safe. Each specialist writes to its own isolated key; `reduce_specialists` merges everything back into `blocks` in a single-threaded step.

---

## 3. Document Loading — `io.py`

### `load_document_pages(input_path, dpi=200)`

Converts a PDF or image file into a list of numpy BGR arrays (one per page) and a list of page size dicts.

**PDF path** (via PyMuPDF):
1. Opens the PDF with `pymupdf.open()`.
2. For each page, renders it to a `Pixmap` at `dpi=200` (zoom = 200/72 ≈ 2.78×). Higher DPI gives PaddleOCR more pixels to work with — 200 DPI is a practical balance between quality and memory.
3. Converts RGBA → BGR (OpenCV native format) via `_pixmap_to_bgr()`.

**Image path** (JPEG, PNG, etc.):
1. Reads directly with `cv2.imread()` — returns BGR array.
2. Wraps it as a single-page document.

**Returns:**
- `pages`: `List[np.ndarray]` — one BGR image per page
- `sizes`: `List[{"page_index": int, "width": int, "height": int}]`

---

## 4. Layout Detection — `layout.py`

### `map_detector_label(label) → BlockType`

Normalises the raw string label from PaddleOCR's layout detector to a `BlockType`. This is the **only place** where the raw detector vocabulary is mapped to the agent's internal vocabulary — making it easy to support new detector versions by adding entries here.

**Mapping table (PP-DocLayout_plus-L labels):**

| Raw label(s) | Mapped to | Notes |
|---|---|---|
| `text`, `paragraph`, `list`, `reference`, `abstract`, `section`, `reference_content`, `doc_title`, `aside_text` | `TEXT` | All prose content |
| `title`, `heading`, `headline`, `section_title`, `sub_title`, `paragraph_title` | `TEXT` | Kept as TEXT; `detector_label` preserved for reading order heuristics |
| `formula`, `equation`, `inline_formula`, `display_formula` | `FORMULA` | Full math expressions |
| `formula_number` | `TEXT` | Equation labels like "(1)", "(2)" — too small for formula processing |
| `table`, `table_caption` | `TABLE` | Tabular structures |
| `image`, `photo` | `IMAGE` | |
| `chart`, `graph`, `plot` | `CHART` | |
| `figure`, `subfigure` | `FIGURE` | |
| `caption`, `figure_caption`, `table_caption_text`, `figure_title` | `CAPTION` | Figure/table captions |
| `header`, `page_header`, `running_head`, `running_title` | `HEADER` | |
| `footer`, `page_footer`, `footnote`, `page_number`, `number` | `FOOTER` | |
| anything else | `OTHER` | Unknown labels — still OCR'd as fallback |

---

### `detect_layout_blocks(pages, warnings) → List[DocumentBlock]`

**Model used:** `PP-DocLayout_plus-L` via `paddleocr.LayoutDetection()`.

This is a **YOLO-style object detection model** fine-tuned on document layout datasets. It outputs bounding boxes with class labels (the labels in the table above) and confidence scores.

**Algorithm:**
1. Instantiates `LayoutDetection()` once (model is cached on disk at `~/.paddlex/`).
2. For each page image, calls `detector.predict(img)`.
3. Parses the result list: each element has `coordinate` (4 floats), `label` (string), `score` (float).
4. Creates one `DocumentBlock` per detection.

**Fallback behaviour:**
- If PaddleOCR is unavailable → single full-page block of type `TEXT`.
- If a page returns no detections → single full-page block of type `TEXT`.

---

## 5. Reading Order — `order.py`

This module answers: **given a set of bounding boxes on a page, in what order should they be read?**

This is a critical and historically hard problem, especially for:
- Two-column academic papers (must finish left column before starting right)
- Documents with full-width headers above multi-column body
- Mixed layouts (abstracts spanning the full width, body in two columns)

---

### `_is_full_width(block, page_w, threshold=0.65) → bool`

Returns `True` if the block spans ≥ 65% of the page width. Used to identify titles, abstracts, and other full-width elements that must be read before column content.

---

### `_detect_column_boundaries(blocks, page_w) → List[int]`

**Technique: X-projection gap analysis**

Finds vertical column separators by looking for horizontal gaps in block coverage.

**Algorithm:**
1. Filter out full-width blocks first (they would mask column gaps).
2. Build a coverage array `covered[x]` = number of blocks covering pixel column `x`.
3. Scan only the central 20%–80% of the page width (margins are excluded to avoid false positives from uneven margins).
4. Find contiguous zero-runs (gaps) of width ≥ 1.5% of page width.
5. Return the midpoint of each gap as a column separator x-coordinate.

**Example:** A two-column A4 page at 200 DPI (≈1654 px wide) has a gutter of ~50 px at x≈827. The function detects this gap and returns `[827]`.

The 20%–80% restriction ensures that gaps at the edges (where running headers or page numbers might create false gaps) are ignored.

---

### `_assign_column(block, separators) → int`

Returns the 0-based column index for a block. Counts how many separators are to the left of the block's centre X.

Example: separators = `[400, 750]` (three-column layout)
- Block cx=200 → 0 separators to the left → column 0
- Block cx=575 → 1 separator to the left → column 1
- Block cx=800 → 2 separators to the left → column 2

---

### `_heuristic_order(blocks, page_w, page_h) → List[int]`

**The complete column-aware fallback reading order algorithm.**

Returns a list of original block indices in the correct reading order.

**Steps:**
1. **Classify** each block as: HEADER, full-width, or columnar body (FOOTER).
2. **Sort headers** by Y (top-to-bottom) — always first.
3. **Sort full-width blocks** by Y — come after headers, before columns.
4. **Detect column boundaries** using `_detect_column_boundaries`.
5. **Sort body blocks** by `(column_index, cy, cx)` — finishes each column top-to-bottom before moving to the next.
6. **Sort footers** by Y — always last.
7. Concatenate: `headers + full_width + body + footers`.

This correctly handles the most common academic paper layout:
```
[header]
[title — full width]
[abstract — full width]
[left col p1] [right col p1]
[left col p2] [right col p2]
[footer]
```

---

### `_layoutreader_predict(blocks, page_w, page_h) → List[int] | None`

**Technique: LayoutReader (microsoft/layoutreader)**

LayoutReader is a seq2seq transformer trained on the **ReadingBank** dataset (~500,000 document pages with ground-truth reading order annotations). It predicts the reading order as a permutation over the input bounding boxes.

**Input format:**
- All bounding boxes normalised to [0, 1000] range (model's expected scale).
- Encoded as position tokens using the model's tokenizer.

**Output:**
- A sequence of indices (e.g. `"2 0 3 1"`) representing the correct reading order.
- Validated to be a proper permutation of `[0, N-1]`.

**When used:** Primary method. Falls back to heuristic if:
- `transformers` is not installed.
- The model cannot be downloaded.
- The output is not a valid permutation.
- Any exception during inference.

**Why LayoutReader beats heuristics on academic papers:**
The model was trained on real documents and learned that column layouts require finishing one column before starting the next, that figures break columns, that footnotes come last, etc. — without any hard-coded rules.

---

### `_normalize_boxes(blocks, page_w, page_h) → List[List[int]]`

Converts pixel bounding boxes to LayoutReader's [0, 1000] normalised space:
```
x_norm = int(x / page_w * 1000)  # clamped to [0, 1000]
y_norm = int(y / page_h * 1000)
```

---

### `apply_reading_order(blocks, page_sizes) → List[DocumentBlock]`

The public entry point called by `node_reading_order`. Groups blocks by page, attempts LayoutReader, falls back to heuristic, and assigns `block.reading_order` for every block.

---

## 6. Hierarchy & Inline Math — `hierarchy.py`

### Geometric Helpers

#### `_iou(a, b) → float`
Intersection over Union between two blocks.
```
IoU = area(A ∩ B) / area(A ∪ B)
```
Range: [0, 1]. Used to detect when two blocks significantly overlap.

#### `_overlap_ratio(child, parent) → float`
What fraction of `child`'s area is covered by `parent`:
```
overlap_ratio = area(child ∩ parent) / area(child)
```
Unlike IoU, this is asymmetric — it measures containment from the child's perspective. A small formula block can be 100% contained in a large text block even if it only covers 5% of the text block's area.

#### `_same_line(a, b, tolerance=0.5) → bool`
Two blocks are on the same text line if their vertical centres are within `min(height_a, height_b) * 0.5` pixels of each other.

**Why use min height as the tolerance unit?** The smaller block's height is a natural scale for what "same line" means — two 30px-tall blocks are on the same line if their centres differ by ≤ 15px.

#### `_x_adjacent(a, b, gap_ratio=1.5) → bool`
Two blocks are horizontally adjacent if the gap between them is ≤ `min(height_a, height_b) * 1.5`. Again the height is used as a scale-invariant unit.

---

### `_stitch_inline_math(page_blocks) → List[DocumentBlock]`

**The inline math problem:** When a sentence contains an embedded formula (e.g. *"The loss is* $\mathcal{L} = \sum_i w_i$ *across all terms"*), the layout detector produces two or three separate bounding boxes: one TEXT and one FORMULA (possibly more for complex expressions). If each is processed independently, the reconstructed output loses the sentence structure.

**Algorithm:**
1. Find all `FORMULA` blocks on the page that are not already `skip_specialist`.
2. For each formula, find `TEXT` blocks that satisfy both:
   - `_same_line()` → they are vertically aligned (on the same text line)
   - `_x_adjacent()` → they are horizontally close (side-by-side, not separated by a column gap)
3. If partners are found, create a **MIXED block**:
   - New `block_id`: `"mixed_<uuid[:8]>"`.
   - `bbox`: the **union bounding box** of all partners + the formula.
   - `block_type`: `BlockType.MIXED`.
   - `child_ids`: IDs of all absorbed blocks.
4. Mark all partner blocks: `skip_specialist = True`, `line_group_id = mixed_block.block_id`.

The MIXED block is dispatched to `node_mixed_specialist` which sends the merged crop (containing both prose and formula) to a VLM with a prompt that asks for the reconstructed sentence with inline LaTeX.

---

### `_link_captions(page_blocks) → None`

Links each `CAPTION` block to the nearest `FIGURE`, `IMAGE`, `CHART`, or `TABLE` block on the same page, using Euclidean distance between block centres.

Sets:
- `caption.relations["describes"] = figure.block_id`
- `figure.relations["captions"].append(caption.block_id)`

---

### `_find_container(block, candidates, min_overlap=0.80) → DocumentBlock | None`

Finds the tightest block that contains at least 80% of `block`'s area. Used for parent-child hierarchy (e.g. a FIGURE block that contains a CAPTION).

---

### `build_hierarchy(blocks) → List[DocumentBlock]`

The main entry point for the hierarchy stage. For each page:
1. **Containment pass**: assigns `parent_id` and `child_ids` for genuinely nested blocks.
2. **Inline math stitching**: calls `_stitch_inline_math`, which may add MIXED blocks.
3. **Caption linking**: calls `_link_captions`.

Returns all original blocks + newly created MIXED blocks.

---

## 7. VLM Prompts & Config — `vlm_openai.py`

### Prompts

Each block type has a carefully engineered prompt that asks for structured JSON output. Using `response_format: {"type": "json_object"}` guarantees valid JSON even with complex content.

#### `IMAGE_PROMPT`
Requests:
- `summary`: one-sentence description
- `key_elements`: list of visual elements (objects, labels, arrows)
- `visible_text`: any legible text inside the image (exact transcription)
- `spatial_layout`: description of element arrangement
- `confidence`: [0,1]

#### `CHART_PROMPT`
Requests:
- `chart_type`: bar / line / scatter / pie / heatmap / box
- `title`: if visible
- `axes`: x/y labels, units, ranges
- `series`: legend entries
- `trends`: qualitative description
- `extrema`: min/max with coordinates
- `comparisons`: relative statements between series
- `data_table`: markdown table of approximate data points (when readable)
- `takeaway`: single main insight
- `confidence`: [0,1]

**Why `data_table`?** For RAG, a prose description of a chart is hard to query. A markdown table is parseable and can answer quantitative questions.

#### `FORMULA_PROMPT`
Requests:
- `latex`: full LaTeX representation
- `symbols`: list of `{symbol, meaning}` pairs
- `formula_type`: definition / theorem / constraint / loss function / update rule
- `meaning`: plain-English explanation
- `confidence`: [0,1]

#### `MIXED_PROMPT`
For inline math blocks (text + formula merged):
- `full_text`: complete sentence with formula as inline LaTeX `$...$`
- `formula_latex`: just the formula(s)
- `plain_text`: surrounding prose without formula
- `confidence`: [0,1]

#### `TABLE_PROMPT`
- `title`: if visible in crop
- `headers`: column header strings in order
- `rows`: list of `{header: value}` dicts — null for empty cells
- `notes`: footnotes / annotations
- `summary`: one-sentence description
- `confidence`: [0,1]

**Why `rows` as list of dicts?** Makes the table directly queryable in downstream code without parsing a markdown table string.

---

### `OpenAIVLMConfig` (dataclass)

Holds the API key and per-specialist model overrides. Each specialist can use a different model — e.g. use `gpt-4o` for charts (higher quality) and `gpt-4.1-mini` for simple images (cheaper).

All values are read from environment variables so no secrets are hardcoded:

```
OPENAI_API_KEY=sk-...
OPENAI_VLM_IMAGE_MODEL=gpt-4.1-mini
OPENAI_VLM_CHART_MODEL=gpt-4.1-mini
OPENAI_VLM_FORMULA_MODEL=gpt-4.1-mini
OPENAI_VLM_MIXED_MODEL=gpt-4.1-mini
OPENAI_VLM_TABLE_MODEL=gpt-4.1-mini
```

---

### `_run_openai_vision(prompt, model, img, api_key) → str`

The shared VLM call function used by all specialists.

1. Encodes the crop image as a PNG base64 data URL (`_img_to_data_url`).
2. Calls `client.chat.completions.create` with:
   - `temperature=0` → deterministic output
   - `response_format={"type": "json_object"}` → guaranteed valid JSON
   - `image_url.detail="high"` → full resolution analysis (not thumbnail)
3. Extracts the text content from the response.

---

## 8. Graph Nodes — `nodes.py`

### Sequential nodes

#### `node_load_document(state) → delta`
Calls `load_document_pages`. Returns `pages`, `page_sizes`, `run_id`, `warnings`, `status="running"`.

#### `node_detect_layout(state) → delta`
Calls `detect_layout_blocks`. Returns `blocks`, `warnings`.

#### `node_reading_order(state) → delta`
Calls `apply_reading_order(blocks, page_sizes)`. Returns `blocks` with `reading_order` set on all blocks.

#### `node_hierarchy(state) → delta`
Calls `build_hierarchy(blocks)`. Returns `blocks` (possibly augmented with MIXED blocks).

---

### Text extraction — `_extract_text_paddle(crop) → Dict`

The core OCR function with a two-stage fallback:

**Stage 1 — PaddleOCR (`PP-OCRv5` text recognition):**
- Model: PP-OCRv5, state-of-the-art text recognition trained on diverse document datasets.
- API: `ocr.predict(crop)` — returns a result object with `rec_texts` attribute (new API) or a list-of-lists (legacy API). Both formats are handled.
- Advantage over Tesseract: Better handling of fonts, orientations, and non-English characters. Natively supports angle classification (`use_angle_cls=True`).

**Stage 2 — Tesseract OCR (fallback):**
- Triggered if PaddleOCR is unavailable (import fails or returns None).
- Calls `pytesseract.image_to_string(crop)`.
- Returns `{"text": ..., "ocr_engine": "pytesseract_fallback"}`.

**Stage 3 — Empty fallback:**
- If both fail, returns `{"text": "", "ocr_engine": "fallback", "ocr_error": "<message>"}`.
- The pipeline continues — a block with empty text is better than a crashed pipeline.

**Model caching:** `_PADDLE_OCR` is a module-level singleton, loaded once and reused across all calls.

---

### `_process_block_task(task) → Dict`

The dispatcher function executed inside `ThreadPoolExecutor`. Receives a task dict with `block`, `crop`, `cfg`, `kind` and routes to the appropriate processing logic.

| `kind` | Processing |
|---|---|
| `"text"` | `_extract_text_paddle(crop)` |
| `"mixed"` | `analyze_mixed(crop, cfg)` → VLM with MIXED_PROMPT |
| `"image"` | `analyze_image(crop, cfg)` → VLM with IMAGE_PROMPT |
| `"chart"` | `analyze_chart(crop, cfg)` → VLM with CHART_PROMPT |
| `"formula"` | `cv2.cvtColor` (grayscale preprocessing) → `analyze_formula(crop, cfg)` → VLM with FORMULA_PROMPT |
| `"table"` | `analyze_table(crop, cfg)` → VLM with TABLE_PROMPT, fallback to OCR |
| `"other"` | `_extract_text_paddle(crop)` → plain OCR for unknown types |

Every branch has an exception handler that logs a warning and returns a safe fallback payload.

---

### `_run_specialist_kind(state, kind) → (updates, warnings)`

Generic runner for a specialist type:
1. Filters `state["blocks"]` to those matching `_KIND_TO_TYPES[kind]` and not `skip_specialist`.
2. Crops each block from the page image using `_crop_block_image`.
3. Submits all tasks to a `ThreadPoolExecutor` (size: min(4, num_tasks), configurable via `DOCAGENT_MAX_WORKERS`).
4. Collects results and returns `{block_id: payload}` and `[warnings]`.

**Why ThreadPoolExecutor inside each specialist node?** The parallel fan-out at the graph level parallelises across block *types*. The executor inside each node parallelises across *individual blocks* of that type. This two-level parallelism is important for documents with many formulas or many tables.

---

### Specialist nodes

Each specialist node calls `_run_specialist_kind` and returns **only its own delta keys** (not the full state) to avoid LangGraph's `InvalidUpdateError` during concurrent state writes:

```python
def node_text_specialist(state):
    updates, warns = _run_specialist_kind(state, "text")
    return {"text_updates": updates, "text_warnings": warns}

def node_mixed_specialist(state):
    updates, warns = _run_specialist_kind(state, "mixed")
    return {"mixed_updates": updates, "mixed_warnings": warns}  # ← own key, not text_updates
```

`node_mixed_specialist` uses `mixed_updates`/`mixed_warnings` (not `text_updates`) because both `text_specialist` and `mixed_specialist` run in parallel — writing to the same key from two concurrent nodes would be rejected by LangGraph.

---

### `node_reduce_specialists(state) → delta`

**Fan-in step.** Merges all specialist update dicts into `blocks[i].payload`:

```python
for key in ["text_updates", "mixed_updates", "image_updates", ...]:
    for block_id, payload in state.get(key, {}).items():
        by_id[block_id].payload.update(payload)
```

Also aggregates all `*_warnings` lists into the main `warnings` list.

---

### `node_cross_reference(state) → delta`

**Technique: regex-based reference extraction**

Scans the OCR text of every TEXT and MIXED block for standard academic cross-reference patterns:

| Pattern | Regex | Matches |
|---|---|---|
| Figure | `\bfig(?:ure)?\.?\s*(\d+[a-z]?)\b` | "Figure 3", "Fig. 2a", "FIGURE 7" |
| Table | `\btable\.?\s*(\d+[a-z]?)\b` | "Table 5", "TABLE 2b" |
| Equation | `\beq(?:uation)?\.?\s*(\d+[a-z]?)\b` | "Equation 12", "Eq. 4" |
| Section | `\bsec(?:tion)?\.?\s*(\d+[\.\d]*)\b` | "Section 3.2", "Sec. 4" |
| Citation | `\[(\d+)\]` | "[14]", "[3]" |

All matches are stored in `block.relations["references"]` as:
```json
[{"type": "figure", "label": "3", "context": "Figure 3"}, ...]
```

This creates a semantic link graph in the output JSON that downstream RAG systems can use to connect text passages to the figures they reference.

---

### `node_association(state) → delta`

Adds spatial and semantic links between blocks:

1. **Caption → figure linking** (if not already linked by hierarchy):
   - For each CAPTION block, finds the nearest FIGURE/IMAGE/CHART/TABLE on the same page by Euclidean distance between centres.
   - Sets `caption.relations["describes"]` and `figure.relations["captions"]`.

2. **Full-page image detection**:
   - If a figure/image/chart covers ≥ 85% of the page area, it's flagged as `is_page_image: True` with the exact `page_coverage` ratio.
   - Useful for slides, posters, and cover pages.

3. **Embedded formula linking**:
   - For FORMULA blocks whose `parent_id` points to a TEXT or MIXED block, sets bidirectional links: `text.relations["embedded_formulas"]` and `formula.relations["embedded_in"]`.

---

### `node_aggregate(state) → delta`

Produces the final output:
1. Sorts all blocks by `(page_index, reading_order)` — blocks with `reading_order=-1` sort last.
2. Calls `block.as_dict()` on each to get JSON-serialisable dicts.
3. Builds the summary: page count, block count, type histogram.
4. Sets `status="done"`.

---

## 9. LangGraph Pipeline — `graph.py`

### `build_graph()`

Constructs and compiles the `StateGraph`. The graph structure:

```python
# Sequential entry
load_document → detect_layout → reading_order → hierarchy

# Parallel fan-out (all edges from "hierarchy")
hierarchy → text_specialist    → reduce_specialists
hierarchy → mixed_specialist   → reduce_specialists
hierarchy → image_specialist   → reduce_specialists
hierarchy → chart_specialist   → reduce_specialists
hierarchy → formula_specialist → reduce_specialists
hierarchy → table_specialist   → reduce_specialists
hierarchy → other_specialist   → reduce_specialists

# Sequential exit
reduce_specialists → cross_reference → association_node → aggregate → END
```

**Fan-out behaviour:** In LangGraph, multiple `add_edge` calls from one node creates parallel execution. All specialist nodes receive the same frozen state snapshot and execute concurrently. LangGraph waits for all of them before proceeding to `reduce_specialists` (fan-in).

**Fan-in requirement:** Because parallel nodes write to the same state, each specialist node returns **only its own delta** — otherwise LangGraph raises `InvalidUpdateError`. The reduction happens in `reduce_specialists`.

**Fallback (`FallbackGraph`):** If LangGraph is not installed, a `FallbackGraph` class runs all nodes sequentially in the correct order. Same input/output contract, no parallelism.

---

## 10. CLI Entry Point — `cli.py`

### `main()`

1. Loads `.env` file (via `python-dotenv`) — sets `OPENAI_API_KEY` and model overrides.
2. Parses `--input` and `--output` arguments.
3. Calls `build_graph().invoke({"input_path": ..., "status": "init"})`.
4. Writes `output.json` at the specified path (UTF-8 encoded, pretty-printed with 2-space indent).

**Usage:**
```bash
py -3.10 -m document_agent --input data/paper.pdf --output outputs/result.json
```

---

## 11. Techniques Reference

### PaddleOCR PP-DocLayout_plus-L

A YOLO-based object detection model fine-tuned on large-scale document layout datasets. Returns bounding boxes classified into semantic document regions. Significantly outperforms rule-based layout analysis on complex academic papers, newspapers, and forms.

**Key advantage over simpler approaches:** Detects formulas, tables, figures, and captions as distinct semantic objects — not just "blocks of text". This enables routing each region to the right specialist.

### LayoutReader / ReadingBank

ReadingBank is a dataset of ~500,000 Word documents with explicit reading order annotations (preserved from the `.docx` paragraph order). LayoutReader is trained on ReadingBank using a seq2seq transformer architecture: bounding boxes in → reading order permutation out.

**Why this beats heuristics:** Real documents break every heuristic. A paper might have an abstract in a single wide column, then two narrow columns, then a figure spanning both columns, then a footnote. LayoutReader learned all these patterns from data.

### X-Projection Gap Analysis (Column Detection)

Builds a 1D histogram of horizontal coverage and finds gaps in the central 20%–80% of the page. Robust, O(page_width) complexity, and requires no training. Used as the LayoutReader fallback.

### IoU (Intersection over Union)

Standard object detection metric repurposed for containment detection. A block with 80% of its area inside another block is considered a child.

### Inline Math Stitching

Novel heuristic that merges adjacent text+formula blocks on the same text line into a MIXED block, then processes the merged region with a VLM prompt designed for mixed-content. Solves the semantic discontinuity caused by separate bounding boxes for inline mathematical expressions.

### ThreadPoolExecutor for Block Parallelism

Each specialist node processes multiple blocks of the same type in parallel using Python's `concurrent.futures.ThreadPoolExecutor`. Since VLM calls are I/O-bound (network requests to OpenAI), this provides near-linear speedup with block count.

### Regex Cross-Reference Extraction

Pattern-based extraction of academic cross-references (Figure X, Equation N, [Citation]) enables building a reference graph over the document's blocks. This is critical for RAG: a text chunk that says "see Figure 3" should carry a pointer to the actual figure block.

---

## 12. Output Schema

The output JSON (`output.json`) has this top-level structure:

```json
{
  "run_id": "uuid",
  "input_path": "data/paper.pdf",
  "status": "done",
  "warnings": ["..."],
  "pages": [
    {"page_index": 0, "width": 1654, "height": 2339}
  ],
  "blocks": [ ... ],
  "summary": {
    "num_pages": 20,
    "num_blocks": 444,
    "types": {"text": 255, "formula": 86, "table": 16, ...}
  }
}
```

Each block in `"blocks"`:

```json
{
  "id": "b42",
  "page_index": 3,
  "bbox": {"x1": 72, "y1": 148, "x2": 756, "y2": 192},
  "type": "text",
  "detector_label": "paragraph_title",
  "confidence": 0.97,
  "reading_order": 5,
  "parent_id": null,
  "child_ids": [],
  "line_group_id": null,
  "relations": {
    "references": [
      {"type": "figure", "label": "3", "context": "Figure 3"}
    ]
  },
  "payload": {
    "text": "3. Experimental Results",
    "ocr_engine": "paddleocr"
  }
}
```

**`payload` contents by block type:**

| Type | Payload keys |
|---|---|
| `text` | `text`, `ocr_engine` |
| `mixed` | `full_text` (prose + inline LaTeX), `formula_latex`, `plain_text`, `vlm_engine`, `vlm_model` |
| `image` | `description` (JSON: summary, key_elements, visible_text, spatial_layout, confidence), `vlm_engine`, `vlm_model` |
| `chart` | `chart_semantics` → `{status, data}` where data is JSON string, `chart_engine`, `chart_model` |
| `formula` | `formula_latex` (JSON: latex, symbols, meaning, confidence), `formula_engine`, `formula_model`, `preprocess` |
| `table` | `table_data` → `{status, data}` where data is JSON string (headers, rows, notes), `table_engine`, `table_model` |
| `caption`, `header`, `footer`, `other` | `text`, `ocr_engine` |

**`relations` contents:**

| Key | Set by | Meaning |
|---|---|---|
| `references` | `cross_reference` | List of `{type, label, context}` for in-text references |
| `describes` | `association`, `hierarchy` | For CAPTION: block_id of the figure/table it describes |
| `captions` | `association`, `hierarchy` | For figures/tables: list of caption block_ids |
| `embedded_formulas` | `association` | For TEXT/MIXED: list of FORMULA block_ids embedded inside |
| `embedded_in` | `association` | For FORMULA: block_id of parent TEXT/MIXED |
| `is_page_image` | `association` | `true` if block covers ≥ 85% of page |
| `page_coverage` | `association` | Float ratio of page area covered |
