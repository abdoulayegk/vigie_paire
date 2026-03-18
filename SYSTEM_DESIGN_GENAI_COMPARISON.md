# System Design: GenAI Quarterly Report Comparison

## 1. Objective

This system compares Canadian bank quarterly reports using a GenAI-first pipeline centered on GPT-4o.

The objective is to:

- extract table content from quarterly PDF reports
- extract first-column indicators and table footnotes
- store extraction outputs by bank and quarter
- match the correct table pairs between two reports
- compare indicators and footnotes between quarters
- persist comparison results for fast analyst review in Dash
- allow analyst validation, rejection, correction, and comments
- produce a final reviewed report per bank

The system must also support extraction reuse:

- if `T2` has already been extracted, then for `T3 vs T2`
- only `T3` should be extracted
- `T2` extraction should be reused from storage


## 2. Core Design Principle

The system is strongly based on GPT-4o, but it should not delegate every task to GenAI.

### GPT-4o should be used for:

- Vision extraction of tables
- extraction of first-column indicators
- extraction of footnotes
- semantic validation of ambiguous table matches
- semantic validation of probable indicator renames
- semantic interpretation of ambiguous footnote changes
- generation of executive summaries

### Deterministic logic should be used for:

- storage and persistence
- cache reuse
- report identity and versioning
- diff computation
- review status tracking
- dashboard loading
- final report assembly

This separation is important for:

- speed
- auditability
- reproducibility
- lower cost
- operational reliability


## 3. Target User Workflow

### Night batch workflow

During the night, the system should:

1. detect newly available quarterly reports
2. verify whether each report has already been extracted
3. run extraction only for missing or outdated reports
4. run quarterly comparison jobs
5. persist comparison outputs
6. generate review-ready files for Dash
7. prepare final pre-reviewed artifacts

### Morning analyst workflow

In the morning, the analyst should:

1. open Dash
2. load already prepared results instantly
3. inspect changes without waiting for extraction or comparison
4. approve, reject, or edit detected changes
5. leave comments
6. save reviewed decisions
7. generate or export a final reviewed report

The Dash interface should be read-fast and review-oriented, not compute-heavy.


## 4. High-Level Architecture

The system should be organized into five modules.

### 4.1 Ingestion Module

Responsibilities:

- ingest PDF reports by bank / quarter / year
- compute file hash
- identify whether extraction already exists
- register source metadata

Inputs:

- bank code
- year
- quarter
- PDF file

Outputs:

- source PDF in persistent storage
- ingestion metadata

### 4.2 Extraction Module

Responsibilities:

- detect table regions
- crop table images
- call GPT-4o Vision
- extract:
  - table title
  - headers
  - first-column indicators
  - full rows when useful
  - footnotes
  - confidence and quality warnings
- persist extraction outputs

This is the main GenAI extraction layer.

### 4.3 Comparison Module

Responsibilities:

- load two previously extracted quarters
- match tables between the two reports
- compare indicators
- compare footnotes
- call GPT-4o only on ambiguous or semantic cases
- persist comparison output

### 4.4 Review Module

Responsibilities:

- load comparison results quickly in Dash
- expose review actions to analysts
- persist approvals, rejections, corrections, and comments

### 4.5 Final Report Module

Responsibilities:

- combine raw extraction + comparison + review decisions
- generate a reviewed final report per bank
- support JSON and optional Excel / CSV / PDF export


## 5. Storage Design

The storage layout should be organized by bank, year, quarter, and comparison pair.

Example:

```text
data/
  rbc/
    2025/
      T1/
        source/
          report.pdf
        extraction/
          extraction_meta.json
          tables.json
          indicators.json
          footnotes.json
      T2/
        source/
          report.pdf
        extraction/
          extraction_meta.json
          tables.json
          indicators.json
          footnotes.json
      comparisons/
        T2_vs_T1/
          comparison.json
          review_state.json
          final_report.json
```

This design gives:

- persistent historical storage
- one extraction per report
- one comparison artifact per quarter pair
- direct reuse of past extractions


## 6. Extraction Outputs

Each report should produce extraction files that are stable and reusable.

### 6.1 `tables.json`

Purpose:

- store the full extracted table content

Recommended content:

- table id
- page number
- title
- section
- headers
- rows
- first-column indicators raw
- first-column indicators normalized
- footnotes
- bbox
- extraction confidence
- extraction warnings
- extraction method

### 6.2 `indicators.json`

Purpose:

- provide a simplified comparison-ready view of first-column indicators

Recommended content:

- table id
- title
- section
- page
- indicator list
- normalized indicator list

### 6.3 `footnotes.json`

Purpose:

- provide a simplified comparison-ready view of footnotes

Recommended content:

- table id
- title
- section
- page
- ordered footnote list
- normalized footnote markers

### 6.4 `extraction_meta.json`

Purpose:

- support cache validation, audit, and reproducibility

Recommended content:

- bank code
- year
- quarter
- source PDF path
- PDF hash
- extraction timestamp
- model name
- prompt version
- pipeline version
- schema version
- extraction status
- table count
- warning count


## 7. Extraction Reuse Strategy

This is a required capability.

Before extracting a report, the system should verify:

- whether extraction files already exist
- whether the stored PDF hash matches the current PDF hash
- whether the pipeline version is still compatible
- whether the prompt version or schema version requires regeneration
- whether the existing extraction is complete and valid

### Reuse rule

Reuse extraction if:

- same bank
- same quarter
- same year
- same PDF hash
- compatible extraction version

Otherwise:

- re-run extraction

### Example

For `T2 vs T1`:

- extract `T1`
- extract `T2`

For `T3 vs T2`:

- extract `T3`
- reuse stored extraction of `T2`

This reduces:

- runtime
- cost
- duplicate API calls


## 8. Role of GPT-4o in Extraction

GPT-4o should be used as the main multimodal extraction engine.

For each table crop, GPT-4o Vision should return a structured result containing:

- visible table title
- visible headers
- first-column indicators in visual order
- rows when needed
- footnotes in visual order
- confidence
- truncation or quality flags

### Why GPT-4o is valuable here

- tables and footnotes are often visually complex
- OCR alone is often insufficient
- first-column structure may be hierarchical
- footnotes may be out of sequence or visually ambiguous

GPT-4o helps preserve semantic structure and visual ordering.


## 9. Table Matching Design

Table matching should be hybrid.

### 9.1 Deterministic first-pass matching

Use deterministic features such as:

- section similarity
- table number
- title similarity
- first-column indicator overlap
- page structure signals

This step should generate:

- high-confidence matches
- unmatched candidates
- ambiguous candidate sets

### 9.2 GPT-4o validation for ambiguous matches

Only ambiguous cases should be escalated to GPT-4o.

GPT-4o should answer:

- are these two tables the same business table?
- yes / no / uncertain
- confidence
- brief reason

This reduces cost while preserving GenAI value where it matters most.


## 10. Indicator Comparison Design

Once a table pair is matched, compare first-column indicators.

### Deterministic diff

Use deterministic logic for:

- exact matches
- normalized matches
- indicators added in current quarter
- indicators removed from previous quarter

### GPT-4o semantic rename validation

When two labels are similar but not equal, GPT-4o should determine:

- same indicator renamed
- meaning changed
- unrelated labels
- uncertain

Possible statuses:

- unchanged
- added
- removed
- renamed
- uncertain


## 11. Footnote Comparison Design

Footnotes should be compared per matched table pair.

### Deterministic comparison

Use deterministic logic for:

- added footnotes
- removed footnotes
- text-modified footnotes

### GPT-4o semantic footnote interpretation

Use GPT-4o for difficult cases such as:

- same note content with changed numbering
- wording change vs true methodological change
- semantic equivalence vs substantive update

This is important because footnotes may carry regulatory or methodological meaning.


## 12. Comparison Output

Each quarter pair should produce one comparison artifact.

Example:

- `T2_vs_T1/comparison.json`

The comparison output should contain:

- comparison identity
- summary counters
- matched table comparisons
- added tables
- removed tables
- validation metadata
- quality metadata

### Main sections

#### `summary`

Should contain:

- extracted table counts
- comparable table counts
- matched table counts
- added / removed table counts
- ambiguous counts
- added / removed / renamed indicator counts
- footnote change counts

#### `table_comparisons`

One entry per matched pair:

- table id in previous quarter
- table id in current quarter
- titles
- pages
- section
- added indicators
- removed indicators
- renamed indicators
- footnote changes
- table-level status
- confidence or validation metadata

#### `tables_added`

Tables only present in current quarter.

#### `tables_removed`

Tables only present in previous quarter.

#### `meta`

Should contain:

- generation timestamp
- model version
- prompt version
- pipeline version
- validation stats
- optional executive summary


## 13. Review State Design

The analyst review state must be stored separately from raw extraction and raw comparison outputs.

This is important for auditability.

### `review_state.json`

Each reviewable item should support:

- item id
- item type
- review status
- reviewer
- timestamp
- comment
- edited value if applicable

### Recommended review statuses

- `pending`
- `approved`
- `rejected`
- `edited`

### Review actions in Dash

The analyst should be able to:

- approve a detected change
- reject a detected change
- correct a label or classification
- add a business comment

Raw extracted data should remain unchanged.
The reviewed output should be derived from overrides and analyst decisions.


## 14. Final Report Generation

The final report must be built from three layers:

1. raw extraction
2. raw comparison
3. review overrides

The final report should include:

- bank identity
- compared quarters
- reviewed table changes
- reviewed indicator changes
- reviewed footnote changes
- analyst decisions
- analyst comments
- optional executive summary

Recommended outputs:

- `final_report.json`
- optional Excel / CSV
- optional PDF or executive note


## 15. Dash Design Principle

Dash should not perform extraction or heavy comparison work at load time.

Dash should only:

- read prepared artifacts
- display comparison results
- display supporting extraction context
- persist analyst decisions

This ensures:

- fast load time
- stable user experience
- no dependency on live API calls during review


## 16. Pipeline Schedule

The recommended operational mode is nightly batch execution.

### Nightly run sequence

1. discover available PDFs
2. validate extraction cache
3. run extraction for missing reports
4. run pairwise comparison jobs
5. generate comparison artifacts
6. generate review-state seeds
7. generate preliminary final artifacts
8. record logs and metrics

### Morning use

1. analyst opens Dash
2. dashboard loads comparison artifacts
3. analyst reviews results
4. decisions are persisted immediately
5. final outputs are updated


## 17. Versioning and Auditability

Every persisted artifact should be version-aware.

Recommended fields:

- `model_version`
- `prompt_version`
- `pipeline_version`
- `schema_version`
- `created_at`
- `source_pdf_hash`

This is required because:

- prompts may change
- extraction schemas may evolve
- model behavior may change
- comparisons must remain traceable


## 18. Recommended Use of GPT-4o

To be able to say that the system is genuinely GenAI-based, GPT-4o should be central in four areas:

### 18.1 Vision extraction

Primary content extraction from tables and footnotes.

### 18.2 Ambiguous table matching

Semantic validation of uncertain table pairs.

### 18.3 Rename and semantic change validation

Interpretation of label changes and footnote wording changes.

### 18.4 Executive summarization

Human-readable summary of major regulatory or structural changes.

This is the strongest design: GPT-4o is central, but the platform remains reliable and industrial.


## 19. Why This Design Is Strong

This design satisfies the supervisor's expectations because:

- GPT-4o is used in the core intelligence layer
- extraction outputs are reusable across quarters
- the system supports overnight runs
- Dash loads quickly in the morning
- analysts can review and correct results
- final outputs are auditable and persistent

It also avoids common failure modes:

- running GPT-4o every time Dash opens
- re-extracting previously processed quarters unnecessarily
- mixing raw extraction and analyst corrections
- relying on GenAI for every single deterministic operation


## 20. Recommended Delivery Message

The system can be presented as follows:

> The platform is a batch-oriented GenAI comparison system for quarterly bank reports. GPT-4o is used as the primary engine for multimodal extraction and semantic validation, while deterministic services handle persistence, cache reuse, diff computation, analyst review, and final reporting. This architecture allows overnight execution, fast dashboard access in the morning, auditability of decisions, and quarter-to-quarter extraction reuse such as reusing T2 when comparing T3 vs T2.


## 21. Minimum Viable Delivery Scope

If implementation time is limited, prioritize the following:

1. persistent extraction by report
2. extraction reuse by quarter and PDF hash
3. persistent comparison by quarter pair
4. Dash review persistence
5. final reviewed report generation

This gives a coherent and defensible first delivery.
