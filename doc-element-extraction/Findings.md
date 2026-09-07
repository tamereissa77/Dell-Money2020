# LightOnOCR-2-1B Variants

## LightOnOCR-2-1B-base

**Desc: Supervised baseline (81.8%) - Fine-tuning base**

**Tested on 4 different documents each with 3 different pages**

**Issues:**

- Classifies header section as `Image` element
- Struggles with page number extraction (shows LATEX code or something similar)

![Page Number](/results/LightOnOcr/LightOnOcr-2-1B-base/Page_Number-content-image.png)

- Classifies signature section as `Section-Header` element and Extract in Eglish!

![Signature Content](/results/LightOnOcr/LightOnOcr-2-1B-base/Signature-content-image.png)

- Some inconsistencies in text extraction

![Inconsistent Text Extraction](/results/LightOnOcr/LightOnOcr-2-1B-base/Inconsistencies-1.png)

- Table numbers are extracted in English instead of Arabic
- Some extracted tables are in reversed order (from bottom to top)

![Reversed Order Table](/results/LightOnOcr/LightOnOcr-2-1B-base/Reversed_Order-table.png)
![Reversed Order Content](/results/LightOnOcr/LightOnOcr-2-1B-base/Reversed_Order-content.png)

- Some extracted tables are missing columns

![Missing Columns T able](/results/LightOnOcr/LightOnOcr-2-1B-base/Missed_Column-table.png)
![Missing Columns Content](/results/LightOnOcr/LightOnOcr-2-1B-base/Missed_Column-content.png)

**My Conclusion:**

1. Elements with limited number of charachters (page numbers, single character header, etc) mostly interpreted as `LATEX` code
2. Page quality (resolution, noise, etc) has a big impact on the model performance, some `complex` tables with moderate to high quality are extracted correctly while `simple` tables with low quality are not extracted correctly
3. Colored tables (e.g. gray background) conflicts the model sometimes

**Proposed Solution:**

1. Introduce a CV-based layer to extract page numbers, footers, etc.
2. Introduce a `normalization/preprocessing` layer for low to medium quality docments.
3. introduce `On-the-Fly` `normalization/preprocessing` for specific (challenging) tables/elements.

## LightOnOCR-2-1B-ocr-soup

**Desc: Task-arithmetic merged - Alternative OCR**

**Compared with `LightOnOCR-2-1B-ocr (Best OCR)`on 4 different documents**

![Comaprison 1: header section](/results/LightOnOcr/LightOnOcr-2-1B-ocr-soup/soup_vs_best_ocr-section-header.png)

![Comaprison 2: table header](/results/LightOnOcr/LightOnOcr-2-1B-ocr-soup/soup_vs_best_ocr-fetching-table-headers.png)

![Comaprison 3: page number](/results/LightOnOcr/LightOnOcr-2-1B-ocr-soup/soup_vs_best_ocr-both-fail-in-page-number.png)

![Comaprison 4: complex table](/results/LightOnOcr/LightOnOcr-2-1B-ocr-soup/soup_vs_best_ocr-complex-tables.png)

**My Conclusion:**

1. Similar results compared with `LightOnOCR-2-1B-ocr (Best OCR)`
2. Minor ordering issues (number order (1 236 - 236 1), columns order (Col A, Col B - Col B, Col A))
3. Both `LightOnOCR-2-1B-ocr (Best OCR)` and `LightOnOCR-2-1B-ocr-soup` show inconsistent with complex tables (multiple row in single cell),**although the extracted html is perfect, the markdown result is not**

![html vs markdown for complex tables](/results/LightOnOcr/LightOnOcr-2-1B-ocr-soup/html_vs_markdown_for_complex_tables.png)

4. Elements with limited number of charachters (page numbers, single character header, etc) mostly interpreted as `LATEX` code

**Proposed Solution:**

3. Introduce more complex `html to markdown` layer (maybe LLM-based)

## LightOnOCR-2-1B-bbox-soup

**Desc: OCR-bbox trade-off**

**My Conclusion:**

1. Shows very good results in element `detection`

![bbox soup detection](/results/LightOnOcr/LightOnOCR-2-1B-bbox-soup/bbox_detection.jpg)

2. Struggles with element content `extraction`

![bbox soup extraction-header](/results/LightOnOcr/LightOnOCR-2-1B-bbox-soup/bbox_extraction-header.png)

![bbox soup extraction-table](/results/LightOnOcr/LightOnOCR-2-1B-bbox-soup/bbox_extraction-text.png)
