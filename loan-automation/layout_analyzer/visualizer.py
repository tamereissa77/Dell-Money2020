"""
Bounding Box Visualization Module
Draws color-coded, labeled bounding boxes on document images.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple
from .config import (
    ID_FRONT_CLASSES, ID_BACK_CLASSES, HR_LETTER_CLASSES,
    BOX_THICKNESS, LABEL_FONT_SCALE, LABEL_THICKNESS
)


def get_class_config(doc_type: str) -> Dict:
    """Return the class configuration for a document type."""
    if doc_type == 'id_front':
        return ID_FRONT_CLASSES
    elif doc_type == 'id_back':
        return ID_BACK_CLASSES
    elif doc_type == 'hr_letter':
        return HR_LETTER_CLASSES
    return {}


def draw_annotated_image(image_path: str, analysis_result: Dict,
                         output_path: str, draw_all_tokens: bool = False) -> str:
    """
    Draw color-coded bounding boxes on the document image.
    Each field class gets a unique color. Labels are drawn with background.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")

    overlay = img.copy()
    doc_type = analysis_result['doc_type']
    class_config = get_class_config(doc_type)
    fields = analysis_result.get('fields', {})

    # Optionally draw all OCR tokens in light gray
    if draw_all_tokens:
        for tok in analysis_result.get('all_tokens', []):
            b = tok['bbox']
            cv2.rectangle(overlay, (b[0], b[1]), (b[2], b[3]),
                         (200, 200, 200), 1)

    # Draw each detected field with its class color
    for class_name, field_data in fields.items():
        cfg = class_config.get(class_name, {})
        color = cfg.get('color', (0, 255, 0))
        label = cfg.get('label_en', class_name)
        bbox = field_data['bbox']
        value = field_data.get('value', '')

        # Draw filled rectangle with transparency
        x1, y1, x2, y2 = bbox
        sub_overlay = overlay.copy()
        cv2.rectangle(sub_overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(sub_overlay, 0.15, overlay, 0.85, 0, overlay)

        # Draw border
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        # Draw label with background
        label_text = f"{label}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(label_text, font,
                                              LABEL_FONT_SCALE, LABEL_THICKNESS)
        label_y = max(y1 - 8, th + 4)
        # Background rectangle for label
        cv2.rectangle(overlay, (x1, label_y - th - 4),
                     (x1 + tw + 8, label_y + 4), color, -1)
        cv2.putText(overlay, label_text, (x1 + 4, label_y),
                   font, LABEL_FONT_SCALE, (255, 255, 255), LABEL_THICKNESS)

    # Draw legend in top-left corner
    legend_y = 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    for class_name in fields:
        cfg = class_config.get(class_name, {})
        color = cfg.get('color', (0, 255, 0))
        label = cfg.get('label_en', class_name)
        cv2.rectangle(overlay, (10, legend_y - 12), (26, legend_y + 4), color, -1)
        cv2.putText(overlay, f" {label}", (30, legend_y + 2),
                   font, 0.45, (0, 0, 0), 1)
        legend_y += 22

    cv2.imwrite(output_path, overlay)
    return output_path


def format_results_json(analysis_result: Dict) -> Dict:
    """Format analysis results as clean JSON output (no token details)."""
    output = {
        'doc_type': analysis_result['doc_type'],
        'image_size': analysis_result['image_size'],
        'detections': {}
    }
    for class_name, field_data in analysis_result.get('fields', {}).items():
        if field_data is None:
            continue
        output['detections'][class_name] = {
            'bbox': field_data.get('bbox', [0, 0, 0, 0]),
            'value': field_data.get('value', ''),
            'confidence': round(
                np.mean([t['conf'] for t in field_data.get('tokens', [])]) 
                if field_data.get('tokens') else 0.0, 3
            )
        }
    return output
