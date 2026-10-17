"""Text prompts for zero-shot anomaly detection with vision–language models.

A prompt set has *templates* (``"A fundus photograph of {}"``) and two lists of *states*: what normal images
show and what anomalous images show (``"healthy"`` / ``"diabetic retinopathy"``). Every state is written
into every template and the embeddings are averaged per state, as in CLIP's prompt ensembles and WinCLIP's
compositional prompts.

The defaults below are a starting point: name the findings that matter in your data (``anomalous=[...]``)
and benchmark the prompt sets like any other configuration.
"""

from __future__ import annotations

from typing import Dict, List, Optional

TEMPLATES: Dict[str, List[str]] = {
    "ophthalmology": ["A fundus photograph of {}", "a colour fundus photograph showing {}", "a retinal photograph of {}"],
    "oct": ["An OCT scan of {}", "an optical coherence tomography B-scan showing {}", "a retinal OCT image of {}"],
    "chest-xray": ["A chest X-ray showing {}", "a chest radiograph of {}", "a frontal chest X-ray with {}"],
    "ct": ["A CT scan showing {}", "a computed tomography slice of {}"],
    "mri": ["An MRI scan showing {}", "a magnetic resonance image of {}"],
    "histopathology": ["An H&E histopathology image of {}", "a histology tile showing {}"],
    "dermatology": ["A dermoscopy image of {}", "a photograph of a skin lesion that is {}"],
    "general": ["a photo of {}", "a cropped photo of {}", "a close-up photo of {}", "an image of {}"],
}

NORMAL: Dict[str, List[str]] = {
    "ophthalmology": ["a healthy retina", "a normal fundus", "no findings", "no lesion signs"],
    "oct": ["a normal retina", "healthy retinal layers", "a normal foveal contour", "no findings"],
    "chest-xray": ["normal lungs", "no findings", "a normal chest", "no acute cardiopulmonary abnormality"],
    "ct": ["normal anatomy", "no findings", "no abnormality"],
    "mri": ["normal anatomy", "no findings", "no abnormality"],
    "histopathology": ["normal tissue", "benign tissue", "no tumour"],
    "dermatology": ["a benign lesion", "normal skin", "a benign nevus"],
    "general": ["a flawless object", "a perfect object", "an unblemished object", "a normal object"],
}

ANOMALOUS: Dict[str, List[str]] = {
    "ophthalmology": ["diabetic retinopathy", "glaucoma", "age related macular degeneration", "drusen",
                      "haemorrhages", "exudates", "retinal vein occlusion", "a macular hole", "pathologic myopia",
                      "optic disc edema", "retinal disease", "lesions"],
    "oct": ["choroidal neovascularization", "diabetic macular edema", "drusen", "retinal fluid",
            "a macular hole", "an epiretinal membrane", "retinal detachment", "retinal disease"],
    "chest-xray": ["pneumonia", "lung opacity", "pleural effusion", "cardiomegaly", "a lung nodule",
                   "pneumothorax", "disease"],
    "ct": ["a tumour", "a lesion", "haemorrhage", "disease"],
    "mri": ["a tumour", "a lesion", "abnormal signal", "disease"],
    "histopathology": ["tumour", "metastasis", "malignant cells", "cancer"],
    "dermatology": ["melanoma", "a malignant lesion", "carcinoma"],
    "general": ["a damaged object", "a broken object", "an object with a defect", "an object with a flaw"],
}


def _key(modality: Optional[str]) -> str:
    if not modality:
        return "general"
    if modality in TEMPLATES:
        return modality
    from .zoo import modality as canonical

    return canonical(modality)


def default_prompts(modality: Optional[str] = None) -> Dict[str, List[str]]:
    """``{"templates": [...], "normal": [...], "anomalous": [...]}`` for a modality (zoo names and aliases,
    e.g. ``"fundus"``, ``"oct"``, ``"cxr"``) or for natural/industrial images (``None``)."""
    k = _key(modality)
    return {"templates": list(TEMPLATES[k]), "normal": list(NORMAL[k]), "anomalous": list(ANOMALOUS[k])}
