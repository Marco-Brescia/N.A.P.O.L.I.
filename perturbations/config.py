import os

# Configurazione dei percorsi di IO
IMAGES_DIR = "test_phase0"
MASKS_DIR  = "masks_experiment"
OUT_DIR    = "images_perturbed"

# Configurazione con minimi allineati al primo step reale dell'effetto
perturbation_ranges = {
    #noise
    "GaussianNoise":       {"param": "std",         "min": 5.0,  "max": 85,   "step": 20.0,  "type": float},
    "ImpulseNoise":        {"param": "ratio",       "min": 0.03,  "max": 0.15, "step": 0.03,  "type": float},
    "ShotNoise":           {"param": "scale",       "min": 20.0,  "max": 100,  "step": 20.0,  "type": float},
    "ISONoise":            {"param": "intensity",   "min": 0.5,   "max": 4.5,  "step": 1.0,   "type": float},
    "PoissonNoise":        {"param": "scale",       "min": 10,  "max": 80,   "step": 17.5,  "type": float},
    "MultiplicativeNoise": {"param": "variance",    "min": 0.05,  "max": 0.35, "step": 0.10,  "type": float},
    #blur/smoothing
    "GaussianLowPass":     {"param": "sigma",       "min": 1.0,   "max": 5.0,  "step": 1.0,   "type": float},
    "DefocusBlur":         {"param": "radius",      "min": 1,     "max": 9,    "step": 2,     "type": int},
    "MotionBlur":          {"param": "ksize",       "min": 5,     "max": 25,   "step": 5,     "type": int},
    "ZoomBlur":            {"param": "zoom_max",    "min": 1.10,  "max": 1.50, "step": 0.10,  "type": float},
    "MedianFilter":        {"param": "ksize",       "min": 3,     "max": 15,   "step": 2,     "type": int},
    #illumination/color
    "Brightness":          {"param": "factor",      "min": 0.5,   "max": 2.5,  "step": 0.5,   "type": float},
    "Contrast":            {"param": "factor",      "min": 0.5,   "max": 2.5,  "step": 0.5,   "type": float},
    "Saturation":          {"param": "factor",      "min": 0.9,   "max": 0.0,  "step": -0.15,   "type": float},
    "GammaContrast":       {"param": "gamma",       "min": 0.4,   "max": 2.0,  "step": 0.4,   "type": float},
    #environmental
    "Snow":                {"param": "intensity",  "min": 0.1,  "max": 1.1,  "step":  0.2,  "type": float},
    "Frost":               {"param": "intensity",   "min": 0.1,   "max": 1.1,  "step": 0.2,   "type": float},
    "Fog":                 {"param": "intensity",   "min": 0.30,  "max": 1.5,  "step": 0.30,  "type": float},
    "Splatter":            {"param": "num_drops",   "min": 65,    "max": 200,  "step": 45.0,  "type": int},
    # digital/compress
    "JPEGCompression":     {"param": "quality",     "min": 90,    "max": 10,   "step": -20.0, "type": float},
    "Pixelate":            {"param": "block_size",  "min": 5,     "max": 20,   "step": 3,     "type": int},
    # geometric
    "Scale":               {"param": "factor",      "min": 0.5,  "max": 1.5,  "step": 0.25,  "type": float},
    "Shear":               {"param": "angle",       "min": 15.0,  "max": 45.0,   "step": 15.0,  "type": float},
    "Jigsaw":              {"param": "grid_size",   "min": 2,     "max": 8,    "step": 2,     "type": int},
    "Rotate":              {"param": "angle",       "min": 36,    "max": 324,  "step": 36,    "type": int},
    # occlusion/dropout
    "Cutout":              {"param": "ratio",       "min": 0.1,   "max": 0.9,    "step": 0.2,   "type": float},
    "CoarseDropout":       {"param": "hole_size",   "min": 8,     "max": 40,   "step": 8,     "type": int},
    # edge-based
    "CannyEdge":           {"param": "th",          "min": 75,    "max": 250,  "step": 75,    "type": int},
    "AdversarialPatch":    {"param": "patch_ratio", "min": 0.075, "max": 0.50, "step": 0.075, "type": float},
}