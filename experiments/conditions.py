"""Experimental conditions compared in the workshop paper (one place, used by every script)."""

# name -> environment variables of the campaign
CONDITIONS = {
    "none":            {"MEMORY_MODE": "none",    "MONITOR": "false", "INJECT_FALSE_CLAIMS": "false"},
    "coord":           {"MEMORY_MODE": "coord",   "MONITOR": "false", "INJECT_FALSE_CLAIMS": "false"},
    "naive":           {"MEMORY_MODE": "naive",   "MONITOR": "false", "INJECT_FALSE_CLAIMS": "false"},
    "evident":         {"MEMORY_MODE": "evident", "MONITOR": "false", "INJECT_FALSE_CLAIMS": "false"},
    "evident+monitor": {"MEMORY_MODE": "evident", "MONITOR": "true",  "INJECT_FALSE_CLAIMS": "false"},
    "naive+inject":    {"MEMORY_MODE": "naive",   "MONITOR": "false", "INJECT_FALSE_CLAIMS": "true"},
    "evident+inject":  {"MEMORY_MODE": "evident", "MONITOR": "false", "INJECT_FALSE_CLAIMS": "true"},
}

# What each comparison isolates (used in the report)
QUESTIONS = {
    "Q1 coordination": ("none", "coord"),
    "Q2 verified vs raw knowledge": ("naive", "evident"),
    "Q3 robustness to a false claim": ("naive+inject", "evident+inject"),
    "Q4 real-time control": ("evident", "evident+monitor"),
}

# Fixed color per condition (color follows the entity), reference categorical palette
COLORS = {
    "none": "#2a78d6", "coord": "#eb6834", "naive": "#1baf7a", "evident": "#eda100",
    "evident+monitor": "#e87ba4", "naive+inject": "#008300", "evident+inject": "#6250d6",
}
