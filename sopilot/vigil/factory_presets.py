"""Factory safety scenario presets for demo scripts.

Usage::

    from sopilot.vigil.factory_presets import get_preset, list_presets

    preset = get_preset("warehouse_safety")
    rules  = preset["rules"]
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

FACTORY_PRESETS: dict[str, dict] = {
    "warehouse_safety": {
        "name": "倉庫作業安全",
        "description": "Warehouse forklift proximity and PPE compliance monitoring",
        "rules": [
            "フォークリフト接近禁止エリア侵入",
            "ヘルメット未着用",
            "安全ベスト未着用",
            "立入禁止エリア侵入",
        ],
        "yolo_classes": ["person", "forklift", "helmet", "vest"],
        "mock_labels": ["person", "helmet"],
    },
    "construction_safety": {
        "name": "建設現場安全",
        "description": "Construction site PPE compliance and danger zone access control",
        "rules": [
            "ヘルメット未着用",
            "安全ハーネス未着用",
            "危険区域への侵入",
            "重機作業半径内立入禁止エリア侵入",
        ],
        "yolo_classes": ["person", "helmet", "hard hat", "safety harness"],
        "mock_labels": ["person"],
    },
    "retail_security": {
        "name": "小売店舗セキュリティ",
        "description": "Retail loss prevention and restricted area access control",
        "rules": [
            "バックヤード立入禁止エリア侵入",
            "不審行動検知",
            "未承認エリア侵入",
        ],
        "yolo_classes": ["person", "bag", "backpack"],
        "mock_labels": ["person"],
    },
    "food_processing": {
        "name": "食品工場衛生管理",
        "description": "Food processing hygiene compliance and clean-zone access monitoring",
        "rules": [
            "マスク未着用",
            "手袋未着用",
            "ヘアネット未着用",
            "衛生管理区域侵入",
        ],
        "yolo_classes": ["person", "mask", "gloves"],
        "mock_labels": ["person", "mask"],
    },
    "server_room": {
        "name": "サーバールームアクセス管理",
        "description": "Data center access control and equipment security monitoring",
        "rules": [
            "無許可アクセス侵入",
            "機器持ち出し検知",
            "制限エリア侵入",
        ],
        "yolo_classes": ["person", "laptop", "bag"],
        "mock_labels": ["person"],
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_preset(name: str) -> dict:
    """Return preset config dict.

    Raises
    ------
    KeyError
        If *name* is not a recognised preset.
    """
    if name not in FACTORY_PRESETS:
        available = ", ".join(sorted(FACTORY_PRESETS))
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    return FACTORY_PRESETS[name]


def list_presets() -> list[str]:
    """Return sorted list of preset names."""
    return sorted(FACTORY_PRESETS)
