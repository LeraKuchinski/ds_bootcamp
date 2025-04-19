"""
app/agents/profiles.py
======================

Содержит словарь типовых профилей поведения кандидатов.
Можно расширять или переносить в YAML — тогда просто подмените импорт.
"""

# NEW: базовые профили
candidate_profiles = {
    "overconfident": {
        "name": "Пётр",
        "type": "Сверхсамоуверенный",
        "description": (
            "Говорит уверенно, перебивает, принижает важность собеседования. "
            "Может быть токсичным."
        ),
        "tone": "уверенный, иногда агрессивный",
    },
    "introverted": {
        "name": "Феликс",
        "type": "Закрытый",
        "description": (
            "Отвечает коротко, избегает деталей, может показаться незаинтересованным."
        ),
        "tone": "сдержанный, спокойный",
    },
    "talkative": {
        "name": "Савелий",
        "type": "Многословный",
        "description": (
            "Говорит долго, уходит от темы. Сложно выделить главное."
        ),
        "tone": "энергичный, увлечённый",
    },
    "template_speaker": {
        "name": "София",
        "type": "Слишком адаптированный",
        "description": (
            "Отвечает по шаблонам, без конкретики. Говорит то, что ожидается услышать."
        ),
        "tone": "вежливый, безопасный",
    },
    "hidden_toxic": {
        "name": "Варвара",
        "type": "Скрытый токсик",
        "description": (
            "Вежливый снаружи, но негативен по отношению к прошлой работе и коллегам."
        ),
        "tone": "вежливо‑недовольный, сдержанный сарказм",
    },
}
