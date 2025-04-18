#!/usr/bin/env python3
"""
run_interviewee_agent.py
------------------------

Пример CLI‑утилиты: выбираем профиль кандидата, собираем system‑prompt,
запускаем агента и ведём диалог в терминале.
"""

import random
from interviewee_agent import create_interviewee_agent
from app.agents.profiles import candidate_profiles


def choose_profile() -> str:
    """Интерактивный выбор профиля из словаря."""
    print("Выберите тип кандидата:")
    keys = list(candidate_profiles.keys())
    for i, k in enumerate(keys, 1):
        print(f"{i}. {k}")
    sel = int(input("Введите номер → ")) - 1
    return keys[sel]


def main() -> None:
    profile_key = choose_profile()
    experience = input("Уровень опыта (Junior/Middle/Senior) → ")
    specialization = input("Специализация (напр. 'Data Analyst') → ")
    skill = input("Навык, который проверяем → ")

    agent = create_interviewee_agent(
        profile_key=profile_key,
        experience=experience,
        specialization=specialization,
        skill=skill,
        key_skills="Python, SQL, коммуникация",  # пример
    )

    print("\nДиалог начат! Пишите вопросы; 'exit' чтобы выйти.\n")
    while True:
        user = input("HR > ")
        if user.lower().strip() == "exit":
            break
        response = agent.run([{"role": "user", "content": user}])
        print(f"\nКандидат ({profile_key}) > {response}\n")


if __name__ == "__main__":
    main()
