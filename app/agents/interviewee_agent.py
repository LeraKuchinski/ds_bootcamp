from agents import Agent
from app.agents.tools.lie_answer import lie_answer

from app.agents.profiles import candidate_profiles



def create_interviewee_agent(system_prompt: str, psyho_profile: str, persona, skill) -> Agent:
    profile = candidate_profiles[psyho_profile]
    return Agent(
        name="Стеснительный кандидат",
        handoff_description="Ты кандидат, который отвечает на вопросы очень стеснительно",
        instructions=system_prompt.format(
            persona=persona,  # <-- как в YAML-шаблоне
            name=profile['name'],
            skill=skill,
        ),
        tools=[lie_answer],
    )
    
