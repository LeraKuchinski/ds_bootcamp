from agents import Agent
from app.agents.tools.lie_answer import lie_answer

from app.agents.profiles import candidate_profiles



def create_interviewee_agent(system_prompt: str, psyho_profile: str, persona: str, skill: str) -> Agent:
    profile = candidate_profiles[psyho_profile]
    profile_description = f"{profile['name']} — {profile['type']}. {profile['description']}"
    
    return Agent(
        name="Кандидат на работу",
        handoff_description="""Ты кандидат, который точно отвечает в соответствии с инструкцией
        Кандидат, который следует психотипу и контексту запроса""", # Инструкция агенту - передача задач агентам если их несколько
        instructions=system_prompt.format(
            name=profile['name'],
            persona=persona,
            profile_description=profile_description,
            skill=skill,
        ),
        tools=[lie_answer],
    )
    
