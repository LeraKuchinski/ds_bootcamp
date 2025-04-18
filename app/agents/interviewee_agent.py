from agents import Agent
from app.agents.tools.lie_answer import lie_answer

from app.agents.profiles import candidate_profiles

profile = candidate_profiles['talkative']

def create_interviewee_agent(system_prompt: str) -> Agent:
    return Agent(
        name="Стеснительный кандидат",
        handoff_description="Ты кандидат, который отвечает на вопросы очень стеснительно",
        # handoff_description="Ты кандидат, который отвечает на вопросы на основе персоны и проверяемого навыка",
        instructions=system_prompt.format(
    persona=f"{profile['name']} — {profile['type']}. {profile['description']}",
    name=profile['name'],
    experience="5 лет в data science",
    skill="Python",
),
        tools=[lie_answer],
    )
    
