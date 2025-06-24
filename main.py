from corelibs.helpers.common.file_utils import FileUtils

from crewai import Agent, LLM
from crewai.lite_agent import LiteAgentOutput

from datetime import datetime

lms_llm_gemma: LLM = LLM(
    model="lm_studio/google/gemma-3-4b",
    base_url="http://127.0.0.1:1234/v1",
    api_key="asdf",
    temperature=1.0
)

# template LLM
# lms_llm_xyz = LLM(
#    model="", # your next model
#    base_url="http://127.0.0.1:1234/v1",
#    api_key="asdf",
#    temperature=1.0
# )

use_llm: LLM = lms_llm_gemma

sole_agent: Agent = Agent(
    role="Awesome Professor Astra",
    goal=("""
        Explain complex concepts in simple, imaginative, and
        unforgettable ways using analogies and stories.
        Structure the result into paragraphs with headlines and
        use markup to emphasis essential words and phases and 
        use markdown for all text markup. Apply suitable emoji.

    """),
    backstory=("""
        You are Professor Astra, not bound to dusty textbooks!
        You journey though galaxies of knowledge, transforming tricky
        ideas into sparkling constellations of understanding. You
        believe any concept, no matter how complex, can be grasped if
        viewed through the lens of wonder.
    """),
    llm=use_llm,
    verbose=True,
    allow_delegation=False
)


def run() -> None:
    prompt = ("""
        Professor Astra, please explain 'photosynthesis' to me.
        Imagine I'm a slightly bored space cadet on a long voyage.
        Make it sound like a secret, vital mission that tiny green
        astronauts (chloroplasts) undertake inside leaves, using 
        sunlight as their hyperdrive fuel to create space snacks (sugar)
        and fresh air (oxygen) for the ship (Earth)! Keep it brief
        but dazzling!
    """)

    result: LiteAgentOutput = sole_agent.kickoff(prompt)

    print("\n--- Professor Astra's Explanation ---\n")
    print(result)
    print("\n========================================================================\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename: str = "./documents/photosynthesis_story_" + use_llm.model.split("/")[-1] + f"_{timestamp}.md"


if __name__ == '__main__':
    run()
