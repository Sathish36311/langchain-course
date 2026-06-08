from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """Mahendra Singh Dhoni ([məˈɦeːnd̪ɾə ˈsɪŋɡʱ ˈd̪ʱoːniː] ⓘ; born 7 July 1981) is an Indian professional cricketer who plays as a right-handed batter and a wicket-keeper. Widely regarded as one of the most prolific wicket-keeper batsmen and captains, he represented the Indian cricket team and was the captain of the team in limited overs formats from 2007 to 2017 and in Test cricket from 2008 to 2014. Dhoni has captained the most international matches and is the most successful Indian captain. He has led India to victory in the 2007 ICC World Twenty20, the 2011 Cricket World Cup, and the 2013 ICC Champions Trophy, being the only captain to win three different limited overs ICC tournaments. He also led the teams that won the Asia Cup in 2010 and 2016, and he was a member of the title winning squad in 2018."""

    summary_template = """
    given the information {information} about a persion I want you to create:
    1. A short summart
    2. two interesting facts about the person
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm = ChatOllama(model="gemma3:270m", temperature=0)

    # LangChain Expression Language (LCEL) expression
    runnable_object = summary_prompt_template | llm
    response = runnable_object.invoke(input={"information": information})

    print(response.content)


if __name__ == "__main__":
    main()