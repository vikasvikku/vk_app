import logging
from typing import List
from app.models.topic_models import TopicList
import openai
import ell
from openai import OpenAI

# Set the OpenAI API key
openai.api_key = "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"

class OpenAIService:
    @ell.complex(
        model="gpt-4o-mini",
        client=OpenAI(api_key=openai.api_key),
        temperature=0.75,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.5,
        response_format=TopicList
    )
    def extract_topics(self, text: str) -> TopicList:
        return [
            ell.system(f"""
                Analyze the provided article to extract topics suitable for creating knowledge capsules. 
                Each topic should be concise and specific enough to be effectively covered in a 15-minute session. For each topic, 
                evaluate its relevance, applicability, and current popularity ('hotness').
               
                Follow these steps:
                1. Extract primary subjects, topics, or trends discussed in the article. Group related ideas under broader topics, 
                   but ensure each identified topic is narrow enough and specific enough to be thoroughly addressed in 15 minutes.

                2. Avoid overly broad or general topics; instead, focus on specific aspects.

                3. Focus on topics that are impactful, interesting, and affect technology and human lives.

                4. Assess how closely each topic aligns with the area of interest. Use a scale of 1 to 10, 
                   where 1 indicates little relevance and 10 indicates strong alignment. Provide a brief explanation for your score.

                5. Evaluate how directly each topic can be applied. Consider real-world use cases, feasibility, and potential benefits. 
                   Use a scale of 1 to 10, and justify your rating.

                6. Research or infer how trending or popular the topic is based on recent developments, media coverage, or industry focus. 
                   Use a scale of 1 to 10 to rate its hotness and briefly explain your reasoning.

                7. Confirm that each topic is focused and not too wide or complex to be explained in a 15-minute knowledge capsule. 
                   Refine broad topics into smaller, actionable sub-topics if necessary.

                8. Based on the above evaluations, prioritize topics with high scores across relevance, applicability, and popularity. 
                   Provide a list of recommended topics with a one-line description of their potential use as a knowledge capsule.

                9. Study the list of topics generated, compare them and identify which ones cover the same subject matter or 
                   address the same problem. Retain only the unique topics and remove any duplicates or overly similar ones. 

                   Provide a final list of distinct topics that represent a wide range of ideas without overlap.

                10. Rewrite the topic heading to sound more exciting, 
                    dynamic and interesting to the audience, while focusing on its futuristic elements.
                       
                11. Ensure each topic has a similarity score below 0.7 compared to the existing database to avoid repetition.0.
                

                Return the output as a JSON object with the following structure:
                {{
                    "topics": [
                        {{
                            "topic": "Topic name (5 words)",
                            "attributes": {{
                                "field": "",
                                "sub_field": "",
                                "subject_matter": "",
                                "relevance": "20 words",
                                "potential_impact": "20 words",
                                "hotness": "(High/Medium/Low)"
                            }}
                        }}
                    ]
                }}
            """),
            ell.user(f"Text chunk: {text}")
        ]