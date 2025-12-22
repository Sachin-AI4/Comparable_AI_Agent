import json
from langchain_openai import ChatOpenAI
from langchain_core.messages
import config

class LLMEnricher:
    """Handles LLM-based enrichment of domain names"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=0.05,
            api_key=config.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"  # OpenRouter API endpoint
        )
    def enrich_domain(self, domain_name:str,prompt_template:str)-> dict:
        """
        Make a single LLM call to generate: 
        -primary_category
        -secondary_category
        - descriptions(1-2)

        Args:
         domain_name: The domain to enrich
         prompt_template: Your pre-defined prompt template

        Returns:
           dict with categories and descriptions

        """
        # Replace placeholder in your prompt
        prompt = prompt_template.replace("{domain_name}", domain_name)

        messages = [
            SystemMessage(content="You are a domain analysis expert. Return valid JSON only."),
            HumanMessage(content=prompt)
        ]

        response = self.llm.invoke(messages)

        # Parse JSON Response
        try:
            result = json.loads(response.content)

            # Validate structure
            assert "primary_category" in result
            assert "secondary_category" in result
            assert "descriptions" in result
            assert isinstance(result["descriptions"], list)
            assert len(result["descriptions"]) >= 1
            assert result["primary_category"] != result["secondary_category"]

            return result

        except (json.JSONDecodeError, AssertionError) as e:
            raise ValueError(f"Invalid LLM response format: {e}\nResponse: {response.content}")