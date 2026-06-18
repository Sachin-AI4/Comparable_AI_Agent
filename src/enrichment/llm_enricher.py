import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
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
            result = json.loads(self._clean_json(response.content))

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

    @staticmethod
    def _clean_json(content: str) -> str:
        """
        Tolerate common LLM formatting artifacts before json.loads:
        - ```json ... ``` / ``` ... ``` code fences
        - doubled braces {{ ... }} (some models echo the prompt's escaping)
        - leading/trailing prose around the JSON object
        Only normalizes; a clean response is returned unchanged.
        """
        if not content:
            return content
        s = content.strip()
        # strip code fences
        if s.startswith("```"):
            s = s.split("```", 2)[1] if s.count("```") >= 2 else s.strip("`")
            if s.lstrip().lower().startswith("json"):
                s = s.lstrip()[4:]
            s = s.strip()
        # slice to the outermost JSON object
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end != -1 and end > start:
            s = s[start:end + 1]
        # collapse doubled braces -> single
        if "{{" in s or "}}" in s:
            s = s.replace("{{", "{").replace("}}", "}")
        return s