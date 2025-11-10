"""
LLM-based route parser for converting natural language descriptions to route configurations.
Supports multiple providers: OpenAI, Anthropic, Ollama (local), and Transformers (local).
"""

import json
import os
import re
from typing import Dict, List, Tuple, Optional
from PySide6.QtCore import QRectF

# OpenAI support
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Anthropic support
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Ollama support (local LLM)
try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Transformers support (fully local)
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class LLMRouteParser:
    """Parser that uses LLM to convert natural language to route configurations."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        provider: str = "ollama",
        model_name: Optional[str] = None,
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        Initialize LLM route parser.
        
        Args:
            api_key: API key for LLM provider (or use environment variable)
            provider: 'openai', 'anthropic', 'ollama' (local), or 'transformers' (fully local)
            model_name: Model name (required for ollama/transformers, optional for others)
            ollama_base_url: Base URL for Ollama API (default: http://localhost:11434)
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        
        if self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package not installed. Install with: pip install openai")
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
            self.client = openai.OpenAI(api_key=self.api_key)
        
        elif self.provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic package not installed. Install with: pip install anthropic")
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable.")
            self.client = Anthropic(api_key=self.api_key)
        
        elif self.provider == "ollama":
            if not OLLAMA_AVAILABLE:
                raise ImportError("requests package not installed. Install with: pip install requests")
            # Check if Ollama is running
            try:
                response = requests.get(f"{ollama_base_url}/api/tags", timeout=2)
                if response.status_code != 200:
                    raise ConnectionError("Ollama server is not responding correctly")
            except requests.exceptions.RequestException as e:
                raise ConnectionError(
                    f"Ollama server not found at {ollama_base_url}. "
                    f"Error: {str(e)}\n"
                    "Please install and start Ollama: https://ollama.ai/\n"
                    "After installation, run: ollama pull llama3.1:8b"
                )
            # Use default model if not specified
            if not model_name:
                self.model_name = "llama3.1:8b"  # Good balance of quality and speed
            else:
                self.model_name = model_name
            
            # Check if model exists
            try:
                models_response = requests.get(f"{ollama_base_url}/api/tags", timeout=2)
                if models_response.status_code == 200:
                    available_models = [model.get("name", "") for model in models_response.json().get("models", [])]
                    if self.model_name not in available_models:
                        raise ValueError(
                            f"Model '{self.model_name}' not found in Ollama. "
                            f"Available models: {', '.join(available_models) if available_models else 'None'}\n"
                            f"Please pull the model: ollama pull {self.model_name}"
                        )
            except requests.exceptions.RequestException:
                # If we can't check, continue anyway - will fail later with better error
                pass
        
        elif self.provider == "transformers":
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "transformers package not installed. Install with: "
                    "pip install transformers torch"
                )
            # Use default model if not specified
            if not model_name:
                self.model_name = "microsoft/Phi-3-mini-4k-instruct"  # Small, efficient model
            else:
                self.model_name = model_name
            
            # Load model and tokenizer
            print(f"Loading model {self.model_name}... This may take a moment.")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            if not torch.cuda.is_available():
                self.model = self.model.to("cpu")
            print("Model loaded successfully.")
        
        else:
            raise ValueError(
                f"Unknown provider: {provider}. Use 'openai', 'anthropic', 'ollama', or 'transformers'"
            )
    
    def parse_description(
        self,
        description: str,
        source_areas: List[Tuple[QRectF, str]],
        target_areas: List[Tuple[QRectF, str]],
        network_parser
    ) -> Dict:
        """
        Parse natural language description into route configuration.
        
        Args:
            description: Natural language description of traffic behavior
            source_areas: List of (rect, id) tuples for source areas
            target_areas: List of (rect, id) tuples for target areas
            network_parser: NetworkParser instance for network information
        
        Returns:
            Dictionary containing parsed route configuration
        """
        # Build prompt
        prompt = self._build_prompt(description, source_areas, target_areas, network_parser)
        
        # Call LLM based on provider
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
        
        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4096,
                system=self._get_system_prompt(),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            # Extract JSON from response
            content = response.content[0].text
            result = self._extract_json(content)
        
        elif self.provider == "ollama":
            # Call Ollama API using chat endpoint
            response = requests.post(
                f"{self.ollama_base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            result_data = response.json()
            content = result_data.get("message", {}).get("content", "")
            if not content:
                # Fallback to old API format
                content = result_data.get("response", "")
            result = self._extract_json(content)
        
        elif self.provider == "transformers":
            # Use transformers for local inference
            full_prompt = f"System: {self._get_system_prompt()}\n\nUser: {prompt}\n\nAssistant:"
            
            # Tokenize
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=1024,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the assistant's response
            if "Assistant:" in generated_text:
                content = generated_text.split("Assistant:")[-1].strip()
            else:
                content = generated_text[len(full_prompt):].strip()
            
            result = self._extract_json(content)
        
        return result
    
    def _extract_json(self, content: str) -> Dict:
        """Extract JSON from LLM response, handling various formats."""
        # Try to find JSON in markdown code blocks
        json_patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'(\{.*\})',  # Just look for JSON object
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
        
        # If no JSON found, try parsing the entire content
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Last resort: try to find JSON-like structure
            # Find first { and last }
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(content[start:end+1])
                except json.JSONDecodeError:
                    pass
        
        raise ValueError(f"Could not extract valid JSON from LLM response: {content[:200]}")
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM."""
        return """You are a traffic simulation expert that converts natural language descriptions 
of traffic patterns into structured route configurations for SUMO traffic simulation.

Your task is to parse user descriptions and extract:
1. Time periods (weekdays, weekends, specific hours, etc.)
2. Route patterns (source areas to target areas)
3. Vehicle counts and distributions
4. Special events or conditions

Return your response as a JSON object with the following structure:
{
    "time_periods": [
        {
            "name": "Morning Rush Hour",
            "type": "weekday",
            "start": "07:00",
            "end": "09:00",
            "days": ["monday", "tuesday", "wednesday", "thursday", "friday"]
        }
    ],
    "route_patterns": [
        {
            "name": "Morning Commute",
            "time_period": "Morning Rush Hour",
            "sources": ["source_1", "source_2"],
            "targets": ["target_1", "target_2"],
            "vehicle_count": 1000,
            "distribution": "uniform",
            "vehicle_types": ["passenger"]
        }
    ],
    "special_events": [
        {
            "name": "Football Game",
            "type": "weekend",
            "time": "14:00",
            "sources": ["source_5"],
            "targets": ["target_5"],
            "vehicle_count": 500,
            "duration": 1800
        }
    ]
}

Be precise and extract all relevant information from the user's description."""
    
    def _build_prompt(
        self,
        description: str,
        source_areas: List[Tuple[QRectF, str]],
        target_areas: List[Tuple[QRectF, str]],
        network_parser
    ) -> str:
        """Build prompt for LLM."""
        # Build area information
        source_info = []
        for i, (rect, area_id) in enumerate(source_areas, 1):
            source_info.append(f"Area {i} (ID: {area_id}): Bounds ({rect.x():.1f}, {rect.y():.1f}) to ({rect.x() + rect.width():.1f}, {rect.y() + rect.height():.1f})")
        
        target_info = []
        for i, (rect, area_id) in enumerate(target_areas, 1):
            target_info.append(f"Area {i} (ID: {area_id}): Bounds ({rect.x():.1f}, {rect.y():.1f}) to ({rect.x() + rect.width():.1f}, {rect.y() + rect.height():.1f})")
        
        prompt = f"""Parse the following traffic behavior description and convert it to a structured route configuration.

User Description:
{description}

Available Source Areas:
{chr(10).join(source_info)}

Available Target Areas:
{chr(10).join(target_info)}

Network Information:
- Number of edges: {len(network_parser.get_edges())}
- Number of nodes: {len(network_parser.get_nodes())}

Instructions:
1. Identify all time periods mentioned (weekdays, weekends, specific hours, etc.)
2. Map source/target area references (e.g., "areas 1, 2" refers to the first two source areas)
3. Extract vehicle counts and distributions
4. Identify special events or conditions
5. Return a complete JSON configuration following the specified structure

Return only valid JSON, no additional text."""
        
        return prompt

