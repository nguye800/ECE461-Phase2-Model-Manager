from metric import BaseMetric  # pyright: ignore[reportMissingTypeStubs]
import requests
from dotenv import load_dotenv
import os, re
from types import SimpleNamespace
import tempfile
import subprocess
import sys
import boto3, os, json, re, tempfile, subprocess, sys
from dotenv import load_dotenv

BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "meta.llama2-13b-chat-v1")

brt = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

huggingface_pattern = re.compile(r"(?:https?://)?(?:www\.)?huggingface\.co/([^/]+)/([^/]+)/?$")

class ReproducibilityMetric(BaseMetric):
    metric_name: str = "reproducibility"

    def __init__(self):
        self.response = None
        self.model_card = None
        self.code_snippets = []
        self.execution_result = None
        super().__init__()

    def _get_owner_model(self):
        """Extract owner and model name from HuggingFace URL."""
        if self.url is None or self.url.model is None:
            return None, None
        m = huggingface_pattern.match(self.url.model)
        if not m:
            raise ValueError("invalid HuggingFace URL")
        return m.group(1), m.group(2)

    def _fetch_model_card(self, owner, model_name):
        """Fetch the README.md (model card) from HuggingFace."""
        url = f"https://huggingface.co/{owner}/{model_name}/raw/main/README.md"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Error fetching model card: {e}")
            return None

    def _extract_code_snippets(self, markdown_text):
        """Extract Python code blocks from markdown."""
        if not markdown_text:
            return []
        
        # Match code blocks with python or py language specifier
        pattern = r"```(?:python|py)\s*\n(.*?)```"
        matches = re.findall(pattern, markdown_text, re.DOTALL)
        
        # Also try generic code blocks that look like Python
        if not matches:
            generic_pattern = r"```\s*\n(.*?)```"
            potential_matches = re.findall(generic_pattern, markdown_text, re.DOTALL)
            # Filter for Python-like code (contains import, from, def, class, etc.)
            python_keywords = ['import', 'from', 'def', 'class', 'if __name__']
            matches = [
                m for m in potential_matches 
                if any(keyword in m for keyword in python_keywords)
            ]
        
        return matches

    def _test_code_execution(self, code_snippet):
        """
        Test if a code snippet runs successfully.
        Returns: 'success', 'failure', or 'error'
        """
        # Create a temporary Python file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False
        ) as f:
            temp_file = f.name
            f.write(code_snippet)
        
        try:
            # Try to run the code with a timeout
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Check if execution was successful
            if result.returncode == 0:
                return 'success'
            else:
                return 'failure'
                
        except subprocess.TimeoutExpired:
            return 'timeout'
        except Exception as e:
            print(f"Execution error: {e}")
            return 'error'
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass

    def _use_llm_to_debug(self, code_snippet, error_message):
        """
        Use AWS Bedrock (Claude Sonnet) to attempt to fix code that doesn't run.
        Returns: (fixed_code, success_bool)
        """
        load_dotenv()

        prompt = (
            "The following Python snippet from a HuggingFace model card failed to run. "
            "Return ONLY the corrected Python code, with no explanations and no markdown fences.\n\n"
            f"Code:\n```python\n{code_snippet}\n```\n\n"
            f"Error:\n{error_message}\n"
        )

        try:
            # Prepare Anthropic Claude request payload for invoke_model
            body = {
                "prompt": f"<s>[INST] {prompt} [/INST]",
                "max_gen_len": 1500,
                "temperature": 0.0,
                "top_p": 0.9,
            }

            res = brt.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body)
            )

            # Body is a streaming payload; read and parse JSON
            raw = res.get("body")
            payload = json.loads(raw.read().decode("utf-8")) if hasattr(raw, "read") else json.loads(raw)

            # Extract text segments from Anthropic response content
            fixed_code = payload.get("generation")
            if not fixed_code and payload.get("generations"):
                fixed_code = payload["generations"][0].get("text")
            fixed_code = (fixed_code or "").strip()

            # If the model still returned fenced code, strip fences
            if fixed_code.startswith("```"):
                lines = fixed_code.split("\n")
                # Remove first and last fence if present
                if len(lines) >= 2 and lines[0].startswith("```"):
                    core = lines[1:]
                    if core and core[-1].startswith("```"):
                        core = core[:-1]
                    fixed_code = "\n".join(core)

            # Test the fixed code
            result = self._test_code_execution(fixed_code)
            return fixed_code, result == 'success'

        except Exception as e:
            print(f"LLM debugging error (Bedrock): {e}")
            return None, False

    def setup_resources(self, debug=False):
        """Fetch model card and extract code snippets."""
        load_dotenv()
        owner, model_name = self._get_owner_model()
        if owner is None:
            self.response = None
            return super().setup_resources()
        if debug:
            print(f"\n📦 Fetching model card for {owner}/{model_name}...")
        self.model_card = self._fetch_model_card(owner, model_name)
        
        if not self.model_card:
            if debug:
                print("❌ Could not fetch model card")
            self.response = {
                "has_model_card": False,
                "code_snippets_found": 0,
                "execution_results": []
            }
            return super().setup_resources()
        
        if debug:
            print("✅ Model card fetched")
        
        # Extract code snippets
        self.code_snippets = self._extract_code_snippets(self.model_card)
        if debug:
            print(f"📝 Found {len(self.code_snippets)} code snippet(s)")
        
        if len(self.code_snippets) == 0:
            self.response = {
                "has_model_card": True,
                "code_snippets_found": 0,
                "execution_results": []
            }
            return super().setup_resources()
        
        # Test each code snippet
        execution_results = []
        for i, snippet in enumerate(self.code_snippets):
            if debug:
                print(f"\n🧪 Testing code snippet {i+1}/{len(self.code_snippets)}...")
                print(f"   Code length: {len(snippet)} characters")
            
            # First attempt: run as-is
            result = self._test_code_execution(snippet)

            if debug:
                print(f"   Initial result: {result}")
            
            snippet_result = {
                "snippet_index": i,
                "initial_result": result,
                "runs_without_changes": result == 'success',
                "runs_with_debugging": False,
                "final_result": result
            }
            
            # If it failed, try debugging with Claude
            if result != 'success':
                if debug:
                    print("   🔧 Attempting to debug with Claude...")
                fixed_code, debug_success = self._use_llm_to_debug(
                    snippet, 
                    f"Initial execution result: {result}"
                )
                
                if debug_success:
                    if debug:
                        print("   ✅ Code runs after debugging!")
                    snippet_result["runs_with_debugging"] = True
                    snippet_result["final_result"] = "success_after_debug"
                else:
                    if debug:
                        print("   ❌ Code still doesn't run after debugging")
                    snippet_result["final_result"] = "failure_after_debug"
            else:
                if debug:
                    print("   ✅ Code runs without changes!")
            
            execution_results.append(snippet_result)
        
        self.response = {
            "has_model_card": True,
            "code_snippets_found": len(self.code_snippets),
            "execution_results": execution_results
        }
        
        return super().setup_resources()

    def calculate_score(self) -> float:
        """
        Calculate reproducibility score:
        - 0.0: No code or doesn't run
        - 0.5: Runs with debugging
        - 1.0: Runs without changes
        """
        owner, model_name = self._get_owner_model()
        if owner is None or model_name is None:
            return -1.0  # No valid HuggingFace URL

        if not self.response:
            self.setup_resources()

        if not self.response["has_model_card"]:
            return 0.0  # No model card
        
        if self.response["code_snippets_found"] == 0:
            return 0.0  # No code snippets found
        
        # Check execution results
        execution_results = self.response["execution_results"]
        
        # If any snippet runs without changes, score is 1.0
        if any(r["runs_without_changes"] for r in execution_results):
            return 1.0
        
        # If any snippet runs with debugging, score is 0.5
        if any(r["runs_with_debugging"] for r in execution_results):
            return 0.5
        
        # Nothing runs
        return 0.0


if __name__ == "__main__":  # pragma: no cover
    load_dotenv()
    
    if os.getenv("GEN_AI_STUDIO_API_KEY") is None:
        print("Warning: GEN_AI_STUDIO_API_KEY not set. Debugging functionality will be disabled.")

    # Test with a well-known model
    test_url = "https://huggingface.co/google-bert/bert-base-uncased"
    url_obj = SimpleNamespace(model=test_url)

    metric = ReproducibilityMetric()
    metric.set_url(url_obj)

    print("Running setup_resources() ...")
    metric.setup_resources()

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    score = metric.calculate_score()
    
    if metric.response:
        print(f"\n📊 Reproducibility Analysis:")
        print(f"   Model card found: {metric.response['has_model_card']}")
        print(f"   Code snippets found: {metric.response['code_snippets_found']}")
        
        if metric.response['code_snippets_found'] > 0:
            print(f"\n   Execution Results:")
            for result in metric.response['execution_results']:
                idx = result['snippet_index']
                print(f"\n   Snippet {idx + 1}:")
                print(f"   ├─ Runs without changes: {result['runs_without_changes']}")
                print(f"   ├─ Runs with debugging: {result['runs_with_debugging']}")
                print(f"   └─ Final result: {result['final_result']}")
        
        print(f"\n✅ Reproducibility Score: {score:.1f}")
        if score == 1.0:
            print("   (Code runs perfectly without modifications)")
        elif score == 0.5:
            print("   (Code runs after automated debugging)")
        else:
            print("   (Code doesn't run or no code found)")
    else:
        print(f"\n✅ Reproducibility Score: {score:.1f}")
