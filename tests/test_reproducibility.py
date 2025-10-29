import unittest
from unittest.mock import patch, MagicMock
from src.metrics.reproducibility import ReproducibilityMetric
from src.metric import ModelURLs

class TestReproducibilityMetric(unittest.TestCase):
    metric_instance: ReproducibilityMetric

    def setUp(self):
        self.metric_instance = ReproducibilityMetric()

    # URL parsing tests

    def test_invalid_url(self):
        urls = ModelURLs(model="sdvx.org")
        self.metric_instance.set_url(urls)
        with self.assertRaises(ValueError):
            self.metric_instance._get_owner_model()

    def test_valid_huggingface_url(self):
        urls = ModelURLs(model="https://huggingface.co/google-bert/bert-base-uncased")
        self.metric_instance.set_url(urls)
        owner, model = self.metric_instance._get_owner_model()
        self.assertEqual(owner, "google-bert")
        self.assertEqual(model, "bert-base-uncased")

    def test_huggingface_url_without_https(self):
        urls = ModelURLs(model="huggingface.co/facebook/bart-large")
        self.metric_instance.set_url(urls)
        owner, model = self.metric_instance._get_owner_model()
        self.assertEqual(owner, "facebook")
        self.assertEqual(model, "bart-large")

    def test_huggingface_url_with_trailing_slash(self):
        urls = ModelURLs(model="https://huggingface.co/openai/whisper-tiny/")
        self.metric_instance.set_url(urls)
        owner, model = self.metric_instance._get_owner_model()
        self.assertEqual(owner, "openai")
        self.assertEqual(model, "whisper-tiny")

    def test_huggingface_url_with_www(self):
        urls = ModelURLs(model="https://www.huggingface.co/microsoft/deberta-v3-base")
        self.metric_instance.set_url(urls)
        owner, model = self.metric_instance._get_owner_model()
        self.assertEqual(owner, "microsoft")
        self.assertEqual(model, "deberta-v3-base")

    def test_none_url(self):
        urls = ModelURLs(model=None)
        self.metric_instance.set_url(urls)
        owner, model = self.metric_instance._get_owner_model()
        self.assertIsNone(owner)
        self.assertIsNone(model)

    # Code extraction tests

    def test_extract_code_snippets_python_tag(self):
        markdown = """
# Model Card

Here's how to use it:

```python
import torch
model = torch.load('model.pt')
```

Some more text.
"""
        snippets = self.metric_instance._extract_code_snippets(markdown)
        self.assertEqual(len(snippets), 1)
        self.assertIn("import torch", snippets[0])
        self.assertIn("model = torch.load('model.pt')", snippets[0])

    def test_extract_code_snippets_py_tag(self):
        markdown = """
```py
from transformers import AutoModel
model = AutoModel.from_pretrained('bert-base')
```
"""
        snippets = self.metric_instance._extract_code_snippets(markdown)
        self.assertEqual(len(snippets), 1)
        self.assertIn("from transformers", snippets[0])

    def test_extract_code_snippets_generic_with_python_keywords(self):
        markdown = """
```
import numpy as np
def process_data(x):
    return np.mean(x)
```
"""
        snippets = self.metric_instance._extract_code_snippets(markdown)
        self.assertEqual(len(snippets), 1)
        self.assertIn("import numpy", snippets[0])

    def test_extract_code_snippets_multiple_blocks(self):
        markdown = """
```python
import torch
```

Some text here.

```python
from transformers import pipeline
```
"""
        snippets = self.metric_instance._extract_code_snippets(markdown)
        self.assertEqual(len(snippets), 2)

    def test_extract_code_snippets_no_code(self):
        markdown = """
# Model Card

This is a great model with no code examples.
"""
        snippets = self.metric_instance._extract_code_snippets(markdown)
        self.assertEqual(len(snippets), 0)

    def test_extract_code_snippets_empty_markdown(self):
        snippets = self.metric_instance._extract_code_snippets("")
        self.assertEqual(len(snippets), 0)

    def test_extract_code_snippets_none_markdown(self):
        snippets = self.metric_instance._extract_code_snippets(None)
        self.assertEqual(len(snippets), 0)

    def test_extract_code_snippets_ignores_non_python(self):
        markdown = """
```javascript
console.log('Hello');
```

```bash
pip install torch
```
"""
        snippets = self.metric_instance._extract_code_snippets(markdown)
        # Should find 0 snippets since neither has Python keywords
        self.assertEqual(len(snippets), 0)

    # Code execution tests

    def test_code_execution_success(self):
        code = "print('Hello, World!')"
        result = self.metric_instance._test_code_execution(code)
        self.assertEqual(result, 'success')

    def test_code_execution_failure(self):
        code = "import nonexistent_module"
        result = self.metric_instance._test_code_execution(code)
        self.assertEqual(result, 'failure')

    def test_code_execution_syntax_error(self):
        code = "print('unclosed string"
        result = self.metric_instance._test_code_execution(code)
        self.assertEqual(result, 'failure')

    def test_code_execution_runtime_error(self):
        code = "x = 1 / 0"
        result = self.metric_instance._test_code_execution(code)
        self.assertEqual(result, 'failure')

    def test_code_execution_creates_temp_file(self):
        code = "x = 42"
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_file = MagicMock()
            mock_file.name = '/tmp/test.py'
            mock_temp.return_value.__enter__.return_value = mock_file
            
            with patch('subprocess.run') as mock_run:
                mock_run.return_value.returncode = 0
                result = self.metric_instance._test_code_execution(code)
                
                mock_temp.assert_called_once()
                self.assertTrue(mock_temp.call_args[1]['delete'] is False)

    # Score calculation tests

    def test_calculate_score_no_model_card(self):
        urls = ModelURLs(model="https://huggingface.co/owner/model")
        self.metric_instance.set_url(urls)
        self.metric_instance.response = {
            "has_model_card": False,
            "code_snippets_found": 0,
            "execution_results": []
        }
        score = self.metric_instance.calculate_score()
        self.assertEqual(score, 0.0)

    def test_calculate_score_no_code_snippets(self):
        urls = ModelURLs(model="https://huggingface.co/owner/model")
        self.metric_instance.set_url(urls)
        self.metric_instance.response = {
            "has_model_card": True,
            "code_snippets_found": 0,
            "execution_results": []
        }
        score = self.metric_instance.calculate_score()
        self.assertEqual(score, 0.0)

    def test_calculate_score_runs_without_changes(self):
        urls = ModelURLs(model="https://huggingface.co/owner/model")
        self.metric_instance.set_url(urls)
        self.metric_instance.response = {
            "has_model_card": True,
            "code_snippets_found": 1,
            "execution_results": [
                {
                    "snippet_index": 0,
                    "initial_result": "success",
                    "runs_without_changes": True,
                    "runs_with_debugging": False,
                    "final_result": "success"
                }
            ]
        }
        score = self.metric_instance.calculate_score()
        self.assertEqual(score, 1.0)

    def test_calculate_score_runs_with_debugging(self):
        urls = ModelURLs(model="https://huggingface.co/owner/model")
        self.metric_instance.set_url(urls)
        self.metric_instance.response = {
            "has_model_card": True,
            "code_snippets_found": 1,
            "execution_results": [
                {
                    "snippet_index": 0,
                    "initial_result": "failure",
                    "runs_without_changes": False,
                    "runs_with_debugging": True,
                    "final_result": "success_after_debug"
                }
            ]
        }
        score = self.metric_instance.calculate_score()
        self.assertEqual(score, 0.5)

    def test_calculate_score_nothing_runs(self):
        urls = ModelURLs(model="https://huggingface.co/owner/model")
        self.metric_instance.set_url(urls)
        self.metric_instance.response = {
            "has_model_card": True,
            "code_snippets_found": 1,
            "execution_results": [
                {
                    "snippet_index": 0,
                    "initial_result": "failure",
                    "runs_without_changes": False,
                    "runs_with_debugging": False,
                    "final_result": "failure_after_debug"
                }
            ]
        }
        score = self.metric_instance.calculate_score()
        self.assertEqual(score, 0.0)

    def test_calculate_score_multiple_snippets_one_success(self):
        urls = ModelURLs(model="https://huggingface.co/owner/model")
        self.metric_instance.set_url(urls)
        self.metric_instance.response = {
            "has_model_card": True,
            "code_snippets_found": 2,
            "execution_results": [
                {
                    "snippet_index": 0,
                    "runs_without_changes": False,
                    "runs_with_debugging": False,
                    "final_result": "failure"
                },
                {
                    "snippet_index": 1,
                    "runs_without_changes": True,
                    "runs_with_debugging": False,
                    "final_result": "success"
                }
            ]
        }
        score = self.metric_instance.calculate_score()
        self.assertEqual(score, 1.0)

    def test_calculate_score_multiple_snippets_one_debug_success(self):
        urls = ModelURLs(model="https://huggingface.co/owner/model")
        self.metric_instance.set_url(urls)
        self.metric_instance.response = {
            "has_model_card": True,
            "code_snippets_found": 2,
            "execution_results": [
                {
                    "snippet_index": 0,
                    "runs_without_changes": False,
                    "runs_with_debugging": False,
                    "final_result": "failure"
                },
                {
                    "snippet_index": 1,
                    "runs_without_changes": False,
                    "runs_with_debugging": True,
                    "final_result": "success_after_debug"
                }
            ]
        }
        score = self.metric_instance.calculate_score()
        self.assertEqual(score, 0.5)

    def test_calculate_score_invalid_url(self):
        urls = ModelURLs(model="invalid-url")
        self.metric_instance.set_url(urls)
        with self.assertRaises(ValueError):
            self.metric_instance.calculate_score()

    def test_calculate_score_none_url(self):
        urls = ModelURLs(model=None)
        self.metric_instance.set_url(urls)
        score = self.metric_instance.calculate_score()
        self.assertEqual(score, -1.0)

    # Model card fetching tests

    @patch('src.metrics.reproducibility.requests.get')
    def test_fetch_model_card_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = "# Model Card\nThis is a test model card."
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        card = self.metric_instance._fetch_model_card("owner", "model")
        self.assertIsNotNone(card)
        self.assertIn("Model Card", card)

    @patch('src.metrics.reproducibility.requests.get')
    def test_fetch_model_card_404(self, mock_get):
        mock_get.side_effect = Exception("404 Not Found")
        card = self.metric_instance._fetch_model_card("owner", "nonexistent")
        self.assertIsNone(card)

    @patch('src.metrics.reproducibility.requests.get')
    def test_fetch_model_card_timeout(self, mock_get):
        mock_get.side_effect = Exception("Timeout")
        card = self.metric_instance._fetch_model_card("owner", "model")
        self.assertIsNone(card)

    # LLM debugging tests

    @patch('src.metrics.reproducibility.requests.post')
    def test_llm_debug_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.text = "print('Fixed code')"
        mock_post.return_value = mock_response

        with patch.dict('os.environ', {'GEN_AI_STUDIO_API_KEY': 'fake_key'}):
            with patch.object(self.metric_instance, '_test_code_execution', return_value='success'):
                fixed_code, success = self.metric_instance._use_llm_to_debug(
                    "print('broken code')", 
                    "SyntaxError"
                )
                self.assertTrue(success)
                self.assertIsNotNone(fixed_code)

    @patch('src.metrics.reproducibility.requests.post')
    def test_llm_debug_failure(self, mock_post):
        mock_response = MagicMock()
        mock_response.text = "print('Still broken')"
        mock_post.return_value = mock_response

        with patch.dict('os.environ', {'GEN_AI_STUDIO_API_KEY': 'fake_key'}):
            with patch.object(self.metric_instance, '_test_code_execution', return_value='failure'):
                fixed_code, success = self.metric_instance._use_llm_to_debug(
                    "print('broken code')", 
                    "SyntaxError"
                )
                self.assertFalse(success)

    def test_llm_debug_missing_api_key(self):
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(RuntimeError):
                self.metric_instance._use_llm_to_debug("code", "error")

    @patch('src.metrics.reproducibility.requests.post')
    def test_llm_debug_removes_markdown(self, mock_post):
        mock_response = MagicMock()
        mock_response.text = "```python\nprint('Hello')\n```"
        mock_post.return_value = mock_response

        with patch.dict('os.environ', {'GEN_AI_STUDIO_API_KEY': 'fake_key'}):
            with patch.object(self.metric_instance, '_test_code_execution', return_value='success'):
                fixed_code, success = self.metric_instance._use_llm_to_debug(
                    "broken", 
                    "error"
                )
                self.assertNotIn("```", fixed_code)
                self.assertIn("print('Hello')", fixed_code)

    # Integration tests with mocking

    @patch('src.metrics.reproducibility.requests.get')
    def test_setup_resources_full_flow(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = """
# Test Model

```python
print('Hello, World!')
```
"""
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        urls = ModelURLs(model="https://huggingface.co/test/model")
        self.metric_instance.set_url(urls)
        
        self.metric_instance.setup_resources(debug=False)
        
        self.assertIsNotNone(self.metric_instance.response)
        self.assertTrue(self.metric_instance.response["has_model_card"])
        self.assertEqual(self.metric_instance.response["code_snippets_found"], 1)
        self.assertEqual(len(self.metric_instance.response["execution_results"]), 1)

    @patch('src.metrics.reproducibility.requests.get')
    def test_setup_resources_no_code(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = "# Model with no code examples"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        urls = ModelURLs(model="https://huggingface.co/test/model")
        self.metric_instance.set_url(urls)
        
        self.metric_instance.setup_resources(debug=False)
        
        self.assertTrue(self.metric_instance.response["has_model_card"])
        self.assertEqual(self.metric_instance.response["code_snippets_found"], 0)


if __name__ == "__main__":
    unittest.main()