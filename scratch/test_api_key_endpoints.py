import unittest
import json
from app import app, init_app

class TestApiKeyEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize the app (e.g. load LLM config)
        init_app()
        cls.client = app.test_client()

    def test_llm_status(self):
        resp = self.client.get('/api/llm-status')
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn('provider', data)
        self.assertIn('has_api_key', data)
        self.assertIn('model', data)
        print("LLM Status response:", data)

    def test_set_gemini_key_invalid(self):
        # Empty key
        resp = self.client.post('/api/set-gemini-key', json={"api_key": ""})
        self.assertEqual(resp.status_code, 400)
        
        # Invalid format/fake key
        resp = self.client.post('/api/set-gemini-key', json={"api_key": "fake_invalid_key_12345"})
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.data)
        self.assertFalse(data['ok'])
        self.assertIn('Invalid API key', data['error'])
        print("Invalid key validation response:", data)

if __name__ == '__main__':
    unittest.main()
