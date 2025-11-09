import json
import base64
import unittest
from src import security


class TestSecurityLambda(unittest.TestCase):
    def test_get_tracks_default_message(self):
        event = {"httpMethod": "GET", "path": "/tracks"}
        response = security.lambda_handler(event, None)
        body = json.loads(response["body"])
        self.assertEqual(response["statusCode"], 200)
        self.assertIn("tracks", body)

    def test_put_authenticate_generates_token(self):
        event = {
            "httpMethod": "PUT",
            "path": "/authenticate",
            "body": json.dumps({"subject": "student123"}),
        }
        response = security.lambda_handler(event, None)
        body = json.loads(response["body"])
        self.assertEqual(response["statusCode"], 201)
        self.assertIn("access_token", body)
        payload_json = base64.urlsafe_b64decode(body["access_token"] + "==").decode(
            "utf-8"
        )
        payload = json.loads(payload_json)
        self.assertEqual(payload["sub"], "student123")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
