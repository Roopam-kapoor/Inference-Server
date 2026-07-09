from locust import HttpUser, task, between

class EmbeddingUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def embed(self):
        self.client.post(
            "/v1/embeddings",
            json={
                "input": "benchmark test sentence.",
                "model": "all-MiniLM-L6-v2"
            },
            headers={
                "x-api-key": "dev-secret-key"
            }
        )