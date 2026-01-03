from app_v7 import app

with app.test_client() as client:
    resp = client.post('/models/benchmark', json={})
    print('status', resp.status_code)
    print(resp.get_json())
