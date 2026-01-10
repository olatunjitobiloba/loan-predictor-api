import requests
url = 'http://127.0.0.1:5000/apispec.json'
try:
    r = requests.get(url, timeout=10)
    print('STATUS', r.status_code)
    print('HEADERS:', r.headers)
    print('\nBODY START:\n')
    print(r.text[:8000])
except Exception as e:
    print('REQUEST ERROR:', e)
