import urllib.request, json
req = urllib.request.Request('http://127.0.0.1:5000/admin/calibrate', data=b'{}', headers={'Content-Type':'application/json'})
try:
    resp = urllib.request.urlopen(req, timeout=120)
    print(resp.read().decode())
except Exception as e:
    import traceback
    traceback.print_exc()
    print('ERROR', e)
