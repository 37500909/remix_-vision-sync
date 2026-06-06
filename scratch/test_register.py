import requests
import base64

# Create a minimal 1x1 transparent/black pixel base64 image
b64_img = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

payload = {
    'first_name': 'Test',
    'last_name': 'User',
    'dni': '99999999',
    'image': b64_img
}

try:
    r = requests.post('http://localhost:5000/register', json=payload)
    print("Status code:", r.status_code)
    print("Response:", r.json())
except Exception as e:
    print("Request failed:", e)
