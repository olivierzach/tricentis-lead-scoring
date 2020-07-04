import requests

# point to local api host for testing
api_url = 'http://127.0.0.1:5000/'

# create a sample row data
data = {
    'email': 'example@domain.com',
    'feature1': '0',
    'feature2': '0',
    'feature3': '0',
    'feature4': '0',
    'feature5': '0',
    'feature6': '0',
}

# test send traffic to local server
for i in range(200):

    # post request to the local api
    r = requests.post(url=api_url, data=data)

    # show what we get back
    print(r.status_code, r.text)