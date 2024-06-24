import os
import requests
import json
import logging
import argparse
from requests.auth import HTTPBasicAuth

# Environment variables
ENV = os.getenv('ENV')
CMR_USER = os.getenv('CMR_USER')
CMR_PASS = os.getenv('CMR_PASS')

def bearer_token() -> str:
    tokens = []
    headers = {'Accept': 'application/json'}
    url = f"https://{'uat.' if ENV == 'uat' else ''}urs.earthdata.nasa.gov/api/users"

    # First just try to get a token that already exists
    try:
        resp = requests.get(url + "/tokens", headers=headers, auth=HTTPBasicAuth(CMR_USER, CMR_PASS))
        response_content = json.loads(resp.content)

        for x in response_content:
            tokens.append(x['access_token'])

    except Exception:  # noqa E722
        logging.warning("Error getting the token - check user name and password", exc_info=True)

    # No tokens exist, try to create one
    if not tokens:
        try:
            resp = requests.post(url + "/token", headers=headers, auth=HTTPBasicAuth(CMR_USER, CMR_PASS))
            response_content = json.loads(resp.content)
            tokens.append(response_content['access_token'])
        except Exception:  # noqa E722
            logging.warning("Error getting the token - check user name and password", exc_info=True)

    # If still no token, then we can't do anything
    if not tokens:
        raise RuntimeError("Unable to get bearer token from EDL")

    return next(iter(tokens))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Update the service image tag.")
    parser.add_argument("--tag", help="The new tag version to update.", required=True)
    args = parser.parse_args()

    url = f"https://harmony.{'uat.' if ENV == 'uat' else ''}earthdata.nasa.gov/service-image-tag/#podaac-l2-subsetter"
    token = bearer_token()

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-type": "application/json"
    }
    data = {
        "tag": args.tag
    }

    """
    response = requests.put(url, headers=headers, json=data)

    print(response.status_code)
    try:
        print(response.json())
    except json.JSONDecodeError:
        print("Response content is not in JSON format")
    """
