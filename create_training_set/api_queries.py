import http.client as httplib
import json
import sys
from urllib.parse import quote as urlencode

def mastQuery(request):
    server = "mast.stsci.edu"

    # Check python version
    version = ".".join(map(str, sys.version_info[3]))

    # Create HTTP Header Variables
    headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "text-plain",
            "User-agent": "python-requests/"+version
    }

    # Encode the request as a JSON string
    requestString = json.dumps(request)
    requestString = urlencode(requestString)

    # Open the https connection
    conn = httplib.HTTPSConnection(server)

    # Making the query
    conn.request("POST", "/api/v0/invoke", "request="+requestString, headers)

    # Get the response
    resp = conn.getresponse()
    head = resp.getheaders()
    content = resp.read().decode('utf-8')

    # Close the https connection
    conn.close()

    return head, content

def fetchTICEntry(tic_id):
    print("Fetching TIC entry")
    request = {
        "service": "Mast.Catalogs.Filtered.Tic",
        "format": "json",
        "params": {
            "columns": "ID", # This filter doesn't seem to work but also seems to be required?
            "filters": [
                { "paramName":"ID", "values":[tic_id] }
            ]
        }
    }

    headers, outString = mastQuery(request)
    outData = json.loads(outString)
    if (outData["status"] == 'COMPLETE'):
        return outData["data"]
    else:
        return False
