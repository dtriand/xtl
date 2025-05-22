import requests
import io
from sphinx.ext.intersphinx._load import InventoryFile

# Download the objects.inv file
url = 'https://docs.pydantic.dev/latest/objects.inv'
resp = requests.get(url)
inv_bytes = resp.content

# Parse the inventory
f = InventoryFile.load(io.BytesIO(inv_bytes), url, lambda base, location: location)
for domain, items in f.items():
    for name, (proj, version, url, dispname) in items.items():
        print(f'{domain}: {name} -> {url}')
