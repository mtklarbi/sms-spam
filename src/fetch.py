import requests, zipfile, io, pathlib

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
pathlib.Path("data").mkdir(exist_ok=True)
r=requests.get(URL, timeout=60); r.raise_for_status()
zipfile.ZipFile(io.BytesIO(r.content)).extractall("data")
print("Extracted. Except data/SMSSpamCollection")