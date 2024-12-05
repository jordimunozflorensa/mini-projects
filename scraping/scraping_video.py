import requests
import re

url = "https://www.vulnhub.com/"

response = requests.get(url)
resultats = response.text

pattern = r"/entry/[\w-]+" 

names = re.findall(pattern, str(resultats))

names = list(set(names))

print('')
for i in range(len(names)):
    print(names[i][7:])
    
print('')
