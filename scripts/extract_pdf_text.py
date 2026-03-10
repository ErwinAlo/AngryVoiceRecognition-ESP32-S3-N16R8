import sys
from pathlib import Path

path = Path(__file__).resolve().parent.parent / 'docs' / 'Plan-de-Trabajo-Erwin Andrés López Ortega.pdf'

print('PDF path:', path)
print('Exists:', path.exists())
print('Size:', path.stat().st_size if path.exists() else 'N/A')

try:
    from pypdf import PdfReader
except ImportError as e:
    print('pypdf not installed, attempting to install...')
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pypdf'])
    from pypdf import PdfReader

reader = PdfReader(str(path))
print('Pages:', len(reader.pages))

# Additionally, show some strings from the TFLite model file (names, ops, etc.)
import re

model_path = Path(__file__).resolve().parent.parent / 'CNN_Python' / 'modelo_tesis_binario_Angry_finalv5.tflite'
print('\nModel path:', model_path)
print('Model size:', model_path.stat().st_size)

with open(model_path, 'rb') as f:
    data = f.read()

strings = re.findall(b'[ -~]{4,}', data)
print('\nSample ASCII strings in the .tflite (first 50 unique):')
seen = set()
count = 0
for s in strings:
    if s in seen:
        continue
    seen.add(s)
    txt = s.decode('ascii', errors='ignore')
    if any(c.isalpha() for c in txt):
        print(txt)
        count += 1
        if count >= 50:
            break

for i in range(min(5, len(reader.pages))):
    page = reader.pages[i]
    text = page.extract_text()
    print('\n--- page', i+1, '---')
    if text is None:
        print('[no extractable text]')
    else:
        print(text[:2000])
