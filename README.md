# Weil positivity on [−0.8, 0.8] and [−0.9, 0.9] — certificate package

Sabrina Hackfort · MEGALODON Research · 6 Oct 2026

* `weil_positivity_en.pdf/.tex` — manuscript (English), `weil_positivitaet_de.pdf/.tex` — German version
* `cert_08_even_v2.json`, `cert_08_odd.json`, `cert_09_even_v2.json`, `cert_09_odd.json` — certificate outputs
* `code/zhu_stage.py` — staged certificate (both parities), `code/zhu_cert.py` — monolithic version, `code/hp_scan.py` — independent position-space check

Reproduce (Python ≥ 3.11, `pip install python-flint mpmath`):

```
# L = 0.8, even sector (≈ 3 min); set PARITY=odd for the odd sector
python3 code/zhu_stage.py part  0.8 200 200 400 320 0 12800 e.txt
python3 code/zhu_stage.py final 0.8 200 200 400 320 e.txt
# L = 0.9 (split node range 0..35200 into parts as convenient)
PARITY=odd python3 code/zhu_stage.py part  0.9 550 400 1100 384 0 14000 o1.txt
PARITY=odd python3 code/zhu_stage.py part  0.9 550 400 1100 384 14000 35200 o2.txt
PARITY=odd python3 code/zhu_stage.py final 0.9 550 400 1100 384 o1.txt o2.txt
```
