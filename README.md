# HGS – Heuristické prehľadávanie: od Manhattanu k lineárnemu konfliktu

**Autor:** Yuvchenko Liliia  
**Dátum:** 06.05.2026

---

## Popis projektu

Tento projekt demonštruje princíp **HGS (General-to-Specific Heuristic Search)** – prechod od všeobecnej heuristiky k špecifickejšej – na probléme **N-puzzle** (8-puzzle a 15-puzzle) s využitím algoritmu **IDA\*** (Iterative Deepening A\*).

Projekt obsahuje interaktívnu **Streamlit aplikáciu**, Jupyter notebook s experimentmi a zdrojový kód algoritmov.

---

## Štruktúra projektu

```
hgs_project/
├── streamlit_app.py             # Interaktívna Streamlit aplikácia
├── data/
│   └── generate_dataset.py      # Generovanie CSV datasetov puzzle stavov
├── figures/
│   ├── experiment_results.csv   # Výsledky experimentov
│   ├── heuristic_comparison.png # Graf porovnania heuristík (uzly)
│   └── time_comparison.png      # Graf porovnania časovej náročnosti
├── notebooks/
│   └── notebook.ipynb           # Jupyter notebook s experimentmi
├── src/
│   ├── __init__.py
│   ├── cli.py                   # CLI rozhranie
│   ├── heuristics.py            # Manhattan a lineárny konflikt
│   ├── ida_star.py              # IDA* solver
│   ├── preprocessing.py         # Načítanie a príprava dát
│   ├── puzzle_utils.py          # Generovanie stavov, sukcesorov, formátovanie
│   └── validation.py            # Validácia vstupných stavov
├── tests/
│   └── test_hgs.py              # Unit testy
├── README.md
└── requirements.txt
```

---

## Inštalácia

```bash
# Klonovanie repozitára
git clone <repo-url>
cd hgs_project

# Vytvorenie virtuálneho prostredia
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
.venv\Scripts\activate          # Windows

# Inštalácia závislostí
pip install streamlit pandas numpy matplotlib scikit-learn
```

---

## Spustenie

### Streamlit aplikácia

```bash
streamlit run app/streamlit_app.py
```

### CLI rozhranie

```bash
# Náhodný stav 3×3 s Manhattan heuristikou
python -m src.cli --size 3 --heuristic manhattan

# Vlastný stav s lineárnym konfliktom
python -m src.cli --size 3 --heuristic linear_conflict --start 1 2 3 4 5 6 7 0 8

# Stav so zaručenou minimálnou hĺbkou riešenia
python -m src.cli --size 3 --min-depth 20 --verbose
```

### Jupyter notebook

```bash
jupyter notebook notebooks/notebook.ipynb
```

---

## Algoritmy a heuristiky

### IDA\* (Iterative Deepening A\*)

Pamäťovo efektívny algoritmus kombinujúci výhody A\* (heuristika) a iteratívneho prehlbovania (bez uzavretého zoznamu). Pracuje s rastúcim limitom `f = g + h` a prehľadáva stavový priestor do hĺbky.

### Manhattan vzdialenosť

Všeobecná heuristika – súčet vzdialeností každej dlaždice od jej cieľovej pozície po riadkoch a stĺpcoch.

### Lineárny konflikt

Špecifickejšia heuristika – Manhattan + penalizácia za dvojice dlaždíc, ktoré sú v správnom riadku/stĺpci, ale v nesprávnom poradí (každý konflikt pridáva +2 ťahy). Priemerná úspora expandovaných uzlov: **20–40 %**.

---

## Prehľad modulov

| Modul | Popis |
|---|---|
| `src/ida_star.py` | Trieda `IDASolver` – rekurzívne IDA\* prehľadávanie |
| `src/heuristics.py` | Trieda `Heuristic` – statické metódy `manhattan` a `linear_conflict` |
| `src/puzzle_utils.py` | `random_state`, `generate_state_with_min_depth`, `get_successors`, `is_solvable`, `format_state` |
| `src/validation.py` | `validate_state`, `validate_heuristic`, `validate_max_depth` |
| `src/preprocessing.py` | Načítanie datasetov (weather, animals, CSV), diskretizácia, train/test split |
| `src/cli.py` | CLI rozhranie pre spustenie solvera z príkazového riadku |
| `data/generate_dataset.py` | Generovanie CSV datasetov puzzle stavov pre rôzne hĺbky |
| `tests/test_hgs.py` | Unit testy algoritmov a utilít |

---

## Streamlit aplikácia – funkcie

- Výber veľkosti puzzle (3×3 alebo 4×4)
- Výber heuristiky (Manhattan / lineárny konflikt)
- Zadanie vlastného štartovného stavu alebo náhodné generovanie
- Spustenie IDA\* a zobrazenie riešenia krok po kroku
- Porovnanie heuristík (počet expandovaných uzlov, čas)
- Benchmark výkonu v závislosti od hĺbky riešenia (grafy)
- Metriky klasifikácie riešiteľnosti stavov (Random Forest, konfúzna matica)

---

## Výsledky experimentov

Z experimentov v notebooku vyplýva:

- **Lineárny konflikt** expanduje v priemere o 20–40 % menej uzlov ako Manhattan
- Celkový čas výpočtu je pri lineárnom konflikte nižší napriek vyššej cene výpočtu heuristiky
- Klasifikátor riešiteľnosti (Random Forest na základe počtu inverzií) dosahuje **accuracy ~0.988**, **F1-score ~0.989**
- Pre 15-puzzle (4×4) je výpočtová náročnosť výrazne vyššia pri hĺbkach > 40 ťahov

---

## Obmedzenia

- Pre 15-puzzle a hĺbky > 40 ťahov môže IDA\* trvať niekoľko minút
- Validácia riešiteľnosti pre 4×4 je v `validate_state` dočasne vypnutá

## Možné rozšírenia

- Implementácia pattern database heuristiky
- Paralelizácia prehľadávania
- Vizualizácia riešenia ako animácia v Streamlit

---

## Licencia

Projekt bol vytvorený ako súčasť kurzu strojového učenia na TUKE.
