import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ida_star import IDASolver
from src.puzzle_utils import random_state, format_state, generate_state_with_min_depth, is_solvable
from src.validation import validate_state

st.set_page_config(page_title="HGS - IDA* Solver", layout="wide")
st.title(" Heuristické prehľadávanie: od Manhattanu k lineárnemu konfliktu")

# ----- Inicializácia session state -----
if 'size' not in st.session_state:
    st.session_state.size = 3
if 'random_state' not in st.session_state:
    st.session_state.random_state = random_state(st.session_state.size, steps=100)
if 'show_metrics' not in st.session_state:
    st.session_state.show_metrics = False
if 'computed_metrics' not in st.session_state:
    st.session_state.computed_metrics = None
if 'depth_results' not in st.session_state:
    st.session_state.depth_results = None

# ----- Bočný panel -----
with st.sidebar:
    st.header("Nastavenia")
    size = st.selectbox("Veľkosť puzzle", [3, 4], format_func=lambda x: f"{x}x{x} ({x*x-1} puzzle)", index=0 if st.session_state.size==3 else 1)
    heuristic = st.selectbox("Heuristika", ["manhattan", "linear_conflict"])
    max_depth = st.number_input("Maximálna hĺbka", min_value=1, value=100)
    run_benchmark = st.checkbox(" Po vyriešení spustiť benchmark (grafy závislosti od hĺbky)", value=True)

    if size != st.session_state.size:
        st.session_state.size = size
        st.session_state.show_metrics = False
        st.session_state.computed_metrics = None
        st.session_state.depth_results = None
        if 'random_state' in st.session_state:
            del st.session_state.random_state
        st.rerun()

    st.divider()
    if st.button(" Generovať nový náhodný stav"):
        st.session_state.random_state = random_state(size, steps=100)
        st.session_state.show_metrics = False
        st.rerun()

# ----- Hlavná časť -----
st.subheader("Startovný stav")

# ----- VLASTNÉ ZADANIE STAVU -----
with st.expander(" Zadať vlastný startovný stav (čísel oddelených medzerou)"):
    default_value = " ".join(map(str, st.session_state.random_state))
    custom_input = st.text_input("Napríklad: 1 2 3 4 5 6 7 8 0 (pre 3x3) alebo 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0 (pre 4x4)", 
                                  value=default_value, key="custom_input")
    col1, col2 = st.columns(2)
    with col1:
        if st.button(" Nastaviť vlastný stav"):
            try:
                custom_state = tuple(map(int, custom_input.split()))
                # Validácia dĺžky
                if len(custom_state) != size*size:
                    st.error(f" Chyba: Počet čísel musí byť {size*size} (zadali ste {len(custom_state)}).")
                # Validácia rozsahu hodnôt
                elif set(custom_state) != set(range(size*size)):
                    st.error(f" Chyba: Stav musí obsahovať všetky čísla od 0 do {size*size-1} bez opakovania.")
                # Validácia riešiteľnosti (použijeme is_solvable priamo)
                elif not is_solvable(custom_state, size):
                    st.error(f" Chyba: Zadaný stav nie je riešiteľný. Skúste prehodiť dve dlaždice alebo použite náhodný generátor.")
                else:
                    st.session_state.random_state = custom_state
                    st.session_state.show_metrics = False
                    st.success(" Vlastný stav bol nastavený!")
                    st.rerun()
            except ValueError:
                st.error(" Chyba: Neplatný formát – zadajte iba čísla oddelené medzerou.")
    with col2:
        if st.button(" Obnoviť náhodný stav"):
            st.session_state.random_state = random_state(size, steps=100)
            st.session_state.show_metrics = False
            st.rerun()

# ----- Zobrazenie aktuálneho stavu -----
if 'random_state' not in st.session_state:
    st.session_state.random_state = random_state(size, steps=100)

start_state = st.session_state.random_state

# Dodatočná kontrola dĺžky (pre istotu)
if len(start_state) != size*size:
    st.error(f"Nesprávna dĺžka stavu: {len(start_state)} namiesto {size*size}. Generujem nový...")
    st.session_state.random_state = random_state(size, steps=100)
    start_state = st.session_state.random_state

st.text(format_state(start_state, size))

# Zobrazenie informácie o riešiteľnosti (voliteľné)
if is_solvable(start_state, size):
    st.success(" Tento stav je riešiteľný")
else:
    st.info(" Riešiteľnosť pre 4x4 puzzle sa overí pri spustení IDA*.")

goal_state = tuple(range(size*size))

# ----- Funkcia na výpočet metrík (klasifikácia) -----
def compute_metrics(size):
    n_samples = 150
    data = []
    for _ in range(n_samples):
        solvable = random_state(size, steps=50)
        inv_s = sum(1 for i in range(len(solvable)) for j in range(i+1, len(solvable))
                    if solvable[i] != 0 and solvable[j] != 0 and solvable[i] > solvable[j])
        data.append([inv_s, 1])

        unsolvable = list(solvable)
        pos = [i for i, val in enumerate(unsolvable) if val != 0]
        if len(pos) >= 2:
            i1, i2 = pos[0], pos[1]
            unsolvable[i1], unsolvable[i2] = unsolvable[i2], unsolvable[i1]
        unsolvable = tuple(unsolvable)
        inv_u = sum(1 for i in range(len(unsolvable)) for j in range(i+1, len(unsolvable))
                    if unsolvable[i] != 0 and unsolvable[j] != 0 and unsolvable[i] > unsolvable[j])
        data.append([inv_u, 0])

    df = pd.DataFrame(data, columns=['inv_count', 'solvable'])
    X = df[['inv_count']]
    y = df['solvable']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return acc, prec, rec, f1, cm

# ----- Funkcia na benchmark podľa hĺbky -----
def benchmark_depth(size, depths=[5,10,15,20,25,30]):
    results = { 'depth': [], 'manhattan_nodes': [], 'linear_nodes': [], 
                'manhattan_time': [], 'linear_time': [] }
    for depth in depths:
        state = generate_state_with_min_depth(size, depth, max_attempts=200)
        # Manhattan
        solver_m = IDASolver(state, tuple(range(size*size)), size, 'manhattan')
        ok_m = solver_m.solve(max_depth=100)
        # Linear conflict
        solver_l = IDASolver(state, tuple(range(size*size)), size, 'linear_conflict')
        ok_l = solver_l.solve(max_depth=100)
        if ok_m and ok_l:
            results['depth'].append(depth)
            results['manhattan_nodes'].append(solver_m.expanded_nodes)
            results['linear_nodes'].append(solver_l.expanded_nodes)
            results['manhattan_time'].append(solver_m.time_taken)
            results['linear_time'].append(solver_l.time_taken)
    return pd.DataFrame(results)

# ----- Spustenie IDA* -----
if st.button(" Spustiť IDA*"):
    try:
        validate_state(start_state, size)
        with st.spinner("Prehľadávam..."):
            solver = IDASolver(start_state, goal_state, size, heuristic)
            ok = solver.solve(max_depth=max_depth)
        if ok:
            st.success(f" Riešenie nájdené! Počet krokov: {len(solver.solution)-1}")
            col1, col2 = st.columns(2)
            col1.metric("Expandované uzly", solver.expanded_nodes)
            col2.metric("Čas (ms)", f"{solver.time_taken*1000:.2f}")
            with st.expander("Zobraziť cestu"):
                for i, state in enumerate(solver.solution):
                    st.text(f"Krok {i}:")
                    st.text(format_state(state, size))

            # Porovnanie heuristík (len pre aktuálny stav)
            other = "linear_conflict" if heuristic == "manhattan" else "manhattan"
            st.subheader(" Porovnanie heuristík (aktuálny stav)")
            solver2 = IDASolver(start_state, goal_state, size, other)
            solver2.solve(max_depth=max_depth)
            df_comp = pd.DataFrame({
                "Heuristika": [heuristic, other],
                "Expandované uzly": [solver.expanded_nodes, solver2.expanded_nodes]
            })
            st.bar_chart(df_comp.set_index("Heuristika"))

            # ----- Benchmark (grafy závislosti od hĺbky) -----
            if run_benchmark:
                with st.spinner("Spúšťam benchmark pre rôzne hĺbky (chvíľu to trvá)..."):
                    df_bench = benchmark_depth(size)
                    st.session_state.depth_results = df_bench
            else:
                st.session_state.depth_results = None

            # ----- Metriky klasifikácie -----
            with st.spinner("Počítam metriky klasifikácie..."):
                acc, prec, rec, f1, cm = compute_metrics(size)
            st.session_state.show_metrics = True
            st.session_state.computed_metrics = (acc, prec, rec, f1, cm)
        else:
            st.error(" Riešenie nebolo nájdené (prekročená max hĺbka alebo neriešiteľný stav)")
            st.session_state.show_metrics = False
    except ValueError as e:
        st.error(f"Chyba validácie: {e}")
        st.session_state.show_metrics = False

# ----- Zobrazenie benchmark grafov (ak existujú) -----
if st.session_state.depth_results is not None and not st.session_state.depth_results.empty:
    df_bench = st.session_state.depth_results
    st.divider()
    st.header(" Výkon v závislosti od hĺbky riešenia")
    
    fig1, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(df_bench['depth'], df_bench['manhattan_nodes'], 'o-', label='Manhattan')
    ax1.plot(df_bench['depth'], df_bench['linear_nodes'], 's-', label='Lineárny konflikt')
    ax1.set_xlabel('Minimálna hĺbka riešenia')
    ax1.set_ylabel('Počet expandovaných uzlov')
    ax1.set_title('Porovnanie počtu uzlov')
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.plot(df_bench['depth'], df_bench['manhattan_time'], 'o-', label='Manhattan')
    ax2.plot(df_bench['depth'], df_bench['linear_time'], 's-', label='Lineárny konflikt')
    ax2.set_xlabel('Minimálna hĺbka riešenia')
    ax2.set_ylabel('Čas výpočtu (s)')
    ax2.set_title('Porovnanie časovej náročnosti')
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

# ----- Zobrazenie metrík klasifikácie -----
if st.session_state.show_metrics and st.session_state.computed_metrics:
    acc, prec, rec, f1, cm = st.session_state.computed_metrics
    st.divider()
    st.header(" Metriky klasifikácie riešiteľnosti stavov")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.3f}")
    col2.metric("Precision", f"{prec:.3f}")
    col3.metric("Recall", f"{rec:.3f}")
    col4.metric("F1-score", f"{f1:.3f}")
    st.write("Konfúzna matica:")
    st.write(pd.DataFrame(cm, index=["Skutočne 0 (nerieš.)", "Skutočne 1 (rieš.)"],
                           columns=["Predikcia 0", "Predikcia 1"]))
    st.caption("Metriky sú vypočítané na základe klasifikácie podľa počtu inverzií (Random Forest).")