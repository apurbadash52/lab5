"""
MDP engine for the wall-following robot (SCITOS-G5).
Grid MDP (front zone x left-wall zone), transitions learned from data,
four reward designs solved with Value Iteration.
"""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- grid
FRONT_EDGES = [0.70, 1.00, 1.50, 2.50]
LEFT_EDGES = [0.45, 0.60, 0.80, 1.20]
FRONT_NAMES = ['F0 Danger', 'F1 Close', 'F2 Medium', 'F3 Far', 'F4 Open']
LEFT_NAMES = ['L0 Too close', 'L1 Close', 'L2 Ideal', 'L3 Drifting', 'L4 Lost']
NF, NL = len(FRONT_NAMES), len(LEFT_NAMES)
N_STATES = NF * NL

ACTIONS = ['Move-Forward', 'Slight-Right-Turn', 'Sharp-Right-Turn', 'Slight-Left-Turn']
A_SHORT = {'Move-Forward': 'F', 'Slight-Right-Turn': 'r', 'Sharp-Right-Turn': 'R', 'Slight-Left-Turn': 'l'}
N_ACTIONS = len(ACTIONS)
A_IDX = {a: i for i, a in enumerate(ACTIONS)}
POLICY_NAMES = ['Balanced', 'Safety-First', 'Speed-First', 'Expert-Imitation']


def to_state(front, left):
    return np.digitize(front, FRONT_EDGES) * NL + np.digitize(left, LEFT_EDGES)


def cell(s):
    return divmod(int(s), NL)


def state_label(s):
    f, l = cell(s)
    return f'F{f}/L{l}'


# ---------------------------------------------------------------- data
def load_data(path_or_buffer):
    df = pd.read_csv(path_or_buffer, header=None,
                     names=['SD_front', 'SD_left', 'SD_right', 'SD_back', 'Class'])
    df['Class'] = df['Class'].astype(str).str.strip()
    df = df[df['Class'].isin(ACTIONS)].reset_index(drop=True)
    df['state'] = to_state(df.SD_front.values, df.SD_left.values)
    df['action'] = df['Class'].map(A_IDX)
    return df


# ---------------------------------------------------------------- transitions
def physics_prior(s, a):
    f, l = cell(s)
    moves = {
        'Move-Forward':      {(-1, 0): .35, (0, 0): .55, (0, 1): .05, (0, -1): .05},
        'Slight-Right-Turn': {(0, 1): .40, (0, 0): .45, (1, 0): .15},
        'Sharp-Right-Turn':  {(1, 1): .30, (1, 0): .30, (0, 1): .15, (0, 0): .25},
        'Slight-Left-Turn':  {(0, -1): .45, (0, 0): .40, (-1, 0): .15},
    }[ACTIONS[a]]
    p = np.zeros(N_STATES)
    for (dfr, dl), pr in moves.items():
        nf, nl = min(max(f + dfr, 0), NF - 1), min(max(l + dl, 0), NL - 1)
        p[nf * NL + nl] += pr
    return p


def build_transitions(df, step=5, k_prior=5.0):
    s_arr, a_arr = df.state.values, df.action.values
    counts = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    for t in range(len(df) - step):
        counts[s_arr[t], a_arr[t], s_arr[t + step]] += 1
    P = np.zeros_like(counts)
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            P[s, a] = (counts[s, a] + k_prior * physics_prior(s, a)) / (counts[s, a].sum() + k_prior)
    return P, counts


# ---------------------------------------------------------------- rewards
DESIGNS = {
    'Balanced':     dict(danger=-10, front_close=-2, wall_too_close=-6, ideal=5, wall_lost=-4,
                         fwd_risk={0: -10, 1: -4},
                         act={'Move-Forward': 1, 'Slight-Right-Turn': -0.3, 'Sharp-Right-Turn': -1, 'Slight-Left-Turn': -0.3}),
    'Safety-First': dict(danger=-30, front_close=-6, wall_too_close=-15, ideal=4, wall_lost=-3,
                         fwd_risk={0: -30, 1: -10},
                         act={'Move-Forward': 2.5, 'Slight-Right-Turn': 0, 'Sharp-Right-Turn': -0.3, 'Slight-Left-Turn': 0}),
    'Speed-First':  dict(danger=-10, front_close=-1, wall_too_close=-5, ideal=3, wall_lost=-2,
                         fwd_risk={0: -6, 1: -1},
                         act={'Move-Forward': 4, 'Slight-Right-Turn': -2, 'Sharp-Right-Turn': -3, 'Slight-Left-Turn': -2}),
}


def expert_policy(df):
    s_arr, a_arr = df.state.values, df.action.values
    return np.array([np.bincount(a_arr[s_arr == s], minlength=N_ACTIONS).argmax()
                     if (s_arr == s).any() else 0 for s in range(N_STATES)])


def build_R(w, expert=None, imitation_bonus=0.0):
    R = np.zeros((N_STATES, N_ACTIONS))
    for s in range(N_STATES):
        f, l = cell(s)
        base = ({0: w['danger'], 1: w['front_close'], 2: 0, 3: 0, 4: 0}[f]
                + {0: w['wall_too_close'], 1: 2, 2: w['ideal'], 3: 1, 4: w['wall_lost']}[l])
        for a in range(N_ACTIONS):
            R[s, a] = base + w['act'][ACTIONS[a]]
            if ACTIONS[a] == 'Move-Forward':
                R[s, a] += w['fwd_risk'].get(f, 0)
            if imitation_bonus and expert is not None and a == expert[s]:
                R[s, a] += imitation_bonus
    return R


def all_rewards(expert):
    R = {k: build_R(w) for k, w in DESIGNS.items()}
    R['Expert-Imitation'] = build_R(DESIGNS['Balanced'], expert, imitation_bonus=3)
    return R


# ---------------------------------------------------------------- solvers
def value_iteration(P, R, gamma=0.9, tol=1e-6, max_iter=20000):
    V = np.zeros(P.shape[0])
    hist = []
    for it in range(1, max_iter + 1):
        Q = R + gamma * (P @ V)
        V_new = Q.max(axis=1)
        delta = float(np.max(np.abs(V_new - V)))
        hist.append(delta)
        V = V_new
        if delta < tol:
            break
    return V, Q.argmax(axis=1), Q, it, hist


def solve_all(df, gamma=0.9, step=5, tol=1e-6):
    P, counts = build_transitions(df, step)
    expert = expert_policy(df)
    Rs = all_rewards(expert)
    results = {}
    for name, R in Rs.items():
        V, pi, Q, it, hist = value_iteration(P, R, gamma, tol)
        results[name] = dict(V=V, pi=pi, Q=Q, iters=it, hist=hist)
    return P, counts, expert, Rs, results


# ---------------------------------------------------------------- evaluation
def simulate(P, pi, R, gamma, starts, T=150, seed=0):
    rng = np.random.default_rng(seed)
    cumP = P.cumsum(axis=2)
    out = dict(ret=[], danger=[], ideal=[], forward=[], lost=[])
    for s0 in starts:
        s, ret, dan, ide, fwd, lost = int(s0), 0.0, 0, 0, 0, 0
        for t in range(T):
            a = pi[s]
            ret += (gamma ** t) * R[s, a]
            fwd += a == 0
            s = min(int(np.searchsorted(cumP[s, a], rng.random())), N_STATES - 1)
            f, l = cell(s)
            dan += (f == 0) or (l == 0)
            ide += (l == 2) and f >= 2
            lost += l == 4
        for k, v in zip(out, [ret, dan / T, ide / T, fwd / T, lost / T]):
            out[k].append(v)
    return {k: float(np.mean(v)) for k, v in out.items()}


def performance_table(df, P, expert, Rs, results, gamma, n_ep=400):
    rng = np.random.default_rng(0)
    starts = rng.choice(df.state.values, n_ep)
    pols = {'Expert (majority)': expert, **{k: r['pi'] for k, r in results.items()}}
    rows = []
    for name, pi in pols.items():
        sim = simulate(P, pi, Rs['Balanced'], gamma, starts)
        rows.append({'Policy': name,
                     'Expert agreement %': round(100 * (pi[df.state.values] == df.action.values).mean(), 1),
                     'Return': round(sim['ret'], 1),
                     'Danger-zone time %': round(100 * sim['danger'], 1),
                     'Ideal-band time %': round(100 * sim['ideal'], 1),
                     'Wall-lost time %': round(100 * sim['lost'], 1),
                     'Forward moves %': round(100 * sim['forward'], 1)})
    return pd.DataFrame(rows).set_index('Policy')


# ---------------------------------------------------------------- pathways
def likely_path(P, pi, s0, max_steps=10):
    path, s = [int(s0)], int(s0)
    for _ in range(max_steps):
        probs = P[s, pi[s]].copy()
        probs[s] = 0
        nxt = int(probs.argmax())
        if nxt in path:
            return path, nxt
        path.append(nxt)
        s = nxt
    return path, None


def turn_calibration(df):
    a = df.action.values
    n_sharp = (a == A_IDX['Sharp-Right-Turn']).sum()
    n_sr = (a == A_IDX['Slight-Right-Turn']).sum()
    n_sl = (a == A_IDX['Slight-Left-Turn']).sum()
    sharp = 4 * 360 / (n_sharp + (n_sr - n_sl) / 3)   # expert completes 4 clockwise rounds
    return {0: 0.0, 1: -sharp / 3, 2: -sharp, 3: sharp / 3}


STEP = {0: 1.0, 1: 0.8, 2: 0.4, 3: 0.8}


def dead_reckon(act_seq, turn):
    n = len(act_seq)
    x, y, h = np.zeros(n + 1), np.zeros(n + 1), np.zeros(n + 1)
    h[0] = 90.0
    for i, a in enumerate(act_seq):
        h[i + 1] = h[i] + turn[a]
        x[i + 1] = x[i] + STEP[a] * np.cos(np.radians(h[i + 1]))
        y[i + 1] = y[i] + STEP[a] * np.sin(np.radians(h[i + 1]))
    return x, y, h
