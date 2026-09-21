"""
Wall-Following Robot – MDP Navigation Dashboard
Run:  streamlit run app.py
"""
import os
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import mdp_engine as m

st.set_page_config(page_title='Robot MDP Navigator', page_icon='🤖', layout='wide')

INK, TEAL, AMBER, RED, GREEN = '#1B2B44', '#0F8B8D', '#E0A100', '#C2410C', '#1E8449'
POLICY_COLORS = {'Expert (data)': '#5B6770', 'Balanced': '#1F6FB2', 'Safety-First': '#2E8B57',
                 'Speed-First': '#D9822B', 'Expert-Imitation': '#7B4FA3'}
RAY_COLORS = {'front': RED, 'left': TEAL, 'right': AMBER, 'back': '#8A94A6'}

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
h1 {font-weight: 700; letter-spacing: -0.5px;}
div[data-testid="stMetricValue"] {font-size: 1.5rem;}
</style>""", unsafe_allow_html=True)


# ================================================================ data & model
@st.cache_data(show_spinner=False)
def load(file_bytes=None):
    if file_bytes is not None:
        import io
        return m.load_data(io.BytesIO(file_bytes))
    for p in ['sensor_readings_4.csv', '/content/sensor_readings_4.csv',
              os.path.join(os.path.dirname(__file__), 'sensor_readings_4.csv')]:
        if os.path.exists(p):
            return m.load_data(p)
    return None


@st.cache_data(show_spinner='Solving the MDP with Value Iteration…')
def solve(_df, key, gamma, step):
    P, counts, expert, Rs, results = m.solve_all(_df, gamma, step)
    perf = m.performance_table(_df, P, expert, Rs, results, gamma)
    return P, counts, expert, Rs, results, perf


with st.sidebar:
    st.header('Controls')
    up = st.file_uploader('Sensor data (sensor_readings_4.csv)', type=['csv', 'data'])
    df = load(up.getvalue() if up else None)
    if df is None:
        st.error('No data found. Upload sensor_readings_4.csv or place it next to app.py.')
        st.stop()
    gamma = st.slider('Discount factor γ', 0.50, 0.99, 0.90, 0.01)
    step = st.slider('Decision step (samples)', 1, 15, 5,
                     help='Rows between decisions. Data is 9 Hz, so 5 ≈ 0.55 s.')
    st.divider()
    policy = st.selectbox('Policy to drive the robot', ['Expert (data)'] + m.POLICY_NAMES, index=1)
    stride = st.slider('Animation stride (samples per frame)', 4, 40, 12,
                       help='Higher = fewer frames, faster to load.')
    speed = st.slider('Frame duration (ms)', 10, 200, 40)
    show_rays = st.checkbox('Show sensor rays', True)

P, counts, expert, Rs, results, perf = solve(df, (len(df), up.name if up else 'default'), gamma, step)
TURN = m.turn_calibration(df)
s_arr, a_arr = df.state.values, df.action.values


def action_sequence(name):
    return a_arr if name == 'Expert (data)' else results[name]['pi'][s_arr]


@st.cache_data(show_spinner=False)
def trajectory(name, key):
    x, y, h = m.dead_reckon(action_sequence(name), TURN)
    return x, y, h


# ================================================================ header
st.title('Wall-following robot · MDP navigator')
st.caption('SCITOS-G5 ultrasound data → 5×5 grid MDP → Value Iteration → robot driven by the chosen policy. '
           'Press ▶ Play under a chart to set the robot moving.')

c1, c2, c3, c4, c5 = st.columns(5)
row = perf.loc['Expert (majority)' if policy == 'Expert (data)' else policy]
c1.metric('Policy', policy)
c2.metric('Return', f"{row['Return']:.1f}")
c3.metric('Danger-zone time', f"{row['Danger-zone time %']:.1f}%")
c4.metric('Ideal wall band', f"{row['Ideal-band time %']:.1f}%")
c5.metric('Forward moves', f"{row['Forward moves %']:.1f}%")

tab_run, tab_race, tab_grid, tab_live, tab_perf = st.tabs(
    ['Robot run', 'Policy race', 'Grid & pathways', 'Live telemetry', 'Performance & files'])


# ================================================================ helpers
def grid_heatmap(V, pi, faint=False):
    G = V.reshape(m.NF, m.NL)
    text = [[m.A_SHORT[m.ACTIONS[pi[f * m.NL + l]]] for l in range(m.NL)] for f in range(m.NF)]
    return go.Heatmap(z=G, x=list(range(m.NL)), y=list(range(m.NF)), colorscale='RdYlGn',
                      text=text, texttemplate='%{text}',
                      textfont=dict(size=14 if faint else 18, color='rgba(27,43,68,0.35)' if faint else INK),
                      showscale=False, hovertemplate='F%{y}/L%{x}<br>V=%{z:.1f}<extra></extra>')


def add_grid_shapes(fig, V, pi, xref='x2', yref='y2'):
    """Draw the MDP grid as layout shapes (stays visible during animation)."""
    from plotly.colors import sample_colorscale
    lo, hi = float(V.min()), float(V.max())
    cols = sample_colorscale('RdYlGn', [(v - lo) / (hi - lo + 1e-9) for v in V])
    for s in range(m.N_STATES):
        f, l = m.cell(s)
        fig.add_shape(type='rect', x0=l - .5, x1=l + .5, y0=f - .5, y1=f + .5, xref=xref, yref=yref,
                      fillcolor=cols[s], line=dict(color='white', width=2), layer='below')
        fig.add_annotation(x=l, y=f, xref=xref, yref=yref, showarrow=False,
                           text=f'<b>{m.A_SHORT[m.ACTIONS[pi[s]]]}</b><br><span style="font-size:10px">V={V[s]:.0f}</span>',
                           font=dict(size=16, color=INK))


def style_grid_axes(fig, row=None, col=None):
    kw = dict(row=row, col=col) if row else {}
    fig.update_xaxes(tickvals=list(range(m.NL)), ticktext=[n.split()[0] + ' ' + n.split()[1] for n in m.LEFT_NAMES],
                     title_text='Left-wall zone', **kw)
    fig.update_yaxes(tickvals=list(range(m.NF)), ticktext=[n.split()[0] + ' ' + n.split()[1] for n in m.FRONT_NAMES],
                     autorange='reversed', title_text='Front zone', **kw)


def rays(x, y, h, i):
    segs_x, segs_y = [], []
    dirs = {'front': 0, 'left': 90, 'right': -90, 'back': 180}
    cols = {'front': 'SD_front', 'left': 'SD_left', 'right': 'SD_right', 'back': 'SD_back'}
    scale = 30
    for k, d in dirs.items():
        L = df[cols[k]].values[min(i, len(df) - 1)] * scale
        ang = np.radians(h[i] + d)
        segs_x += [x[i], x[i] + L * np.cos(ang), None]
        segs_y += [y[i], y[i] + L * np.sin(ang), None]
    return segs_x, segs_y


def play_buttons(duration):
    return [dict(type='buttons', showactive=False, x=0.0, y=-0.02, xanchor='left', yanchor='top', direction='left',
                 pad=dict(t=0, r=10),
                 buttons=[dict(label='▶ Play', method='animate',
                               args=[None, dict(frame=dict(duration=duration, redraw=False),
                                                transition=dict(duration=0), fromcurrent=True, mode='immediate')]),
                          dict(label='⏸ Pause', method='animate',
                               args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])])]


def frame_slider(n):
    return [dict(active=0, y=-0.02, x=0.14, len=0.86, yanchor='top', pad=dict(t=0),
                 currentvalue=dict(prefix='Frame ', font=dict(size=12)), ticklen=0, font=dict(size=1, color='rgba(0,0,0,0)'),
                 steps=[dict(method='animate', label=str(k),
                             args=[[str(k)], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
                        for k in range(n)])]


# ================================================================ tab 1: robot run
with tab_run:
    x, y, h = trajectory(policy, (gamma, step))
    idx = np.arange(0, len(x) - 1, stride)
    idx = np.append(idx, len(x) - 1)
    V_show = results[policy]['V'] if policy != 'Expert (data)' else results['Expert-Imitation']['V']
    pi_show = results[policy]['pi'] if policy != 'Expert (data)' else expert
    color = POLICY_COLORS[policy]

    fig = make_subplots(rows=1, cols=2, column_widths=[0.64, 0.36], horizontal_spacing=0.08,
                        subplot_titles=('Robot in the room (dead-reckoned path)', 'Current state on the MDP grid'))
    # 0 full path (faint), 1 trail, 2 robot, 3 rays, 4 start, 5 end, 6 heatmap, 7 grid marker
    fig.add_trace(go.Scatter(x=x[idx], y=y[idx], mode='lines', line=dict(color=color, width=1), opacity=.18,
                             name='Full route', hoverinfo='skip'), 1, 1)
    fig.add_trace(go.Scatter(x=[x[0]], y=[y[0]], mode='lines', line=dict(color=color, width=3),
                             name='Travelled'), 1, 1)
    fig.add_trace(go.Scatter(x=[x[0]], y=[y[0]], mode='markers', name='Robot',
                             marker=dict(symbol='triangle-up', size=22, color=color, angle=90 - h[0],
                                         line=dict(color=INK, width=2))), 1, 1)
    rx, ry = rays(x, y, h, 0)
    fig.add_trace(go.Scatter(x=rx if show_rays else [None], y=ry if show_rays else [None], mode='lines',
                             line=dict(color=TEAL, width=1.5, dash='dot'), name='Sensor rays (scaled)'), 1, 1)
    fig.add_trace(go.Scatter(x=[x[0]], y=[y[0]], mode='markers+text', text=['START'], textposition='top center',
                             marker=dict(size=14, color=GREEN, line=dict(color=INK, width=1)),
                             textfont=dict(color=GREEN, size=12), name='Start'), 1, 1)
    fig.add_trace(go.Scatter(x=[x[-1]], y=[y[-1]], mode='markers+text', text=['END'], textposition='bottom center',
                             marker=dict(size=14, color=RED, symbol='square', line=dict(color=INK, width=1)),
                             textfont=dict(color=RED, size=12), name='End'), 1, 1)
    f0, l0 = m.cell(s_arr[0])
    xr0, yr0 = x[idx].min(), y[idx].max()
    fig.add_trace(go.Scatter(x=[xr0], y=[yr0 + 25], mode='text', text=['Step 0 · press ▶ Play'],
                             textposition='middle right', textfont=dict(size=13, color=INK),
                             showlegend=False, hoverinfo='skip'), 1, 1)
    fig.add_trace(go.Scatter(x=[l0], y=[f0], mode='markers', name='Current state',
                             marker=dict(symbol='square-open', size=46, color=INK, line=dict(width=4))), 1, 2)
    add_grid_shapes(fig, V_show, pi_show)

    seq = action_sequence(policy)
    frames = []
    for k, i in enumerate(idx):
        i_data = min(i, len(df) - 1)
        rx, ry = rays(x, y, h, i) if show_rays else ([None], [None])
        f_, l_ = m.cell(s_arr[i_data])
        info = (f'Step {i}/{len(x) - 1} · F{f_}/L{l_} · {m.ACTIONS[seq[i_data]]} · '
                f'front {df.SD_front.values[i_data]:.2f} m · left {df.SD_left.values[i_data]:.2f} m')
        frames.append(go.Frame(
            name=str(k),
            data=[go.Scatter(x=x[idx[:k + 1]], y=y[idx[:k + 1]]),
                  go.Scatter(x=[x[i]], y=[y[i]], marker=dict(angle=90 - h[i])),
                  go.Scatter(x=rx, y=ry),
                  go.Scatter(text=[info]),
                  go.Scatter(x=[l_], y=[f_])],
            traces=[1, 2, 3, 6, 7]))
    fig.frames = frames
    fig.update_layout(
        height=680, plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=90, b=90, l=10, r=10),
        legend=dict(orientation='h', y=1.14, x=0),
        updatemenus=play_buttons(speed), sliders=frame_slider(len(idx)))
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='x', scaleratio=1, row=1, col=1)
    style_grid_axes(fig, 1, 2)
    fig.update_xaxes(range=[-0.5, m.NL - 0.5], showgrid=False, zeroline=False, title_standoff=4, row=1, col=2)
    fig.update_yaxes(range=[m.NF - 0.5, -0.5], autorange=False, showgrid=False, zeroline=False, row=1, col=2)
    st.plotly_chart(fig, width='stretch')
    heading = h[-1] - 90
    st.caption(f'{len(idx)} frames · net heading change {heading:.0f}° ({abs(heading) / 360:.1f} rounds) · '
               f'START→END gap {np.hypot(x[-1] - x[0], y[-1] - y[0]):.0f} units. '
               'Turn rates are calibrated so the expert completes the 4 clockwise rounds in the README. '
               'Sensor rays show the recorded distances (red = front, scaled).')


# ================================================================ tab 2: race
with tab_race:
    st.subheader('All policies driving at once')
    chosen = st.multiselect('Robots in the race', list(POLICY_COLORS), default=list(POLICY_COLORS))
    if chosen:
        race_stride = max(stride, 15)
        trajs = {n: trajectory(n, (gamma, step)) for n in chosen}
        n_pts = len(next(iter(trajs.values()))[0])
        ridx = np.append(np.arange(0, n_pts - 1, race_stride), n_pts - 1)
        rf = go.Figure()
        for n in chosen:                         # faint full routes
            x, y, _ = trajs[n]
            rf.add_trace(go.Scatter(x=x[ridx], y=y[ridx], mode='lines', opacity=.15, showlegend=False,
                                    line=dict(color=POLICY_COLORS[n], width=1), hoverinfo='skip'))
        for n in chosen:                         # trails
            rf.add_trace(go.Scatter(x=[0], y=[0], mode='lines', name=n, line=dict(color=POLICY_COLORS[n], width=2.5)))
        for n in chosen:                         # robots
            x, y, h = trajs[n]
            rf.add_trace(go.Scatter(x=[x[0]], y=[y[0]], mode='markers', showlegend=False,
                                    marker=dict(symbol='triangle-up', size=18, color=POLICY_COLORS[n],
                                                angle=90 - h[0], line=dict(color=INK, width=1.5))))
        rf.add_trace(go.Scatter(x=[0], y=[0], mode='markers+text', text=['START'], textposition='top center',
                                marker=dict(size=14, color=GREEN), textfont=dict(color=GREEN), name='Start'))
        k_n = len(chosen)
        frames = []
        for k, i in enumerate(ridx):
            data, tr = [], []
            for j, n in enumerate(chosen):
                x, y, _ = trajs[n]
                data.append(go.Scatter(x=x[ridx[:k + 1]], y=y[ridx[:k + 1]])); tr.append(k_n + j)
            for j, n in enumerate(chosen):
                x, y, h = trajs[n]
                data.append(go.Scatter(x=[x[i]], y=[y[i]], marker=dict(angle=90 - h[i]))); tr.append(2 * k_n + j)
            frames.append(go.Frame(name=str(k), data=data, traces=tr))
        for j, n in enumerate(chosen):           # END labels
            x, y, _ = trajs[n]
            rf.add_trace(go.Scatter(x=[x[-1]], y=[y[-1]], mode='markers+text', text=[f'END {n}'],
                                    textposition='bottom center', showlegend=False,
                                    textfont=dict(size=10, color=POLICY_COLORS[n]),
                                    marker=dict(symbol='square', size=11, color=RED,
                                                line=dict(color=POLICY_COLORS[n], width=2))))
        rf.frames = frames
        rf.update_layout(height=700, plot_bgcolor='white', updatemenus=play_buttons(speed),
                         margin=dict(t=40, b=90), legend=dict(orientation='h', y=1.06),
                         sliders=frame_slider(len(ridx)))
        rf.update_xaxes(visible=False); rf.update_yaxes(visible=False, scaleanchor='x', scaleratio=1)
        st.plotly_chart(rf, width='stretch')
        st.caption('Every robot reads the same recorded sensor stream but chooses its own actions, '
                   'so the routes separate. The expert closes 4 rounds; policies that over- or under-turn drift away.')


# ================================================================ tab 3: grid & pathways
with tab_grid:
    st.subheader('Pathway through the MDP grid')
    g1, g2, g3 = st.columns(3)
    gp = g1.selectbox('Policy', m.POLICY_NAMES, index=m.POLICY_NAMES.index(policy) if policy in m.POLICY_NAMES else 0)
    sf = g2.selectbox('Start: front zone', m.FRONT_NAMES, index=1)
    sl = g3.selectbox('Start: left-wall zone', m.LEFT_NAMES, index=0)
    start = m.FRONT_NAMES.index(sf) * m.NL + m.LEFT_NAMES.index(sl)
    r = results[gp]
    path, loop_to = m.likely_path(P, r['pi'], start)

    gf = go.Figure(grid_heatmap(r['V'], r['pi'], faint=True))
    rows = []
    for k, (s1, s2) in enumerate(zip(path[:-1], path[1:]), start=1):
        (f1, l1), (f2, l2) = m.cell(s1), m.cell(s2)
        a = r['pi'][s1]
        gf.add_annotation(x=l2, y=f2, ax=l1, ay=f1, xref='x', yref='y', axref='x', ayref='y',
                          showarrow=True, arrowhead=3, arrowsize=1.4, arrowwidth=3, arrowcolor=INK,
                          standoff=16, startstandoff=16)
        gf.add_annotation(x=(l1 + l2) / 2, y=(f1 + f2) / 2, text=f'<b>{k}:{m.A_SHORT[m.ACTIONS[a]]}</b>',
                          showarrow=False, font=dict(color='white', size=13), bgcolor=INK, borderpad=3)
        rows.append({'Step': str(k), 'From': m.state_label(s1), 'Action': m.ACTIONS[a], 'To': m.state_label(s2),
                     'Probability': round(P[s1, a, s2], 3)})
    if loop_to is not None:
        (f1, l1), (f2, l2) = m.cell(path[-1]), m.cell(loop_to)
        gf.add_annotation(x=l2 + .12, y=f2 + .12, ax=l1 + .12, ay=f1 + .12, xref='x', yref='y', axref='x', ayref='y',
                          showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor='#8A94A6', standoff=16, startstandoff=16)
        rows.append({'Step': 'loop', 'From': m.state_label(path[-1]), 'Action': m.ACTIONS[r['pi'][path[-1]]],
                     'To': m.state_label(loop_to), 'Probability': round(P[path[-1], r['pi'][path[-1]], loop_to], 3)})
    (a0, b0), (a1, b1) = m.cell(path[0]), m.cell(path[-1])
    gf.add_trace(go.Scatter(x=[b0], y=[a0], mode='markers+text', text=['START'], textposition='top center',
                            marker=dict(symbol='circle-open', size=48, color=GREEN, line=dict(width=4)),
                            textfont=dict(color=GREEN, size=13), showlegend=False))
    gf.add_trace(go.Scatter(x=[b1], y=[a1], mode='markers+text', text=['END'], textposition='bottom center',
                            marker=dict(symbol='square-open', size=48, color=RED, line=dict(width=4)),
                            textfont=dict(color=RED, size=13), showlegend=False))
    gf.update_layout(height=560, margin=dict(t=20, b=10), plot_bgcolor='white')
    style_grid_axes(gf)
    cA, cB = st.columns([0.6, 0.4])
    cA.plotly_chart(gf, width='stretch')
    cB.markdown('**Liaisons (step by step)**')
    cB.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    cB.caption('Letters: F forward · r slight right · R sharp right · l slight left. '
               'END is the last new state before the robot settles into a loop (grey arrow).')


# ================================================================ tab 4: live telemetry
with tab_live:
    st.subheader('Live run – robot reading its sensors in real time')
    l1, l2, l3 = st.columns(3)
    start_row = l1.number_input('Start at time step', 0, len(df) - 2, 0, 50)
    n_steps = l2.slider('Steps to play', 20, 600, 200, 20)
    delay = l3.slider('Delay per step (s)', 0.0, 0.3, 0.05, 0.01)
    go_live = st.button('Start live run', type='primary')

    metric_box = st.empty()
    chart_box = st.empty()
    if go_live:
        x, y, h = trajectory(policy, (gamma, step))
        seq = action_sequence(policy)
        live_stride = max(1, n_steps // 120)
        for i in range(int(start_row), int(start_row) + n_steps, live_stride):
            i = min(i, len(df) - 1)
            f_, l_ = m.cell(s_arr[i])
            with metric_box.container():
                k1, k2, k3, k4, k5, k6 = st.columns(6)
                k1.metric('Step', i)
                k2.metric('Front', f'{df.SD_front.values[i]:.2f} m')
                k3.metric('Left', f'{df.SD_left.values[i]:.2f} m')
                k4.metric('Right', f'{df.SD_right.values[i]:.2f} m')
                k5.metric('State', f'F{f_}/L{l_}')
                k6.metric('Action', m.ACTIONS[seq[i]].replace('-Turn', ''),
                          'matches expert' if seq[i] == a_arr[i] else f'expert: {m.A_SHORT[m.ACTIONS[a_arr[i]]]}',
                          delta_color='off')
            lo = max(0, i - 400)
            lf = go.Figure()
            lf.add_trace(go.Scatter(x=x[lo:i + 1], y=y[lo:i + 1], mode='lines',
                                    line=dict(color=POLICY_COLORS[policy], width=3), name='Recent path'))
            rx, ry = rays(x, y, h, i)
            lf.add_trace(go.Scatter(x=rx, y=ry, mode='lines', line=dict(color=TEAL, dash='dot'), name='Sensor rays'))
            lf.add_trace(go.Scatter(x=[x[i]], y=[y[i]], mode='markers', name='Robot',
                                    marker=dict(symbol='triangle-up', size=26, color=POLICY_COLORS[policy],
                                                angle=90 - h[i], line=dict(color=INK, width=2))))
            lf.add_trace(go.Scatter(x=[x[int(start_row)]], y=[y[int(start_row)]], mode='markers+text',
                                    text=['START'], textposition='top center', marker=dict(color=GREEN, size=12),
                                    textfont=dict(color=GREEN), name='Start'))
            lf.update_layout(height=520, plot_bgcolor='white', margin=dict(t=10, b=10),
                             xaxis=dict(visible=False, range=[x[i] - 70, x[i] + 70]),
                             yaxis=dict(visible=False, range=[y[i] - 55, y[i] + 55], scaleanchor='x'))
            chart_box.plotly_chart(lf, width='stretch', key=f'live_{i}')
            time.sleep(delay)
        st.success(f'Run finished at step {i}.')
    else:
        chart_box.info('Choose a starting time step and press Start live run. The camera follows the robot.')


# ================================================================ tab 5: performance
with tab_perf:
    st.subheader('How the policies compare')
    st.dataframe(perf, width='stretch')
    pc = st.columns(3)
    for col, metric in zip(pc, ['Return', 'Danger-zone time %', 'Ideal-band time %']):
        bf = go.Figure(go.Bar(x=perf[metric], y=perf.index, orientation='h',
                              marker_color=['#5B6770', '#1F6FB2', '#2E8B57', '#D9822B', '#7B4FA3'],
                              text=perf[metric], textposition='outside'))
        bf.update_layout(title=metric, height=300, margin=dict(l=10, r=30, t=40, b=10),
                         yaxis=dict(autorange='reversed'), plot_bgcolor='white')
        col.plotly_chart(bf, width='stretch')

    cf = go.Figure()
    for n, r in results.items():
        cf.add_trace(go.Scatter(y=r['hist'], mode='lines', name=f"{n} ({r['iters']} it.)",
                                line=dict(color=POLICY_COLORS[n])))
    cf.add_hline(y=1e-6, line_dash='dash', line_color='grey', annotation_text='tolerance 1e-6')
    cf.update_layout(title=f'Value Iteration convergence (γ = {gamma})', yaxis_type='log', height=340,
                     xaxis_title='Iteration', yaxis_title='max |ΔV|', plot_bgcolor='white')
    st.plotly_chart(cf, width='stretch')

    out = pd.DataFrame({'state_id': range(m.N_STATES),
                        'front_zone': [m.FRONT_NAMES[m.cell(s)[0]] for s in range(m.N_STATES)],
                        'left_zone': [m.LEFT_NAMES[m.cell(s)[1]] for s in range(m.N_STATES)],
                        'observations': np.bincount(s_arr, minlength=m.N_STATES)})
    for n, r in results.items():
        out[f'V_{n}'] = r['V'].round(4)
        out[f'policy_{n}'] = [m.ACTIONS[a] for a in r['pi']]
    d1, d2 = st.columns(2)
    d1.download_button('Download optimal_value_function.csv', out.to_csv(index=False),
                       'optimal_value_function.csv', 'text/csv')
    d2.download_button('Download policy_performance.csv', perf.to_csv(), 'policy_performance.csv', 'text/csv')
