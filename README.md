# Wall-Following Robot – MDP Navigator (Streamlit)

An animated dashboard: the robot drives around the room using policies found by Value Iteration.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard (5 tabs) |
| `mdp_engine.py` | Grid MDP, transitions, rewards, Value Iteration, pathways |
| `sensor_readings_4.csv` | Dataset (SD_front, SD_left, SD_right, SD_back, class) |
| `requirements.txt` | Python packages |
| `.streamlit/config.toml` | Theme |

## Run on your laptop
    pip install -r requirements.txt
    streamlit run app.py

Opens at http://localhost:8501

## Run in Google Colab
Upload all files to /content (keep config.toml inside a .streamlit folder), then run:

    !pip install -q streamlit plotly
    !wget -q -O cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x cloudflared
    !nohup streamlit run app.py --server.port 8501 > log.txt 2>&1 &
    !./cloudflared tunnel --url http://localhost:8501

Open the https://....trycloudflare.com link printed in the output.

## Deploy online (free)
Push the folder to GitHub, go to share.streamlit.io, click New app and select app.py.

## Tabs
1. Robot run – press Play: the robot drives its route with sensor rays; its current grid state is highlighted on the MDP grid.
2. Policy race – all policies drive at the same time from the same START.
3. Grid & pathways – pick a start cell; numbered liaisons show the route to END, with a probability table.
4. Live telemetry – step-by-step live run with sensor readings, state and action (camera follows the robot).
5. Performance & files – comparison table, charts, convergence, CSV downloads.
