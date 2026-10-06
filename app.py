from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_VERSION = "v0.4.0-alpha"
st.set_page_config(page_title=f"GeoSeis Viewer {APP_VERSION}", page_icon="◫", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
#MainMenu, footer {visibility:hidden}
.block-container {padding-top:1.35rem; padding-bottom:1rem; max-width:100%}
[data-testid="stSidebar"] {background:#111827}
[data-testid="stSidebar"] * {color:#f9fafb}
.header {display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #e5e7eb;padding:0 0 14px 0;margin-bottom:14px}
.brand {font-size:26px;font-weight:750}.sub {font-size:13px;color:#6b7280;margin-top:3px}
.ver {padding:7px 12px;border:1px solid #d1d5db;border-radius:10px;background:#fff;font-weight:650}
.empty {height:58vh;border:2px dashed #d1d5db;border-radius:16px;display:grid;place-items:center;text-align:center;color:#6b7280;background:#fafafa}
[data-testid="stMetric"] {border:1px solid #e5e7eb;border-radius:12px;padding:12px;background:#fff}
.stTabs [data-baseweb="tab-list"] {gap:8px}.stTabs [data-baseweb="tab"] {border-radius:9px;padding:8px 15px}
</style>
""", unsafe_allow_html=True)

st.markdown(f'''<div class="header"><div><div class="brand">GeoSeis Viewer</div><div class="sub">SEG-Y · HDF5 · QC · Visualización</div></div><div class="ver">{APP_VERSION}</div></div>''', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_segy(raw: bytes, filename: str):
    import segyio
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / filename
        path.write_bytes(raw)
        with segyio.open(str(path), "r", strict=False, ignore_geometry=True) as f:
            ntr = f.tracecount
            samples = np.asarray(f.samples, dtype=np.float32)
            data = np.empty((ntr, len(samples)), dtype=np.float32)
            iline = np.zeros(ntr, dtype=np.int32)
            xline = np.zeros(ntr, dtype=np.int32)
            cdp_x = np.zeros(ntr, dtype=np.int64)
            cdp_y = np.zeros(ntr, dtype=np.int64)
            for i in range(ntr):
                data[i] = f.trace[i]
                h = f.header[i]
                iline[i] = h.get(segyio.TraceField.INLINE_3D, 0)
                xline[i] = h.get(segyio.TraceField.CROSSLINE_3D, 0)
                cdp_x[i] = h.get(segyio.TraceField.CDP_X, 0)
                cdp_y[i] = h.get(segyio.TraceField.CDP_Y, 0)
            dt = int(segyio.tools.dt(f))
    headers = pd.DataFrame({"Trace":np.arange(1,ntr+1),"Inline":iline,"Crossline":xline,"CDP_X":cdp_x,"CDP_Y":cdp_y})
    return data, samples, dt, headers

@st.cache_data(show_spinner=False)
def load_hdf5(raw: bytes, filename: str):
    import h5py
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / filename
        path.write_bytes(raw)
        with h5py.File(path, "r") as f:
            candidates=[]
            def visitor(name,obj):
                if isinstance(obj,h5py.Dataset) and obj.ndim in (2,3) and np.issubdtype(obj.dtype,np.number): candidates.append(name)
            f.visititems(visitor)
            preferred = "seismic/data" if "seismic/data" in candidates else (candidates[0] if candidates else None)
            if preferred is None: raise ValueError("No se encontró un dataset sísmico 2D o 3D.")
            arr = np.asarray(f[preferred], dtype=np.float32)
            if arr.ndim == 3: arr = arr.reshape(-1, arr.shape[-1])
            data = arr
            samples = np.arange(data.shape[1], dtype=np.float32)
            dt = int(f["seismic"].attrs.get("sample_interval_us",0)) if "seismic" in f else 0
            headers={}
            if "headers" in f:
                for key in f["headers"].keys():
                    v=np.asarray(f[f"headers/{key}"])
                    if v.ndim==1 and len(v)==data.shape[0]: headers[key]=v
            hdr=pd.DataFrame(headers) if headers else pd.DataFrame({"Trace":np.arange(1,data.shape[0]+1)})
    return data,samples,dt,hdr

def seismic_figure(data, samples, clip, gain, max_traces, palette):
    n=min(max_traces,data.shape[0]); idx=np.linspace(0,data.shape[0]-1,n).astype(int)
    view=data[idx].T * gain
    finite=view[np.isfinite(view)]
    vmax=float(np.percentile(np.abs(finite),clip)) if finite.size else 1.0
    vmax=vmax or 1.0
    colorscale={"Gris":"gray","Rojo/Azul":"RdBu_r","Viridis":"Viridis"}[palette]
    fig=go.Figure(go.Heatmap(z=view,x=idx+1,y=samples,colorscale=colorscale,zmin=-vmax,zmax=vmax,colorbar_title="Amp."))
    fig.update_layout(height=690,margin=dict(l=55,r=20,t=15,b=45),xaxis_title="Traza",yaxis_title="Tiempo / muestra",plot_bgcolor="white")
    fig.update_yaxes(autorange="reversed")
    return fig

def qc(data):
    safe=np.nan_to_num(data)
    rms=np.sqrt(np.mean(np.square(safe),axis=1))
    peak=np.max(np.abs(safe),axis=1)
    return rms,peak,int(np.all(safe==0,axis=1).sum()),int((np.std(safe,axis=1)==0).sum()),int((~np.isfinite(data)).sum())

with st.sidebar:
    st.markdown("## Abrir archivo")
    uploaded=st.file_uploader("Seleccione SEG-Y o HDF5",type=["sgy","segy","h5","hdf5"],label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### Visualización")
    clip=st.slider("Clip (%)",90.0,100.0,99.0,0.5)
    gain=st.slider("Ganancia",0.1,10.0,1.0,0.1)
    palette=st.selectbox("Paleta",["Gris","Rojo/Azul","Viridis"])
    max_traces=st.slider("Trazas en pantalla",100,3000,1000,100)

if uploaded is None:
    st.markdown('''<div class="empty"><div><div style="font-size:54px">▥</div><h2>Abra un archivo sísmico</h2><p>Use el panel izquierdo para seleccionar un archivo .sgy, .segy, .h5 o .hdf5.</p></div></div>''',unsafe_allow_html=True)
    st.stop()

raw=uploaded.getvalue(); suffix=Path(uploaded.name).suffix.lower()
try:
    if suffix in (".sgy",".segy"): data,samples,dt,headers=load_segy(raw,uploaded.name)
    else: data,samples,dt,headers=load_hdf5(raw,uploaded.name)
except Exception as e:
    st.error(f"No fue posible abrir el archivo: {e}")
    st.stop()

rms,peak,dead,constant,bad=qc(data)
name_col,size_col,trace_col,sample_col,dt_col=st.columns([2.3,1,1,1,1])
name_col.metric("Archivo",uploaded.name)
size_col.metric("Tamaño",f"{len(raw)/1024/1024:.1f} MB")
trace_col.metric("Trazas",f"{data.shape[0]:,}")
sample_col.metric("Muestras",f"{data.shape[1]:,}")
dt_col.metric("Intervalo",f"{dt} μs" if dt else "N/D")

view_tab,qc_tab,header_tab,export_tab=st.tabs(["Visualización","QC","Headers","Exportar HDF5"])
with view_tab:
    st.plotly_chart(seismic_figure(data,samples,clip,gain,max_traces,palette),use_container_width=True,config={"displaylogo":False,"scrollZoom":True})
with qc_tab:
    a,b,c,d=st.columns(4);a.metric("Trazas muertas",dead);b.metric("Trazas constantes",constant);c.metric("NaN / Inf",bad);d.metric("RMS medio",f"{np.mean(rms):.4g}")
    qcdf=pd.DataFrame({"Traza":np.arange(1,len(rms)+1),"RMS":rms,"Pico":peak})
    st.line_chart(qcdf.set_index("Traza"),height=310)
    hist=np.histogram(data[np.isfinite(data)][::max(1,data.size//200000)],bins=100)
    hdf=pd.DataFrame({"Amplitud":hist[1][:-1],"Frecuencia":hist[0]}).set_index("Amplitud")
    st.bar_chart(hdf,height=260)
with header_tab:
    st.dataframe(headers,use_container_width=True,height=610)
with export_tab:
    import h5py, os
    outpath=Path(tempfile.gettempdir())/(Path(uploaded.name).stem+".h5")
    with h5py.File(outpath,"w") as f:
        g=f.create_group("seismic");g.create_dataset("data",data=data,chunks=True,compression="gzip",shuffle=True);g.create_dataset("samples",data=samples);g.attrs["sample_interval_us"]=dt
        hg=f.create_group("headers")
        for col in headers.columns:
            if col!="Trace": hg.create_dataset(str(col).lower(),data=headers[col].to_numpy(),compression="gzip")
    st.download_button("Descargar HDF5",outpath.read_bytes(),file_name=outpath.name,mime="application/x-hdf5",use_container_width=True)
