import streamlit as st

def load_css():

    st.markdown("""
    <style>

    .main{
        background-color:#F8FAFC;
    }

    .title{
        font-size:38px;
        font-weight:700;
        color:#1E3A8A;
        text-align:center;
        margin-bottom:25px;
    }

    .subtitle{
        font-size:18px;
        color:#64748B;
        text-align:center;
        margin-bottom:30px;
    }

    .card{
        background:white;
        padding:20px;
        border-radius:18px;
        box-shadow:0px 4px 20px rgba(0,0,0,0.08);
        text-align:center;
        transition:0.3s;
    }

    .card:hover{
        transform:translateY(-5px);
        box-shadow:0px 10px 25px rgba(0,0,0,0.15);
    }

    .metric-title{
        color:#64748B;
        font-size:16px;
        font-weight:600;
    }

    .metric-value{
        color:#2563EB;
        font-size:32px;
        font-weight:bold;
    }

    .section{
        font-size:24px;
        font-weight:bold;
        color:#1E293B;
        margin-top:30px;
        margin-bottom:15px;
    }

    div[data-testid="stSidebar"]{
        background:#0F172A;
    }

    div[data-testid="stSidebar"] *{
        color:white;
    }

    </style>
    """, unsafe_allow_html=True)