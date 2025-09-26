import streamlit as st
from contextlib import contextmanager
import time

@contextmanager
def status(msg: str):
    with st.status(msg, expanded=False) as s:
        yield s

def toast_ok(msg: str):
    st.toast(msg, icon="✅")

def toast_warn(msg: str):
    st.toast(msg, icon="⚠️")
