from contextlib import contextmanager
import streamlit as st

@contextmanager
def status(label="Working…"):
    with st.status(label) as s:
        yield s
        s.update(label="Done", state="complete")

def toast_ok(msg: str): st.toast(msg, icon="?")
def toast_warn(msg: str): st.toast(msg, icon="??")
