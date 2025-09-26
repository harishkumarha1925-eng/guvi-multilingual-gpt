import streamlit as st

def status(msg, level="info"):
    # level: "info", "success", "warning", "error"
    fn = getattr(st, level, st.info)
    fn(msg)

def toast_ok(msg): st.toast(msg, icon="✅")
def toast_warn(msg): st.toast(msg, icon="⚠️")
