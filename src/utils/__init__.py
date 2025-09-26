from contextlib import contextmanager
import streamlit as st

@contextmanager
def status(label: str = "Working..."):
    # Minimal no-op context so imports don't fail
    yield

def toast_ok(msg: str) -> None:
    st.success(msg)

def toast_warn(msg: str) -> None:
    st.warning(msg)

