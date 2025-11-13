from streamlit.web.server.websocket_headers import _get_websocket_headers
import streamlit as st

def get_client_info():
    """Extract IP address and user agent from Streamlit headers"""
    try:
        headers = st.context.headers
        ip_address = headers.get("X-Forwarded-For", headers.get("Remote-Addr"))
        user_agent = headers.get("User-Agent")
        return ip_address, user_agent
    except:
        return None, None


# # Initialize logger (add this near the top of your file, after imports)
# @st.cache_resource
# def get_logger():
#     """Initialize query logger once"""
#     return QueryLogger()  # Reads from DATABASE_URL env variable
#
# logger = get_logger()