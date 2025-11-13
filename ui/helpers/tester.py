from query_logger import QueryLogger
from ui.helpers.client import get_client_info

logger = QueryLogger()

# Get client info
ip_address, user_agent = "192.168.1.10", "me"
error_msg = None


# resposne  = logger.log_query(
#     query_text="WHAT THE HELLLLLLLLLLLLLLLLLLLLLLLLLL",
#     ip_address=ip_address,
#     user_agent=user_agent,
#     error_message=error_msg,
#     response="hello there"
# )
#
resposne  = logger.log_query(
    id =12,
    error_message="TOO MANY ERRORS",
)

#  id = resposne.data[0]['id']