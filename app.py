from langcorn import create_service

app = create_service(
    "api.main:chain"
)