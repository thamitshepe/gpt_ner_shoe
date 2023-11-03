from langcorn import create_service

app = create_service(
    "main:chain"
)