import os
from openenv.core import create_fastapi_app

from models import EmailAction, EmailObservation
from server.environment import EmailNegotiationEnvironment


app = create_fastapi_app(EmailNegotiationEnvironment, EmailAction, EmailObservation)


def main():
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()