import os
from openenv.core.env_server.http_server import create_app

from models import EmailAction, EmailObservation
from server.environment import EmailNegotiationEnvironment


app = create_app(EmailNegotiationEnvironment, EmailAction, EmailObservation)


def main():
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()