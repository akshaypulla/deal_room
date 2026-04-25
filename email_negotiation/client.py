from openenv.core.env_client import EnvClient as HTTPEnvClient

from models import EmailAction, EmailObservation, EmailState


class EmailNegotiationEnv(HTTPEnvClient):
    action_class = EmailAction
    observation_class = EmailObservation
    state_class = EmailState