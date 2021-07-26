import numpy as np
from messagingclient import MessagingClient


def callback(payload):
    data = np.frombuffer(payload.data, dtype=np.uint64)
    print(data)



c = MessagingClient()
c.consume("test", callback)
