import paramiko
import socket
import threading
import select
from rich.console import Console

from config import config
console = Console()


HOST = config["host"]
USERNAME = config["username"]
KEY_PATH = config["key_path"]

REMOTE_HOST = config["remote_host"]
REMOTE_PORT = config["remote_port"]

LOCAL_HOST = config["local_host"]
LOCAL_PORT = config["local_port"]


class ForwardServer(threading.Thread):
    def __init__(self, transport):
        super().__init__()
        self.transport = transport
        self.daemon = True

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((LOCAL_HOST, LOCAL_PORT))
        sock.listen(5)

        print(f"\nTunnel active:")
        print(f"http://{LOCAL_HOST}:{LOCAL_PORT} -> {REMOTE_HOST}:{REMOTE_PORT}\n")

        while True:
            client_socket, addr = sock.accept()

            channel = self.transport.open_channel(
                "direct-tcpip",
                (REMOTE_HOST, REMOTE_PORT),
                client_socket.getsockname()
            )

            threading.Thread(
                target=self.handler,
                args=(client_socket, channel),
                daemon=True
            ).start()

    def handler(self, client_socket, channel):
        while True:
            r, w, x = select.select([client_socket, channel], [], [])

            if client_socket in r:
                data = client_socket.recv(1024)
                if len(data) == 0:
                    break
                channel.send(data)

            if channel in r:
                data = channel.recv(1024)
                if len(data) == 0:
                    break
                client_socket.send(data)

        channel.close()
        client_socket.close()


def main():
    passphrase = config["passphrase"]

    key = paramiko.RSAKey.from_private_key_file(
        KEY_PATH,
        password=passphrase
    )

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    print("Connecting to server...")

    client.connect(
        hostname=HOST,
        username=USERNAME,
        pkey=key
    )

    print("SSH connected.")

    transport = client.get_transport()

    server = ForwardServer(transport)
    server.start()

    input("Press ENTER to close tunnel...\n")

    client.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        console.rule(style="bold red")
        console.print("[bold red blink on white]LOST CONNECTION[/bold red blink on white]", justify="center")
        console.rule(style="bold red")

