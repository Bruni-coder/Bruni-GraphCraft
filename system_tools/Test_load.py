import os


def ping_server(server_url: str) -> None:
    response = os.system(f"ping -c 4 {server_url}")
    if response == 0:
        print(f"Ping to {server_url} was successful.")
    else:
        print(f"Ping to {server_url} failed.")


ping_server("data.pyg.org")
