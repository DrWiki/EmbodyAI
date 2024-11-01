
import socket

class UDPCommunication:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_message(self, message):
        self.sock.sendto(message.encode(), (self.ip, self.port))

    def receive_message(self):
        data, addr = self.sock.recvfrom(1024)
        return data.decode()
