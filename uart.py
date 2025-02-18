class Screen:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        self.ser = None
        try:
            import serial
            self.ser = serial.Serial(port, baudrate, timeout=1)
        except Exception as e:
            print(f"Serial init failed: {e}")

    def send(self, message: str, header: int):
        if self.ser is None:
            return

        try:
            data_to_send = message.encode("utf-8")
            header = bytes([header, len(data_to_send)])

            # Send the data
            self.ser.write(header)
            self.ser.write(data_to_send)
        except Exception as e:
            print(f"Serial send: {e}")

    def send_text(self, message: str, speakerid: int, is_translation: bool, is_confirmed: bool):
        header =  16 + \
                speakerid + \
                (int(is_translation) << 1) + \
                (int(is_confirmed) << 2)
        self.send(message, header)

    def close(self):
        if self.ser is None:
            return
        self.ser.close()

