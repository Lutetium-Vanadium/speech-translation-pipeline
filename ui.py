import time

MIN_WAIT_DIFF = 0.005
SETUP = 0
SELECT_LANG = 128

UI_CODE_IM_WAITING = 64
UI_EXIT_CHAT = 128
UI_SELECT_LANG_MASK = 0b1100_0000

ALL_LANGUAGES = [
    "en", "zh", "id", "hi",
    "ms", "tl", "vi", "th",
]

class Screen:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200, set_languages=None):
        self.ser = None
        self.last_time_send = 0
        self.state = SETUP
        self.set_languages = set_languages

        try:
            import serial
            self.ser = serial.Serial(port, baudrate, timeout=1)
        except Exception as e:
            print(f"Serial init failed: {e}")

    def send(self, message: str, header: int):
        if self.ser is None:
            return

        diff = time.time() - self.last_time_send
        if diff < MIN_WAIT_DIFF:
            time.sleep(MIN_WAIT_DIFF - diff)

        self.last_time_send = time.time()

        try:
            data_to_send = message.encode("utf-8")
            header = bytes([header, len(data_to_send)])

            # Send the data
            self.ser.write(header)
            self.ser.write(data_to_send)
        except Exception as e:
            print(f"Serial send: {e}")

    def send_text(self, message: str, speakerid: int, is_translation: bool, is_confirmed: bool):
        header = 16 + \
                speakerid + \
                (int(is_translation) << 1) + \
                (int(is_confirmed) << 2)
        self.send(message, header)

    def _set_state(self, new_state):
        self.state = new_state
        self.send("", new_state)

    def send_go_select_lang(self):
        self._set_state(SELECT_LANG)

    def send_go_chat(self, lang1: str, lang2: str):
        lang1_id = ALL_LANGUAGES.index(lang1)
        lang2_id = ALL_LANGUAGES.index(lang2)

        new_state = 0b11_000_000
        new_state |= (lang1_id&8) << 3
        new_state |= (lang2_id&8)

        self._set_state(new_state)

    def close(self):
        if self.ser is None:
            return
        self.ser.close()
    
    def listen(self):
        """ Function to listen for incoming data on the serial port. """
        while True:
            if self.ser is None:
                return

            try:
                header_code, length = self.ser.read(2)
                data = None
                if length > 0:
                    data = self.ser.read(length)

                self.process_received_data(header_code, data)
            except Exception as e:
                print(f"Error while listening: {e}")
            time.sleep(0.1)

    def process_received_data(self, header, data):
        if header == UI_CODE_IM_WAITING:
            if self.state != SETUP:
                self.send("", self.state)

        elif header == UI_EXIT_CHAT:
            self.send_go_select_lang()

        elif (header & UI_SELECT_LANG_MASK) == UI_SELECT_LANG_MASK:
            lang1 = ALL_LANGUAGES[(header >> 3) & 0b111]
            lang2 = ALL_LANGUAGES[(header >> 0) & 0b111]
            if self.set_languages is not None:
                self.set_languages([lang1, lang2])

    def start_listening(self):
        """ Start the listener in a separate thread. """
        listen_thread = threading.Thread(target=self.listen)
        listen_thread.daemon = True  # Ensure the thread exits when the main program exits
        listen_thread.start()

