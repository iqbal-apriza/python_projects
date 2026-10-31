import serial

class Arduino:
    def __init__(self, port, baud):
        self.arduino = serial.Serial(port=port, baudrate=baud, timeout=1.0)

    def handshake(self):
        self.connected = False

        while not self.connected:
            if self.arduino.in_waiting >= 1:
                data = self.arduino.read(1)

                if data[0] == 0xAA:
                    self.connected = True
                    self.arduino.write(b'o')


    def read_request(self):
        while self.connected:
            data = self.arduino.read(1)

            if not data:
                continue
            
            if data[0] == 0xAB:
                return True


    def send_data(self, cpu, ram, gpu, temp):
        TxData = bytearray(9)

        if cpu is None:
            cpu = 0

        if ram is None:
            ram = 0

        if gpu is None:
            gpu = 0

        if temp is None:
            temp = 0

        cpu = int(round(cpu))
        ram = int(round(ram))
        gpu = int(round(gpu))
        temp = int(temp)

        TxData[0] = 0xFE
        TxData[1] = 0xFD

        TxData[2] = cpu & 0xFF
        TxData[3] = ram & 0xFF
        TxData[4] = gpu & 0xFF

        TxData[5] = temp & 0xFF
        TxData[6] = (temp >> 8) & 0xFF
        TxData[7] = (temp >> 16) & 0xFF

        checksum = 0
        for i in range(2, 8):
            checksum += TxData[i]

        TxData[8] = (~checksum) & 0xFF

        # print(temp)

        self.arduino.write(TxData)


    def cleanup(self):
        self.arduino.close()