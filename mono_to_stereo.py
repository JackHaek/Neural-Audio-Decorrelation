with open('2043_orig.wav', 'rb') as f:
    in_data = bytearray(f.read())

in_data[0x16] = 2
in_data[0x17] = 0

byte_rate = int.from_bytes(in_data[0x1C:0x20], 'little')
byte_rate *= 2
in_data[0x1C:0x20] = byte_rate.to_bytes(4, 'little')

bytes_per_frame = int.from_bytes(in_data[0x20:0x22], 'little')
bytes_per_frame *= 2
in_data[0x20:0x22] = bytes_per_frame.to_bytes(2, 'little')

data_size = int.from_bytes(in_data[0x28:0x2C], 'little')
data_size *= 2
in_data[0x28:0x2C] = data_size.to_bytes(4, 'little')

with open('2043_orig_doubled.wav', 'wb') as f:
    f.write(in_data[0:0x2C])
    for i in range(0x2C, len(in_data), 2):
        b1 = in_data[i:i+1]
        b2 = in_data[i+1:i+2]
        f.write(b1)
        f.write(b2)
        f.write(b1)
        f.write(b2)
        