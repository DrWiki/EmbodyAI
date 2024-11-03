import serial
import time


def Motor_control(position, speed, direction, subdivision_value, ser):
    frame_head = bytes([0x7B])
    device_address = bytes([0x01])
    control_mode = bytes([0x02])
    turn_direction = bytes([direction])
    subdivision_value_bytes = bytes([subdivision_value])

    pos_h, pos_l = divmod(position * 10, 256)
    speed_h, speed_l = divmod(speed * 10, 256)

    data = [frame_head, device_address, control_mode, turn_direction, subdivision_value_bytes,
            bytes([pos_h]), bytes([pos_l]), bytes([speed_h]), bytes([speed_l])]
    rcc_checksum = 0
    for byte in data:
        rcc_checksum ^= byte[0]
    rcc_checksum_bytes = bytes([rcc_checksum])
    frame_tail = bytes([0x7D])

    command = b"".join(data) + rcc_checksum_bytes + frame_tail

    ser.write(command)

    # 发送请求数据指令
    time.sleep(0.005)
    request_command = bytes.fromhex('7b 01 00 00 00 00 00 00 00 7a 7d')
    ser.write(request_command)

    # 等待一段时间确保有足够数据可读（可根据实际情况调整等待时间）
    time.sleep(0.1)

    if ser.in_waiting:
        data = ser.read(ser.in_waiting)
        position_value = ((data[4] << 24) + (data[5] << 16) + (data[6] << 8) + data[7])
        return position_value // 10
    return None


if __name__ == "__main__":
    ser = serial.Serial('COM6', 115200)  # 根据实际情况修改串口号
    # 以顺时针方向（direction=1），20Rad/s的速度，32细分，转动到1872度位置
    position_read = Motor_control(600, 20, 1, 32, ser)
    if position_read is not None:
        print("读取到的位置（十进制）:", position_read)
    else:
        print("未读取到有效位置数据")