from DM_CAN import *
import serial
import time

Motor1 = Motor(DM_Motor_Type.DM4340, 0x0F, 0x23)
Motor2 = Motor(DM_Motor_Type.DM4340, 0x10, 0x24)
Motor3 = Motor(DM_Motor_Type.DM4310, 0x11, 0x25)
Motor4 = Motor(DM_Motor_Type.DM4310, 0x12, 0x26)


serial_device = serial.Serial('COM4', 921600, timeout=0.5)

Motor_Control = MotorControl(serial_device)
Motor_Control.addMotor(Motor1)
Motor_Control.addMotor(Motor2)
Motor_Control.addMotor(Motor3)
Motor_Control.addMotor(Motor4)

Motor_Control.zero_position(Motor1)
time.sleep(0.2)
Motor_Control.zero_position(Motor2)
time.sleep(0.2)
Motor_Control.zero_position(Motor3)
time.sleep(0.2)
Motor_Control.zero_position(Motor4)
time.sleep(0.2)

print("Set Zero Position Successfully!")

# 语句结束关闭串口
serial_device.close()

