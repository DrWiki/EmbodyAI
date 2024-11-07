from DM_CAN import *
import serial
import time

Motor1 = Motor(DM_Motor_Type.DM4340, 0x01, 0x15)
Motor2 = Motor(DM_Motor_Type.DM4340, 0x02, 0x16)
Motor3 = Motor(DM_Motor_Type.DM4340, 0x03, 0x17)
Motor4 = Motor(DM_Motor_Type.DM4310, 0x04, 0x18)
Motor5 = Motor(DM_Motor_Type.DM4310, 0x05, 0x19)
Motor6 = Motor(DM_Motor_Type.DM4310, 0x06, 0x1A)
Motor7 = Motor(DM_Motor_Type.DM4310, 0x07, 0x1B)

serial_device = serial.Serial('COM5', 921600, timeout=0.5)

Motor_Control = MotorControl(serial_device)
Motor_Control.addMotor(Motor1)
Motor_Control.addMotor(Motor2)
Motor_Control.addMotor(Motor3)
Motor_Control.addMotor(Motor4)
Motor_Control.addMotor(Motor5)
Motor_Control.addMotor(Motor6)
Motor_Control.addMotor(Motor7)

Motor_Control.zero_position(Motor1)
time.sleep(0.2)
Motor_Control.zero_position(Motor2)
time.sleep(0.2)
Motor_Control.zero_position(Motor3)
time.sleep(0.2)
Motor_Control.zero_position(Motor4)
time.sleep(0.2)
Motor_Control.zero_position(Motor5)
time.sleep(0.2)
Motor_Control.zero_position(Motor6)
time.sleep(0.2)
Motor_Control.zero_position(Motor7)
time.sleep(0.2)

print("Set Zero Position Successfully!")

# 语句结束关闭串口
serial_device.close()

